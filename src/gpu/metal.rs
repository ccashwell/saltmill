//! Native Metal compute backend for Apple GPUs.
//!
//! macOS's OpenCL implementation is deprecated and runs as a shim over
//! Metal; driving the same kernel through Metal directly is roughly ten
//! times faster on Apple silicon. Buffers use the unified-memory shared
//! storage mode, so the inputs are written and the solutions are read back
//! without any copies, and each dispatch simply blocks on completion.

use super::{Device, Flavor, Found, Program, Target, MAX_SOLUTIONS};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCopyAllDevices,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

/// The kernel's input buffer: the nonce and the eleven target words.
const INPUT_WORDS: usize = 12;

type Object<T> = Retained<ProtocolObject<T>>;

pub(super) struct MetalDevice {
    device: Object<dyn MTLDevice>,
}

impl MetalDevice {
    pub(super) fn open(index: usize) -> Result<Self, String> {
        let mut devices = MTLCopyAllDevices().to_vec();
        if devices.is_empty() {
            devices.extend(MTLCreateSystemDefaultDevice());
        }
        if index >= devices.len() {
            return Err(match devices.len() {
                0 => "no Metal device available".to_string(),
                n => format!("no Metal device at index {index} ({n} available)"),
            });
        }
        Ok(Self {
            device: devices.swap_remove(index),
        })
    }

    fn buffer(&self, words: usize) -> Result<Object<dyn MTLBuffer>, String> {
        // unified memory: the host writes inputs and reads solutions directly
        self.device
            .newBufferWithLength_options(4 * words, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| "Metal buffer allocation failed".to_string())
    }
}

impl Device for MetalDevice {
    fn describe(&self) -> String {
        format!("Metal \u{00b7} {}", self.device.name())
    }

    fn flavor(&self) -> Flavor {
        Flavor::Metal
    }

    fn compile(&self, source: &str) -> Result<Box<dyn Program>, String> {
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), None)
            .map_err(|e| {
                format!(
                    "Metal kernel compilation failed: {}",
                    e.localizedDescription()
                )
            })?;
        let function = library
            .newFunctionWithName(&NSString::from_str("mine"))
            .ok_or("Metal kernel has no `mine` function")?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                format!(
                    "Metal pipeline creation failed: {}",
                    e.localizedDescription()
                )
            })?;

        Ok(Box::new(MetalProgram {
            queue: self
                .device
                .newCommandQueue()
                .ok_or("Metal command queue creation failed")?,
            input: self.buffer(INPUT_WORDS)?,
            count: self.buffer(1)?,
            slots: self.buffer(MAX_SOLUTIONS)?,
            group: pipeline.maxTotalThreadsPerThreadgroup().min(256),
            pipeline,
        }))
    }
}

struct MetalProgram {
    queue: Object<dyn MTLCommandQueue>,
    pipeline: Object<dyn MTLComputePipelineState>,
    /// the dispatch nonce, then the target words
    input: Object<dyn MTLBuffer>,
    count: Object<dyn MTLBuffer>,
    slots: Object<dyn MTLBuffer>,
    group: usize,
}

impl Program for MetalProgram {
    fn set_target(&mut self, target: &Target) -> Result<(), String> {
        // SAFETY: input holds INPUT_WORDS u32s of shared storage owned by
        // self, and no command buffer is in flight between dispatches
        unsafe {
            let words = self.input.contents().cast::<u32>().as_ptr();
            std::ptr::copy_nonoverlapping(target.as_ptr(), words.add(1), target.len());
        }
        Ok(())
    }

    fn dispatch(&mut self, nonce: u32, size: u32) -> Result<Found, String> {
        // SAFETY: as in set_target; count is 4 bytes of shared storage
        unsafe {
            self.input.contents().cast::<u32>().as_ptr().write(nonce);
            self.count.contents().cast::<u32>().as_ptr().write(0);
        }

        // command buffers and encoders are autoreleased; drain each dispatch
        let status = autoreleasepool(|_| {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or("Metal command buffer creation failed")?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or("Metal compute encoder creation failed")?;
            encoder.setComputePipelineState(&self.pipeline);
            // SAFETY: each buffer outlives the command buffer, which has
            // completed before this function returns, and the kernel stays
            // inside them: it reads INPUT_WORDS words of the first, and
            // writes one word of the second and at most MAX_SOLUTIONS words
            // of the third
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.input), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.count), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&self.slots), 0, 2);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: size as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.group,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            Ok::<_, String>(command_buffer.status())
        })?;
        if status != MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer ended as {status:?}"));
        }

        // SAFETY: the command buffer has completed, so the GPU no longer
        // writes either buffer; slots holds MAX_SOLUTIONS u32s
        let (count, gids) = unsafe {
            let count = self.count.contents().cast::<u32>().as_ptr().read();
            let reported = (count as usize).min(MAX_SOLUTIONS);
            let slots =
                std::slice::from_raw_parts(self.slots.contents().cast::<u32>().as_ptr(), reported);
            (count, slots.to_vec())
        };
        Ok(Found { count, gids })
    }
}
