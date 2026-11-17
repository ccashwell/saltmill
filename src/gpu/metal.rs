//! Native Metal compute backend for Apple GPUs.
//!
//! macOS's OpenCL implementation is deprecated and runs as a shim over
//! Metal; driving the same kernel through Metal directly is roughly ten
//! times faster on Apple silicon. Buffers use the unified-memory shared
//! storage mode, so the nonce is written and the solutions are read back
//! without any copies, and each dispatch simply blocks on completion.

use super::{Device, Flavor, Found, Program, MAX_SOLUTIONS};
use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use objc::rc::autoreleasepool;

pub(super) struct MetalDevice {
    device: metal::Device,
}

impl MetalDevice {
    pub(super) fn open(index: usize) -> Result<Self, String> {
        let mut devices = metal::Device::all();
        if devices.is_empty() {
            devices.extend(metal::Device::system_default());
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
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(|e| format!("Metal kernel compilation failed: {e}"))?;
        let function = library
            .get_function("mine", None)
            .map_err(|e| format!("Metal kernel function lookup failed: {e}"))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Metal pipeline creation failed: {e}"))?;

        // unified memory: the host writes inputs and reads solutions directly
        let shared = MTLResourceOptions::StorageModeShared;
        Ok(Box::new(MetalProgram {
            queue: self.device.new_command_queue(),
            nonce: self.device.new_buffer(4, shared),
            count: self.device.new_buffer(4, shared),
            slots: self.device.new_buffer(4 * MAX_SOLUTIONS as u64, shared),
            group: pipeline.max_total_threads_per_threadgroup().min(256),
            pipeline,
        }))
    }
}

struct MetalProgram {
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    nonce: metal::Buffer,
    count: metal::Buffer,
    slots: metal::Buffer,
    group: u64,
}

impl Program for MetalProgram {
    fn dispatch(&mut self, nonce: u32, size: u32) -> Result<Found, String> {
        // SAFETY: both buffers are 4 bytes of shared storage owned by self,
        // and no command buffer is in flight between dispatches
        unsafe {
            *(self.nonce.contents() as *mut u32) = nonce;
            *(self.count.contents() as *mut u32) = 0;
        }

        // command buffers are autoreleased objects; drain each dispatch
        let status = autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(&self.nonce), 0);
            encoder.set_buffer(1, Some(&self.count), 0);
            encoder.set_buffer(2, Some(&self.slots), 0);
            encoder.dispatch_threads(
                MTLSize::new(size as u64, 1, 1),
                MTLSize::new(self.group, 1, 1),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            command_buffer.status()
        });
        if status != MTLCommandBufferStatus::Completed {
            return Err(format!("Metal command buffer ended as {status:?}"));
        }

        // SAFETY: the command buffer has completed, so the GPU no longer
        // writes either buffer; slots holds MAX_SOLUTIONS u32s
        let (count, gids) = unsafe {
            let count = *(self.count.contents() as *const u32);
            let reported = (count as usize).min(MAX_SOLUTIONS);
            let slots = std::slice::from_raw_parts(self.slots.contents() as *const u32, reported);
            (count, slots.to_vec())
        };
        Ok(Found { count, gids })
    }
}
