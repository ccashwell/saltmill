//! OpenCL compute backend, for GPUs Metal does not cover.
//!
//! It runs the same bit-interleaved kernel as the Metal backend. On macOS
//! OpenCL is a deprecated shim over Metal and about ten times slower than
//! the native backend, so there it is only useful for checking the kernel.

use super::{Device, Flavor, Found, Program, MAX_SOLUTIONS};
use ocl::{Buffer, Context, Kernel, MemFlags, Platform, Queue};

pub(super) struct OpenClDevice {
    device: ocl::Device,
    context: Context,
    queue: Queue,
}

impl OpenClDevice {
    pub(super) fn open(index: usize) -> Result<Self, String> {
        let open = || -> ocl::Result<Self> {
            let platform = Platform::new(ocl::core::default_platform()?);
            let device = ocl::Device::by_idx_wrap(platform, index)?;
            let context = Context::builder()
                .platform(platform)
                .devices(device)
                .build()?;
            let queue = Queue::new(&context, device, None)?;
            Ok(Self {
                device,
                context,
                queue,
            })
        };
        open().map_err(|e| format!("could not open OpenCL device {index}: {e}"))
    }
}

impl Device for OpenClDevice {
    fn describe(&self) -> String {
        let name = self
            .device
            .name()
            .unwrap_or_else(|_| "unknown device".to_string());
        format!("OpenCL \u{00b7} {name}")
    }

    fn flavor(&self) -> Flavor {
        Flavor::OpenCl
    }

    fn compile(&self, source: &str) -> Result<Box<dyn Program>, String> {
        let compile = || -> ocl::Result<OpenClProgram> {
            let program = ocl::Program::builder()
                .devices(self.device)
                .src(source)
                .build(&self.context)?;
            let buffer = |len: usize, flags: MemFlags| {
                Buffer::<u32>::builder()
                    .queue(self.queue.clone())
                    .flags(flags)
                    .len(len)
                    .build()
            };
            let nonce = buffer(1, MemFlags::new().read_only())?;
            let count = buffer(1, MemFlags::new().read_write())?;
            let slots = buffer(MAX_SOLUTIONS, MemFlags::new().write_only())?;
            let kernel = Kernel::builder()
                .program(&program)
                .name("mine")
                .queue(self.queue.clone())
                .arg(&nonce)
                .arg(&count)
                .arg(&slots)
                .build()?;
            Ok(OpenClProgram {
                queue: self.queue.clone(),
                kernel,
                nonce,
                count,
                slots,
            })
        };
        compile()
            .map(|program| Box::new(program) as Box<dyn Program>)
            .map_err(|e| format!("OpenCL kernel build failed: {e}"))
    }
}

struct OpenClProgram {
    queue: Queue,
    kernel: Kernel,
    nonce: Buffer<u32>,
    count: Buffer<u32>,
    slots: Buffer<u32>,
}

impl Program for OpenClProgram {
    fn dispatch(&mut self, nonce: u32, size: u32) -> Result<Found, String> {
        let dispatch = || -> ocl::Result<Found> {
            self.nonce.write(&[nonce][..]).enq()?;
            self.count.write(&[0u32][..]).enq()?;
            // SAFETY: the kernel only writes the count and slots buffers,
            // within their bounds (slot indices are checked against
            // MAX_SOLUTIONS in the kernel)
            unsafe {
                self.kernel.cmd().global_work_size(size as usize).enq()?;
            }
            self.queue.finish()?;

            let mut count = [0u32];
            self.count.read(&mut count[..]).enq()?;
            let mut slots = vec![0u32; MAX_SOLUTIONS];
            self.slots.read(&mut slots).enq()?;
            slots.truncate((count[0] as usize).min(MAX_SOLUTIONS));
            Ok(Found {
                count: count[0],
                gids: slots,
            })
        };
        dispatch().map_err(|e| e.to_string())
    }
}
