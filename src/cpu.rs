//! The CPU miner: one thread per core, each hashing its own stripe of the
//! counter space.
//!
//! On aarch64 CPUs with the ARMv8.4 SHA3 extension (every Apple M-series
//! chip) each thread derives two candidates at once in the NEON registers;
//! elsewhere it falls back to one candidate at a time through the CRYPTOGAMS
//! assembly keccak.

use crate::session::Session;
use std::sync::Arc;
use std::thread::JoinHandle;

/// Which hashing path the CPU threads use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum CpuKernel {
    /// 2-way SHA3 NEON when the CPU has it, scalar otherwise
    Auto,
    /// One candidate at a time (CRYPTOGAMS assembly keccak)
    Scalar,
    /// Two candidates at a time in the ARMv8.4 SHA3 NEON registers
    Sha3Neon,
}

/// A resolved hashing path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    Sha3Neon,
}

impl Kernel {
    pub fn label(self) -> &'static str {
        match self {
            Kernel::Scalar => "scalar (CRYPTOGAMS asm)",
            Kernel::Sha3Neon => "2-way SHA3 NEON",
        }
    }
}

fn has_sha3_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    return std::arch::is_aarch64_feature_detected!("sha3");
    #[cfg(not(target_arch = "aarch64"))]
    return false;
}

/// Pick the hashing path, refusing an explicit request the CPU cannot run.
pub fn resolve_kernel(requested: CpuKernel) -> Result<Kernel, String> {
    match requested {
        CpuKernel::Scalar => Ok(Kernel::Scalar),
        CpuKernel::Auto if has_sha3_neon() => Ok(Kernel::Sha3Neon),
        CpuKernel::Auto => Ok(Kernel::Scalar),
        CpuKernel::Sha3Neon if has_sha3_neon() => Ok(Kernel::Sha3Neon),
        CpuKernel::Sha3Neon => {
            Err("this CPU has no ARMv8.4 SHA3 extension (use --cpu-kernel scalar)".into())
        }
    }
}

/// The most threads the counter striping supports: the thread index is the
/// counter's top byte.
pub const MAX_THREADS: usize = 256;

/// Candidates between checks of the stop flag and updates of the shared
/// counter, keeping both off the innermost loop.
const CHUNK: u64 = 1 << 14;

/// Start one miner per index on the session, all searching under `tag`.
/// A thread's index selects its stripe of the counter space, so indices
/// must not repeat within a run.
pub fn spawn(
    session: &Arc<Session>,
    indices: std::ops::Range<usize>,
    tag: Vec<u8>,
    kernel: Kernel,
) -> Vec<JoinHandle<()>> {
    assert!(indices.end <= MAX_THREADS);
    indices
        .map(|index| {
            let session = session.clone();
            let tag = tag.clone();
            std::thread::Builder::new()
                .name(format!("cpu-{index}"))
                .spawn(move || worker(&session, &tag, index as u64, kernel))
                .expect("could not start a CPU mining thread")
        })
        .collect()
}

fn worker(session: &Session, tag: &[u8], index: u64, kernel: Kernel) {
    let plan = &session.plan;
    let criteria = &session.criteria;
    // each thread owns the counters whose top byte is its index
    let mut counter = index << 56;

    #[cfg(target_arch = "aarch64")]
    if kernel == Kernel::Sha3Neon {
        let job = plan.lane_job(tag);
        while !session.stopped() {
            for _ in 0..CHUNK / 2 {
                // SAFETY: Kernel::Sha3Neon is only resolved after runtime
                // detection of the sha3 feature
                let (first, second) =
                    unsafe { crate::keccak_x2::address_pair(&job, counter, counter + 1) };
                if criteria.matches(&first) {
                    session.submit(plan.salt(tag, counter), "cpu");
                }
                if criteria.matches(&second) {
                    session.submit(plan.salt(tag, counter + 1), "cpu");
                }
                counter = counter.wrapping_add(2);
            }
            session.add_tried(CHUNK);
        }
        return;
    }

    debug_assert_eq!(kernel, Kernel::Scalar);
    while !session.stopped() {
        for _ in 0..CHUNK {
            let salt = plan.salt(tag, counter);
            if criteria.matches(&plan.address(&salt)) {
                session.submit(salt, "cpu");
            }
            counter = counter.wrapping_add(1);
        }
        session.add_tried(CHUNK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::{Criteria, Pattern};
    use crate::plan::tests::all_plans;
    use crate::session::StatusMode;

    /// Mine an easy target on every plan shape with each available kernel
    /// and check the hits against the reference derivation.
    #[test]
    fn mines_verified_hits_on_every_plan() {
        let mut kernels = vec![Kernel::Scalar];
        if has_sha3_neon() {
            kernels.push(Kernel::Sha3Neon);
        }
        let criteria = Criteria {
            pattern: Pattern::from_prefix(&[0xa, 0xb]),
            total_zero_bytes: None,
        };

        for plan in all_plans() {
            for &kernel in &kernels {
                let tag = vec![0x5a; plan.tag_len()];
                let session = Arc::new(Session::new(
                    plan.clone(),
                    criteria,
                    3,
                    None,
                    StatusMode::Quiet,
                ));
                for handle in spawn(&session, 0..2, tag.clone(), kernel) {
                    handle.join().unwrap();
                }

                let hits = session.finish().unwrap();
                assert_eq!(hits.len(), 3, "{kernel:?}");
                for hit in hits {
                    assert_eq!(hit.address[0], 0xab);
                    assert_eq!(hit.address, plan.address(&hit.salt));
                    assert!(plan.accepts(&hit.salt));
                    assert_eq!(hit.salt[24 - tag.len()..24], tag[..]);
                    assert!(hit.salt[31] < 2, "counter stripe is the thread index");
                }
            }
        }
    }

    #[test]
    fn explicit_kernel_requests_are_honoured_or_refused() {
        assert_eq!(resolve_kernel(CpuKernel::Scalar), Ok(Kernel::Scalar));
        assert_eq!(resolve_kernel(CpuKernel::Sha3Neon).is_ok(), has_sha3_neon());
        assert!(resolve_kernel(CpuKernel::Auto).is_ok());
    }
}
