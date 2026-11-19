//! saltmill: a vanity salt miner for CREATE2 and CREATE3 addresses.
//!
//! A run resolves the command line into a [`plan::Plan`] (how a salt
//! becomes an address) and [`pattern::Criteria`] (which addresses count),
//! then mines on the GPU, the CPU, or both at once until enough verified
//! hits are found.

#![deny(unused_must_use, rust_2018_idioms)]

pub mod cli;
pub mod cpu;
pub mod gpu;
pub mod keccak;
#[cfg(target_arch = "aarch64")]
mod keccak_x2;
pub mod pattern;
pub mod plan;
pub mod session;

use cli::{Compute, Config};
use plan::Plan;
use session::{Hit, Session, StatusMode};
use std::fs::OpenOptions;
use std::io::IsTerminal;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

/// Run a configuration to completion and return its hits.
pub fn run(config: Config) -> Result<Vec<Hit>, String> {
    if let Some(salt) = config.check {
        return check(&config, salt).map(|hit| vec![hit]);
    }

    let output = match &config.output {
        Some(path) => Some(
            OpenOptions::new()
                .append(true)
                .create(true)
                .open(path)
                .map_err(|e| format!("could not open {}: {e}", path.display()))?,
        ),
        None => None,
    };
    let mode = if config.quiet {
        StatusMode::Quiet
    } else if std::io::stderr().is_terminal() {
        StatusMode::Panel
    } else {
        StatusMode::Lines
    };
    let session = Arc::new(Session::new(
        config.plan.clone(),
        config.criteria,
        config.count,
        output,
        mode,
    ));
    let (cpu_tag, gpu_tag) = tags(&config.plan, config.seed)?;
    let deadline = config
        .time_limit
        .map(|secs| Instant::now() + Duration::from_secs(secs));
    let mut miners = Vec::new();

    let try_gpu = match config.compute {
        Compute::Cpu => false,
        Compute::Gpu | Compute::Hybrid => true,
        Compute::Auto => config.backend != gpu::Backend::Auto || gpu::auto_backend_compiled(),
    };

    // the CPU starts hashing at once; the GPU needs a second to compile
    let cpu_kernel = match config.compute {
        Compute::Gpu => None,
        _ => Some(cpu::resolve_kernel(config.cpu_kernel)?),
    };
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let cpu_label = |threads: usize, kernel: cpu::Kernel| {
        format!("{threads} threads \u{00b7} {}", kernel.label())
    };
    let mut cpu_threads = 0;
    if let Some(kernel) = cpu_kernel {
        // leave a core to the GPU's host thread so dispatches go out
        // without waiting for a time slice
        cpu_threads = config
            .threads
            .unwrap_or((cores - try_gpu as usize).clamp(1, cpu::MAX_THREADS));
        miners.extend(cpu::spawn(
            &session,
            0..cpu_threads,
            cpu_tag.clone(),
            kernel,
        ));
        session.set_miner("cpu", cpu_label(cpu_threads, kernel));
    }

    let mut gpu_starting = None;
    if try_gpu {
        let gpu_config = gpu::GpuConfig {
            backend: config.backend,
            device: config.gpu_device,
            batch_bits: config.gpu_batch_bits,
            tag: gpu_tag,
        };
        let (ready, started) = mpsc::channel();
        let gpu_session = session.clone();
        match std::thread::Builder::new()
            .name("gpu".into())
            .spawn(move || gpu::run(&gpu_session, &gpu_config, ready))
        {
            Ok(handle) => {
                session.set_miner("gpu", "compiling the kernel\u{2026}".into());
                gpu_starting = Some((handle, started));
            }
            Err(e) => session.fail(format!("could not start the GPU thread: {e}")),
        }
    }

    let mut timed_out = false;
    loop {
        // the GPU reports in once its kernel has compiled and passed the
        // self-test; until then the loop keeps the display and the stop
        // conditions live, so an easy target the CPU finishes first does
        // not wait on the compile
        if let Some((handle, started)) = gpu_starting.take() {
            let failure = match started.try_recv() {
                Err(mpsc::TryRecvError::Empty) => {
                    gpu_starting = Some((handle, started));
                    None
                }
                Ok(Ok(description)) => {
                    session.set_miner("gpu", description);
                    miners.push(handle);
                    None
                }
                Ok(Err(e)) => Some(e),
                Err(mpsc::TryRecvError::Disconnected) => {
                    Some("the GPU thread exited before mining".to_string())
                }
            };
            match (failure, cpu_kernel) {
                (None, _) => {}
                (Some(e), Some(kernel)) if config.compute == Compute::Auto => {
                    eprintln!("warning: mining on the CPU only, the GPU is unavailable: {e}");
                    session.set_miner("gpu", format!("unavailable: {e}"));
                    // the core held back for the GPU goes to the CPU miner
                    if config.threads.is_none() && cpu_threads < cores.min(cpu::MAX_THREADS) {
                        let extra = cpu_threads..cpu_threads + 1;
                        miners.extend(cpu::spawn(&session, extra, cpu_tag.clone(), kernel));
                        cpu_threads += 1;
                        session.set_miner("cpu", cpu_label(cpu_threads, kernel));
                    }
                }
                (Some(e), _) => session.fail(e),
            }
        }

        if session.stopped() {
            break;
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            timed_out = true;
            session.stop();
            break;
        }
        session.tick();
        std::thread::sleep(Duration::from_millis(20));
    }
    // a GPU thread still compiling is left behind: it sees the stop flag as
    // soon as it is ready, and ends with the process otherwise
    for miner in miners {
        if miner.join().is_err() {
            session.fail("a mining thread panicked".into());
        }
    }
    let hits = session.finish()?;
    if timed_out && (hits.len() as u64) < config.count {
        return Err(format!(
            "time limit reached with {} of {} hits",
            hits.len(),
            config.count
        ));
    }
    Ok(hits)
}

/// Print the address one salt deploys to.
fn check(config: &Config, salt: [u8; 32]) -> Result<Hit, String> {
    let plan = &config.plan;
    // an unguarded factory derives the same way from any salt; CreateX
    // branches on the salt's first 21 bytes
    if plan.guard != plan::Guard::None && !plan.accepts(&salt) {
        return Err(format!(
            "this salt does not start with the bytes this derivation fixes (0x{}); CreateX \
             would guard it differently, so its address is not computed here",
            keccak::hex(&plan.salt(&vec![0; plan.tag_len()], 0)[..32 - plan.mined_bytes()])
        ));
    }
    let hit = Hit {
        salt,
        address: plan.address(&salt),
    };
    println!("{}", hit.line());
    if !config.criteria.matches(&hit.address) {
        return Err(format!(
            "the address does not meet the target {}",
            config.criteria
        ));
    }
    Ok(hit)
}

/// The salt bytes that keep the CPU's and the GPU's searches apart, and
/// separate one run's search from another's: a random (or seeded) tag whose
/// last bit says which miner it belongs to.
fn tags(plan: &Plan, seed: Option<u32>) -> Result<(Vec<u8>, Vec<u8>), String> {
    let seed = match seed {
        Some(seed) => seed,
        None => {
            let mut bytes = [0u8; 4];
            getrandom::getrandom(&mut bytes)
                .map_err(|e| format!("could not draw a random seed (pass --seed): {e}"))?;
            u32::from_be_bytes(bytes)
        }
    };
    let mut cpu = seed.to_be_bytes()[4 - plan.tag_len()..].to_vec();
    let mut gpu = cpu.clone();
    let last = cpu.len() - 1;
    cpu[last] &= !1;
    gpu[last] |= 1;
    Ok((cpu, gpu))
}

#[cfg(test)]
mod tests {
    use super::*;
    use plan::tests::all_plans;

    #[test]
    fn tags_differ_between_miners_and_follow_the_seed() {
        for plan in all_plans() {
            let (cpu, gpu) = tags(&plan, Some(0xa1b2_c3d4)).unwrap();
            assert_eq!(cpu.len(), plan.tag_len());
            assert_eq!(gpu.len(), plan.tag_len());
            assert_eq!(cpu[cpu.len() - 1], 0xd4);
            assert_eq!(gpu[gpu.len() - 1], 0xd5);
            assert_eq!(cpu[..cpu.len() - 1], gpu[..gpu.len() - 1]);
            assert_eq!(cpu[cpu.len() - 2], 0xc3);
        }
        let plan = all_plans().remove(0);
        assert_ne!(tags(&plan, None).unwrap().0, tags(&plan, None).unwrap().1);
    }
}
