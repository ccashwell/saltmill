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
use std::time::Duration;

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
    let mut miners = Vec::new();

    // the GPU first: whether it comes up decides how many cores the CPU
    // miner takes
    let try_gpu = match config.compute {
        Compute::Cpu => false,
        Compute::Gpu | Compute::Hybrid => true,
        Compute::Auto => config.backend != gpu::Backend::Auto || gpu::auto_backend_compiled(),
    };
    let mut gpu_running = false;
    if try_gpu {
        let gpu_config = gpu::GpuConfig {
            backend: config.backend,
            device: config.gpu_device,
            batch_bits: config.gpu_batch_bits,
            tag: gpu_tag,
        };
        let (ready, started) = mpsc::channel();
        let gpu_session = session.clone();
        let handle = std::thread::Builder::new()
            .name("gpu".into())
            .spawn(move || gpu::run(&gpu_session, &gpu_config, ready))
            .map_err(|e| format!("could not start the GPU thread: {e}"))?;
        let outcome = started
            .recv()
            .unwrap_or_else(|_| Err("the GPU thread exited before it started mining".into()));
        match outcome {
            Ok(description) => {
                session.add_miner("gpu", description);
                miners.push(handle);
                gpu_running = true;
            }
            Err(e) if config.compute == Compute::Auto => {
                eprintln!("warning: mining on the CPU only, the GPU is unavailable: {e}");
            }
            Err(e) => return Err(e),
        }
    }

    if config.compute != Compute::Gpu {
        let kernel = cpu::resolve_kernel(config.cpu_kernel)?;
        let threads = config.threads.unwrap_or_else(|| {
            let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
            // leave a core to the GPU's host thread so dispatches go out
            // without waiting for a time slice
            (cores - gpu_running as usize).clamp(1, cpu::MAX_THREADS)
        });
        session.add_miner(
            "cpu",
            format!("{threads} threads \u{00b7} {}", kernel.label()),
        );
        miners.extend(cpu::spawn(&session, threads, cpu_tag, kernel));
    }

    while !session.stopped() {
        session.tick();
        std::thread::sleep(Duration::from_millis(50));
    }
    for miner in miners {
        if miner.join().is_err() {
            session.fail("a mining thread panicked".into());
        }
    }
    session.finish()
}

/// Print the address one salt deploys to.
fn check(config: &Config, salt: [u8; 32]) -> Result<Hit, String> {
    let plan = &config.plan;
    if !plan.accepts(&salt) {
        return Err(format!(
            "this salt does not start with the bytes this derivation fixes (0x{}); the factory \
             would treat it differently, so its address is not computed here",
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
