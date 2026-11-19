//! The GPU miner: generates the kernel for a plan and target, checks it on
//! the device against the reference derivation, then dispatches it in
//! batches of 2^N candidates.
//!
//! The kernel is specialized at build time: every constant byte of every
//! message and the position of the counter are compiled in. The target and
//! a 32-bit nonce are runtime inputs, so the kernel that passes the
//! self-test is the one that mines. A candidate's counter is
//! `gid | nonce << 32`, where `gid` is the thread's position in the batch.

use crate::keccak::{bit_interleave, block_lanes, RC};
use crate::pattern::{Criteria, Pattern};
use crate::plan::{LaneJob, Plan};
use crate::session::Session;
use std::fmt::Write as _;
use std::sync::mpsc::Sender;

#[cfg(target_os = "macos")]
mod metal;
#[cfg(feature = "opencl")]
mod opencl;

static KERNEL_CORE: &str = include_str!("../kernels/keccak_bi_core.cl");
static WRAPPER_METAL: &str = include_str!("../kernels/wrapper.metal");
static WRAPPER_OPENCL: &str = include_str!("../kernels/wrapper.cl");

/// Hits reported per dispatch. More than this in one batch only happens
/// with targets easy enough that losing the surplus does not matter.
pub const MAX_SOLUTIONS: usize = 64;

/// GPU compute backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// Metal on macOS, OpenCL elsewhere
    Auto,
    /// Native Metal (macOS only)
    Metal,
    /// OpenCL (needs a build with `--features opencl`)
    Opencl,
}

/// The shading language a kernel is generated for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavor {
    Metal,
    OpenCl,
}

/// An opened GPU that can build kernels.
pub trait Device {
    /// Backend and device name for the status display.
    fn describe(&self) -> String;
    fn flavor(&self) -> Flavor;
    fn compile(&self, source: &str) -> Result<Box<dyn Program>, String>;
}

/// The kernel's runtime target: a mask and a value for each of the five
/// little-endian address words, then the minimum number of zero bytes.
pub type Target = [u32; 11];

/// A compiled kernel with its buffers.
pub trait Program {
    /// Set what the following dispatches look for.
    fn set_target(&mut self, target: &Target) -> Result<(), String>;
    /// Run `size` candidates under `nonce` and wait for the result.
    fn dispatch(&mut self, nonce: u32, size: u32) -> Result<Found, String>;
}

/// The outcome of one dispatch.
#[derive(Debug, PartialEq, Eq)]
pub struct Found {
    /// How many candidates matched.
    pub count: u32,
    /// The batch positions of the first `MAX_SOLUTIONS` of them, in the
    /// order their threads claimed a slot.
    pub gids: Vec<u32>,
}

/// What the GPU thread needs beyond the session.
#[derive(Clone, Debug)]
pub struct GpuConfig {
    pub backend: Backend,
    pub device: usize,
    /// log2 of the candidates per dispatch
    pub batch_bits: u32,
    pub tag: Vec<u8>,
}

/// Open the configured device.
pub fn open(backend: Backend, index: usize) -> Result<Box<dyn Device>, String> {
    let use_metal = match backend {
        Backend::Metal => true,
        Backend::Opencl => false,
        Backend::Auto => cfg!(target_os = "macos"),
    };

    if use_metal {
        #[cfg(target_os = "macos")]
        return Ok(Box::new(metal::MetalDevice::open(index)?));
        #[cfg(not(target_os = "macos"))]
        return Err("the Metal backend is only available on macOS".into());
    }

    #[cfg(feature = "opencl")]
    return Ok(Box::new(opencl::OpenClDevice::open(index)?));
    #[cfg(not(feature = "opencl"))]
    {
        let _ = index;
        Err("this build has no OpenCL backend (rebuild with `--features opencl`)".into())
    }
}

/// Whether `Backend::Auto` has a backend to try on this build.
pub fn auto_backend_compiled() -> bool {
    cfg!(any(target_os = "macos", feature = "opencl"))
}

/// Mine on the GPU until the session stops. Runs on its own thread; the
/// device description, or the reason the GPU cannot be used, is sent once
/// before the first dispatch.
pub fn run(session: &Session, config: &GpuConfig, ready: Sender<Result<String, String>>) {
    let mut program = match prepare(session, config) {
        Ok((description, program)) => {
            let _ = ready.send(Ok(description));
            program
        }
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };

    let size = 1u32 << config.batch_bits;
    let mut nonce = 0u32;
    while !session.stopped() {
        let found = match program.dispatch(nonce, size) {
            Ok(found) => found,
            Err(e) => {
                session.fail(format!("GPU dispatch failed: {e}"));
                return;
            }
        };
        session.add_tried(size as u64);
        for gid in found.gids {
            let counter = gid as u64 | (nonce as u64) << 32;
            session.submit(session.plan.salt(&config.tag, counter), "gpu");
        }
        nonce = nonce.wrapping_add(1);
    }
}

fn prepare(session: &Session, config: &GpuConfig) -> Result<(String, Box<dyn Program>), String> {
    let device = open(config.backend, config.device)?;
    let job = session.plan.lane_job(&config.tag);
    let mut program = device.compile(&kernel_source(&job, device.flavor()))?;
    self_test(program.as_mut(), &session.plan, &config.tag)
        .map_err(|e| format!("{e} ({})", device.describe()))?;
    program.set_target(&target_words(&session.criteria))?;
    Ok((device.describe(), program))
}

/// Aim the kernel at one known candidate's address and check that the
/// device finds exactly that candidate. A kernel that derived addresses
/// wrongly would otherwise mine forever in silence. Leaves the program's
/// target set to the test address.
pub fn self_test(program: &mut dyn Program, plan: &Plan, tag: &[u8]) -> Result<(), String> {
    const GRID: u32 = 4096;
    const GID: u32 = 0x0abc;
    const NONCE: u32 = 0xa5c3_0f69;

    let counter = GID as u64 | (NONCE as u64) << 32;
    let expected = plan.address(&plan.salt(tag, counter));
    program.set_target(&target_words(&Criteria {
        pattern: Pattern::exact(&expected),
        total_zero_bytes: None,
    }))?;
    let found = program.dispatch(NONCE, GRID)?;
    if found.gids != [GID] {
        return Err(format!(
            "GPU kernel self-test failed: expected candidate {GID:#x} to match, got {:?}",
            found.gids
        ));
    }
    Ok(())
}

/// The criteria in the form the kernel tests.
pub fn target_words(criteria: &Criteria) -> Target {
    let (mask, value) = criteria.pattern.words();
    let mut target = [0u32; 11];
    target[..5].copy_from_slice(&mask);
    target[5..10].copy_from_slice(&value);
    target[10] = criteria.total_zero_bytes.unwrap_or(0) as u32;
    target
}

/// Generate the kernel source (MSL or OpenCL C) for a job.
pub fn kernel_source(job: &LaneJob, flavor: Flavor) -> String {
    let mut src = String::with_capacity(8192 + KERNEL_CORE.len());

    let define_lane = |src: &mut String, name: &str, lane: u64| {
        let (even, odd) = bit_interleave(lane);
        writeln!(src, "#define {name}_E {even}u").unwrap();
        writeln!(src, "#define {name}_O {odd}u").unwrap();
    };

    for (i, &lane) in job.first.iter().enumerate() {
        define_lane(&mut src, &format!("M{i}"), lane);
    }
    writeln!(src, "#define CTR_LANE {}", job.ctr_lane).unwrap();
    writeln!(src, "#define CTR_SHIFT {}", job.ctr_shift).unwrap();

    writeln!(
        src,
        "#define HAS_GUARD {}",
        job.guarded_create2.is_some() as u8
    )
    .unwrap();
    if let Some(create2) = &job.guarded_create2 {
        for (i, &lane) in create2.iter().enumerate() {
            define_lane(&mut src, &format!("P{i}"), lane);
        }
    }

    writeln!(src, "#define HAS_CREATE {}", job.create3 as u8).unwrap();
    if job.create3 {
        // rlp([proxy, 1]) with the proxy zeroed: the list and string
        // prefixes, the nonce, and the padding around them
        let mut message = [0u8; 23];
        message[0] = 0xd6;
        message[1] = 0x94;
        message[22] = 0x01;
        let lanes = block_lanes(&message);
        for i in [0, 2, 16] {
            define_lane(&mut src, &format!("C{i}"), lanes[i]);
        }
    }

    for (i, &rc) in RC.iter().enumerate() {
        define_lane(&mut src, &format!("RC{i}"), rc);
    }
    writeln!(src, "#define MAX_SOLUTIONS {MAX_SOLUTIONS}u").unwrap();

    // splice the shared core into the per-API wrapper
    let wrapper = match flavor {
        Flavor::Metal => WRAPPER_METAL,
        Flavor::OpenCl => WRAPPER_OPENCL,
    };
    let (prelude, entry) = wrapper
        .split_once("//__CORE__//")
        .expect("kernel wrapper is missing the //__CORE__// marker");
    src.push_str(prelude);
    src.push_str(KERNEL_CORE);
    src.push_str(entry);
    src
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::tests::all_plans;

    /// Mirrors the delta-swap networks in keccak_bi_core.cl so the kernel's
    /// runtime interleaving is pinned against the host's.
    fn delta_swap(x: u32, shift: u32, mask: u32) -> u32 {
        let t = (x ^ (x >> shift)) & mask;
        x ^ t ^ (t << shift)
    }

    fn kernel_unshuffle(mut x: u32) -> u32 {
        x = delta_swap(x, 1, 0x22222222);
        x = delta_swap(x, 2, 0x0c0c0c0c);
        x = delta_swap(x, 4, 0x00f000f0);
        x = delta_swap(x, 8, 0x0000ff00);
        x
    }

    fn kernel_shuffle(mut x: u32) -> u32 {
        x = delta_swap(x, 8, 0x0000ff00);
        x = delta_swap(x, 4, 0x00f000f0);
        x = delta_swap(x, 2, 0x0c0c0c0c);
        x = delta_swap(x, 1, 0x22222222);
        x
    }

    fn kernel_interleave(lane: u64) -> (u32, u32) {
        let lo = kernel_unshuffle(lane as u32);
        let hi = kernel_unshuffle((lane >> 32) as u32);
        ((lo & 0xffff) | (hi << 16), (lo >> 16) | (hi & 0xffff0000))
    }

    fn sample_lanes() -> Vec<u64> {
        // deterministic pseudorandom coverage (splitmix64)
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut lanes = vec![0, 1, u64::MAX, 0x8000000080008008];
        for _ in 0..1000 {
            x = x.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            lanes.push(z ^ (z >> 31));
        }
        lanes
    }

    #[test]
    fn kernel_bit_interleaving_matches_reference() {
        for lane in sample_lanes() {
            assert_eq!(
                kernel_interleave(lane),
                bit_interleave(lane),
                "lane {lane:#x}"
            );
            let (e, o) = bit_interleave(lane);
            // the kernel's digest de-interleaving must invert the interleave
            let lo = kernel_shuffle((e & 0xffff) | ((o & 0xffff) << 16));
            let hi = kernel_shuffle((e >> 16) | (o & 0xffff0000));
            assert_eq!(((hi as u64) << 32) | lo as u64, lane, "lane {lane:#x}");
        }
    }

    /// The kernel moves digest bytes between stages by shifting each
    /// interleaved component by half the bit distance.
    #[test]
    fn even_shifts_commute_with_interleaving() {
        for lane in sample_lanes() {
            let (e, o) = bit_interleave(lane);
            for (bits, half) in [(40, 20), (48, 24), (16, 8)] {
                assert_eq!(bit_interleave(lane << bits), (e << half, o << half));
            }
            for (bits, half) in [(24, 12), (16, 8), (32, 16)] {
                assert_eq!(bit_interleave(lane >> bits), (e >> half, o >> half));
            }
        }
    }

    #[test]
    fn kernel_source_embeds_the_job() {
        let plans = all_plans();

        // raw CREATE2: one stage, counter straddling lanes 5 and 6
        let job = plans[0].lane_job(&[0; 4]);
        let src = kernel_source(&job, Flavor::Metal);
        let (e, o) = bit_interleave(job.first[0]);
        assert!(src.contains(&format!("#define M0_E {e}u\n#define M0_O {o}u\n")));
        assert!(src.contains("#define CTR_LANE 5\n#define CTR_SHIFT 40\n"));
        assert!(src.contains("#define HAS_GUARD 0\n"));
        assert!(src.contains("#define HAS_CREATE 0\n"));
        assert!(!src.contains("#define P0_E") && !src.contains("#define C0_E"));
        assert!(src.contains("kernel void mine(") && src.contains("metal_stdlib"));

        // CreateX CREATE3: all three stages
        let job = plans[5].lane_job(&[0; 3]);
        let src = kernel_source(&job, Flavor::OpenCl);
        assert!(src.contains("#define CTR_LANE 7\n#define CTR_SHIFT 0\n"));
        assert!(src.contains("#define HAS_GUARD 1\n"));
        assert!(src.contains("#define HAS_CREATE 1\n"));
        let (e, o) = bit_interleave(0x94d6);
        assert!(src.contains(&format!("#define C0_E {e}u\n#define C0_O {o}u\n")));
        let (e, o) = bit_interleave(0x0101 << 48);
        assert!(src.contains(&format!("#define C2_E {e}u\n#define C2_O {o}u\n")));
        let (e, o) = bit_interleave(RC[23]);
        assert!(src.contains(&format!("#define RC23_E {e}u\n#define RC23_O {o}u\n")));
        assert!(src.contains("__kernel void mine(") && !src.contains("metal_stdlib"));
    }

    #[test]
    fn target_words_are_masks_values_then_zero_bytes() {
        let target = target_words(&Criteria {
            pattern: Pattern::from_hook_flags(0x2400),
            total_zero_bytes: Some(6),
        });
        assert_eq!(target[..4], [0; 4]);
        assert_eq!(target[4], 0xff3f_0000);
        assert_eq!(target[5..9], [0; 4]);
        assert_eq!(target[9], 0x0024_0000);
        assert_eq!(target[10], 6);
        assert_eq!(target_words(&Criteria::default()), [0; 11]);
    }

    #[cfg(any(target_os = "macos", feature = "opencl"))]
    /// Every candidate of a batch, checked on the host: the device must
    /// report exactly the candidates the reference derivation accepts.
    fn check_batch(
        program: &mut dyn Program,
        plan: &Plan,
        tag: &[u8],
        criteria: &Criteria,
        size: u32,
    ) {
        const NONCE: u32 = 0x8001_0203;
        program.set_target(&target_words(criteria)).unwrap();
        let found = program.dispatch(NONCE, size).unwrap();

        let expected: Vec<u32> = (0..size)
            .filter(|&gid| {
                let counter = gid as u64 | (NONCE as u64) << 32;
                criteria.matches(&plan.address(&plan.salt(tag, counter)))
            })
            .collect();
        assert!(
            !expected.is_empty() && expected.len() <= MAX_SOLUTIONS,
            "test batch should have a reportable number of hits, has {}",
            expected.len()
        );
        let mut gids = found.gids.clone();
        gids.sort_unstable();
        assert_eq!(gids, expected, "{:?} {:?}", plan.scheme, plan.guard);
        assert_eq!(found.count as usize, expected.len());
    }

    #[cfg(any(target_os = "macos", feature = "opencl"))]
    fn check_device(device: &dyn Device) {
        let prefix = Criteria {
            pattern: Pattern::from_prefix(&[0xc, 0x0, 0xd]),
            total_zero_bytes: None,
        };
        let suffix_and_zeros = Criteria {
            pattern: Pattern::from_suffix(&[0x7]),
            total_zero_bytes: Some(1),
        };
        for plan in all_plans() {
            let tag: Vec<u8> = (0..plan.tag_len() as u8).map(|i| 0xe1 + i).collect();
            let source = kernel_source(&plan.lane_job(&tag), device.flavor());
            let mut program = device.compile(&source).unwrap();
            self_test(program.as_mut(), &plan, &tag).unwrap();
            check_batch(program.as_mut(), &plan, &tag, &prefix, 1 << 16);
            check_batch(program.as_mut(), &plan, &tag, &suffix_and_zeros, 1 << 12);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_kernel_agrees_with_reference_on_every_plan() {
        match open(Backend::Metal, 0) {
            Ok(device) => check_device(device.as_ref()),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_kernel_agrees_with_reference_on_every_plan() {
        match open(Backend::Opencl, 0) {
            Ok(device) => check_device(device.as_ref()),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn reports_the_true_count_past_the_slot_limit() {
        let Ok(device) = open(Backend::Metal, 0) else {
            return;
        };
        let plan = all_plans().remove(0);
        // one address in sixteen matches: far more hits than slots
        let criteria = Criteria {
            pattern: Pattern::from_prefix(&[0x5]),
            total_zero_bytes: None,
        };
        let tag = vec![0u8; plan.tag_len()];
        let source = kernel_source(&plan.lane_job(&tag), device.flavor());
        let mut program = device.compile(&source).unwrap();
        program.set_target(&target_words(&criteria)).unwrap();
        let found = program.dispatch(7, 1 << 14).unwrap();
        assert_eq!(found.gids.len(), MAX_SOLUTIONS);
        assert!(found.count > 800 && found.count < 1250, "{}", found.count);
        for gid in found.gids {
            let counter = gid as u64 | 7 << 32;
            assert!(criteria.matches(&plan.address(&plan.salt(&tag, counter))));
        }
    }

    /// A self-test must fail when the kernel derives something else: here
    /// it is compiled for one tag and tested against another.
    #[cfg(target_os = "macos")]
    #[test]
    fn self_test_rejects_a_kernel_for_another_derivation() {
        let Ok(device) = open(Backend::Metal, 0) else {
            return;
        };
        let plan = all_plans().remove(5);
        let source = kernel_source(&plan.lane_job(&[1, 2, 3]), device.flavor());
        let mut program = device.compile(&source).unwrap();
        assert!(self_test(program.as_mut(), &plan, &[1, 2, 3]).is_ok());
        let error = self_test(program.as_mut(), &plan, &[1, 2, 4]).unwrap_err();
        assert!(error.contains("self-test failed"), "{error}");
    }
}
