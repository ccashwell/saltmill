//! Command-line flags and their resolution into a run configuration.

use crate::cpu::{CpuKernel, MAX_THREADS};
use crate::gpu::Backend;
use crate::keccak::{keccak256, parse_fixed};
use crate::pattern::{parse_nibbles, Criteria, Pattern, HOOK_FLAG_MASK};
use crate::plan::{Guard, Plan, Scheme, CREATE3_PROXY_INIT_CODE, CREATEX};
use clap::Parser;
use std::path::PathBuf;

/// Mine salts for vanity CREATE2 and CREATE3 contract addresses.
///
/// Give the derivation (--scheme and the addresses it depends on) and a
/// target (--prefix, --suffix, --hook-flags, zero bytes; all given must
/// hold). Each hit is printed to stdout as `<salt> <address>`; progress goes
/// to stderr. By default the GPU and every CPU core mine together.
///
/// CREATE2 through any factory:
///   saltmill --scheme create2 --factory <F> --init-code-hash <H> --prefix c0ffee
///
/// CREATE3 through CreateX (same address on every chain, for any init code):
///   saltmill --scheme create3 --caller <DEPLOYER> --prefix 0000000000
#[derive(Parser, Debug)]
#[command(name = "saltmill", version, verbatim_doc_comment)]
pub struct Args {
    /// How the contract is created
    #[arg(short, long, value_enum, help_heading = "Derivation")]
    pub scheme: Scheme,

    /// The contract that executes CREATE2. Defaults to CreateX with the
    /// CreateX guard; required otherwise
    #[arg(short, long, value_name = "ADDRESS", value_parser = parse_fixed::<20>, help_heading = "Derivation")]
    pub factory: Option<[u8; 20]>,

    /// The account that will call the factory. It becomes the first 20 bytes
    /// of the salt (CreateX and many factories only accept such a salt from
    /// that account, which stops others from taking the address). Required
    /// with the CreateX guard, zero otherwise
    #[arg(short, long, visible_alias = "deployer", value_name = "ADDRESS", value_parser = parse_fixed::<20>, help_heading = "Derivation")]
    pub caller: Option<[u8; 20]>,

    /// keccak256 of the init code, constructor arguments included (CREATE2
    /// only; a CREATE3 address does not depend on the init code)
    #[arg(short, long, value_name = "HASH", value_parser = parse_fixed::<32>, help_heading = "Derivation")]
    pub init_code_hash: Option<[u8; 32]>,

    /// What the factory does to the salt before CREATE2. Defaults to
    /// `createx` when the factory is CreateX, `none` otherwise
    #[arg(long, value_enum, help_heading = "Derivation")]
    pub guard: Option<GuardArg>,

    /// Bind the address to one chain with CreateX's cross-chain redeploy
    /// protection: salt byte 20 becomes 0x01 and the guard hashes this chain
    /// id. Without it the same salt gives the same address on every chain
    #[arg(long, value_name = "ID", help_heading = "Derivation")]
    pub chain_id: Option<u64>,

    /// keccak256 of the CREATE3 proxy init code, for factories that do not
    /// use the proxy CreateX, solmate and Solady share
    #[arg(long, value_name = "HASH", value_parser = parse_fixed::<32>, help_heading = "Derivation")]
    pub proxy_init_code_hash: Option<[u8; 32]>,

    /// Address must start with these hex characters (odd lengths allowed)
    #[arg(long, value_name = "HEX", help_heading = "Target")]
    pub prefix: Option<String>,

    /// Address must end with these hex characters (odd lengths allowed)
    #[arg(long, value_name = "HEX", help_heading = "Target")]
    pub suffix: Option<String>,

    /// Uniswap v4 hook permission flags: address & 0x3fff must equal FLAGS
    /// exactly (hex or decimal)
    #[arg(long, value_name = "FLAGS", value_parser = parse_hook_flags, help_heading = "Target")]
    pub hook_flags: Option<u16>,

    /// Address must start with at least this many zero bytes
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u8).range(0..=20), help_heading = "Target")]
    pub leading_zero_bytes: Option<u8>,

    /// Address must contain at least this many zero bytes anywhere
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u8).range(0..=20), help_heading = "Target")]
    pub total_zero_bytes: Option<u8>,

    /// Stop after this many hits (0 = mine until interrupted)
    #[arg(
        short = 'n',
        long,
        default_value_t = 1,
        value_name = "N",
        help_heading = "Run"
    )]
    pub count: u64,

    /// Give up after this many seconds
    #[arg(long, value_name = "SECONDS", help_heading = "Run")]
    pub time_limit: Option<u64>,

    /// Also append each hit to this file
    #[arg(short, long, value_name = "FILE", help_heading = "Run")]
    pub output: Option<PathBuf>,

    /// Print the address a salt deploys to and exit. With a target, fail
    /// unless the address meets it
    #[arg(long, value_name = "SALT", value_parser = parse_fixed::<32>, help_heading = "Run")]
    pub check: Option<[u8; 32]>,

    /// Fix the salt bytes that distinguish one run's search space from
    /// another's (hex, up to 4 bytes) instead of drawing them at random
    #[arg(long, value_name = "HEX", value_parser = parse_seed, help_heading = "Run")]
    pub seed: Option<u32>,

    /// Print hits and errors only
    #[arg(short, long, help_heading = "Run")]
    pub quiet: bool,

    /// Which processors mine. `auto` is the GPU plus the CPU, or the CPU
    /// alone when no GPU can be used
    #[arg(long, value_enum, default_value_t = Compute::Auto, help_heading = "Compute")]
    pub compute: Compute,

    /// CPU mining threads (default: every core, less one for the GPU's host
    /// thread when the GPU mines too)
    #[arg(short, long, value_name = "N", value_parser = parse_threads, help_heading = "Compute")]
    pub threads: Option<usize>,

    /// CPU hashing path
    #[arg(long, value_enum, default_value_t = CpuKernel::Auto, help_heading = "Compute")]
    pub cpu_kernel: CpuKernel,

    /// GPU backend
    #[arg(long, value_enum, default_value_t = Backend::Auto, help_heading = "Compute")]
    pub backend: Backend,

    /// Index of the GPU to use
    #[arg(
        short,
        long,
        default_value_t = 0,
        value_name = "INDEX",
        help_heading = "Compute"
    )]
    pub gpu_device: usize,

    /// log2 of the candidates per GPU dispatch. Larger batches waste less
    /// time between dispatches but take longer to notice a stop
    #[arg(long, default_value_t = 26, value_name = "BITS", value_parser = clap::value_parser!(u32).range(10..=30), help_heading = "Compute")]
    pub gpu_batch_bits: u32,
}

/// The salt transformation a factory applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum GuardArg {
    /// The salt is used as given
    None,
    /// CreateX: the salt is hashed with the caller (and --chain-id)
    Createx,
}

/// Which processors mine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Compute {
    /// GPU and CPU when a GPU is usable, CPU otherwise
    Auto,
    /// CPU only
    Cpu,
    /// GPU only, leaving the CPU free
    Gpu,
    /// GPU and CPU together; fail if the GPU is unusable
    Hybrid,
}

fn parse_hook_flags(s: &str) -> Result<u16, String> {
    let flags = match s.strip_prefix("0x") {
        Some(hex) => u16::from_str_radix(hex, 16),
        None => s.parse::<u16>(),
    }
    .map_err(|e| e.to_string())?;
    if flags > HOOK_FLAG_MASK {
        return Err(format!(
            "hook flags must fit in 14 bits (max 0x{HOOK_FLAG_MASK:x})"
        ));
    }
    Ok(flags)
}

fn parse_seed(s: &str) -> Result<u32, String> {
    let digits = s.strip_prefix("0x").unwrap_or(s);
    if digits.is_empty() || digits.len() > 8 {
        return Err("expected 1 to 8 hex characters".into());
    }
    u32::from_str_radix(digits, 16).map_err(|e| e.to_string())
}

fn parse_threads(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(n) if (1..=MAX_THREADS).contains(&n) => Ok(n),
        _ => Err(format!("expected a thread count from 1 to {MAX_THREADS}")),
    }
}

/// A validated run: what to derive, what to look for, and on what.
#[derive(Clone, Debug)]
pub struct Config {
    pub plan: Plan,
    pub criteria: Criteria,
    pub count: u64,
    pub time_limit: Option<u64>,
    pub output: Option<PathBuf>,
    pub check: Option<[u8; 32]>,
    pub seed: Option<u32>,
    pub quiet: bool,
    pub compute: Compute,
    pub threads: Option<usize>,
    pub cpu_kernel: CpuKernel,
    pub backend: Backend,
    pub gpu_device: usize,
    pub gpu_batch_bits: u32,
}

impl Config {
    /// Parse the process arguments (exiting with usage on a parse error)
    /// and validate them.
    pub fn parse_args() -> Result<Self, String> {
        Self::new(Args::parse())
    }

    /// Validate the arguments: the derivation must be fully specified and
    /// free of flags that would be silently ignored.
    pub fn new(args: Args) -> Result<Self, String> {
        let guard = match args.guard {
            Some(GuardArg::None) => Guard::None,
            Some(GuardArg::Createx) => Guard::CreateX {
                chain_id: args.chain_id,
            },
            // CreateX always guards, so a salt mined without the guard
            // would not deploy where the miner said
            None if args
                .factory
                .map_or(args.scheme == Scheme::Create3, |f| f == CREATEX) =>
            {
                Guard::CreateX {
                    chain_id: args.chain_id,
                }
            }
            None => Guard::None,
        };

        let (factory, caller) = match guard {
            Guard::CreateX { .. } => (
                args.factory.unwrap_or(CREATEX),
                args.caller.ok_or(
                    "--caller is required with the CreateX guard: CreateX hashes the calling \
                     account into the address",
                )?,
            ),
            Guard::None => {
                if args.chain_id.is_some() {
                    return Err("--chain-id only applies to the CreateX guard".into());
                }
                (
                    args.factory.ok_or(
                        "--factory is required: the contract that executes CREATE2 (omit \
                         --guard none to mine for CreateX)",
                    )?,
                    args.caller.unwrap_or_default(),
                )
            }
        };

        let init_code_hash = match args.scheme {
            Scheme::Create2 => {
                if args.proxy_init_code_hash.is_some() {
                    return Err("--proxy-init-code-hash only applies to --scheme create3".into());
                }
                args.init_code_hash
                    .ok_or("--init-code-hash is required with --scheme create2")?
            }
            Scheme::Create3 => {
                if args.init_code_hash.is_some() {
                    return Err(
                        "--init-code-hash does not apply to --scheme create3: the address does \
                         not depend on the contract's init code"
                            .into(),
                    );
                }
                args.proxy_init_code_hash
                    .unwrap_or_else(|| keccak256(&CREATE3_PROXY_INIT_CODE))
            }
        };

        let mut pattern = Pattern::default();
        if let Some(prefix) = &args.prefix {
            pattern = pattern.merge(Pattern::from_prefix(&parse_nibbles(prefix, "prefix")?))?;
        }
        if let Some(suffix) = &args.suffix {
            pattern = pattern.merge(Pattern::from_suffix(&parse_nibbles(suffix, "suffix")?))?;
        }
        if let Some(flags) = args.hook_flags {
            pattern = pattern.merge(Pattern::from_hook_flags(flags))?;
        }
        if let Some(count) = args.leading_zero_bytes {
            pattern = pattern.merge(Pattern::from_leading_zero_bytes(count as usize))?;
        }
        let criteria = Criteria {
            pattern,
            total_zero_bytes: args.total_zero_bytes.filter(|&n| n > 0),
        };
        if args.check.is_none() {
            if criteria.is_empty() {
                return Err("no target given: use --prefix, --suffix, --hook-flags, \
                     --leading-zero-bytes or --total-zero-bytes"
                    .into());
            }
            if criteria.difficulty_bits().is_infinite() {
                return Err(
                    "no address can meet this target: the pattern fixes too many bytes to \
                     non-zero values for --total-zero-bytes"
                        .into(),
                );
            }
        }

        Ok(Self {
            plan: Plan::new(args.scheme, guard, factory, caller, init_code_hash),
            criteria,
            count: args.count,
            time_limit: args.time_limit,
            output: args.output,
            check: args.check,
            seed: args.seed,
            quiet: args.quiet,
            compute: args.compute,
            threads: args.threads,
            cpu_kernel: args.cpu_kernel,
            backend: args.backend,
            gpu_device: args.gpu_device,
            gpu_batch_bits: args.gpu_batch_bits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::tests::CALLER;

    const FACTORY: &str = "0x4e59b44847b379578588920cA78FbF26c0B4956C";
    const HASH: &str = "0x2222222222222222222222222222222222222222222222222222222222222222";

    fn config(flags: &[&str]) -> Result<Config, String> {
        let args =
            Args::try_parse_from(["saltmill"].iter().chain(flags)).map_err(|e| e.to_string())?;
        Config::new(args)
    }

    #[test]
    fn create2_needs_a_factory_and_an_init_code_hash() {
        let ok = config(&[
            "--scheme",
            "create2",
            "--factory",
            FACTORY,
            "-i",
            HASH,
            "--prefix",
            "c0ffee",
        ])
        .unwrap();
        assert_eq!(ok.plan.scheme, Scheme::Create2);
        assert_eq!(ok.plan.guard, Guard::None);
        assert_eq!(ok.plan.caller, [0u8; 20]);
        assert_eq!(ok.plan.init_code_hash, [0x22; 32]);
        assert_eq!((ok.count, ok.compute), (1, Compute::Auto));

        assert!(config(&["-s", "create2", "-i", HASH, "--prefix", "00"])
            .unwrap_err()
            .contains("--factory is required"));
        assert!(config(&["-s", "create2", "-f", FACTORY, "--prefix", "00"])
            .unwrap_err()
            .contains("--init-code-hash is required"));
    }

    #[test]
    fn create3_defaults_to_createx_and_needs_the_caller() {
        let ok = config(&[
            "--scheme",
            "create3",
            "--deployer",
            CALLER,
            "--suffix",
            "4444",
        ])
        .unwrap();
        assert_eq!(ok.plan.scheme, Scheme::Create3);
        assert_eq!(ok.plan.guard, Guard::CreateX { chain_id: None });
        assert_eq!(ok.plan.factory, CREATEX);
        assert_eq!(ok.plan.init_code_hash, keccak256(&CREATE3_PROXY_INIT_CODE));

        assert!(config(&["-s", "create3", "--suffix", "4444"])
            .unwrap_err()
            .contains("--caller is required"));
        assert!(
            config(&["-s", "create3", "-c", CALLER, "-i", HASH, "--suffix", "44"])
                .unwrap_err()
                .contains("does not apply")
        );

        let bound = config(&[
            "-s",
            "create3",
            "-c",
            CALLER,
            "--chain-id",
            "8453",
            "--suffix",
            "44",
        ])
        .unwrap();
        assert_eq!(
            bound.plan.guard,
            Guard::CreateX {
                chain_id: Some(8453)
            }
        );
    }

    #[test]
    fn guard_follows_the_factory_unless_overridden() {
        let createx = "0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed";
        // CREATE2 through CreateX is guarded too
        let guarded = config(&[
            "-s", "create2", "-f", createx, "-c", CALLER, "-i", HASH, "--prefix", "00",
        ])
        .unwrap();
        assert_eq!(guarded.plan.guard, Guard::CreateX { chain_id: None });

        // CREATE3 through another factory uses the salt as given
        let raw = config(&["-s", "create3", "-f", FACTORY, "--prefix", "00"]).unwrap();
        assert_eq!(raw.plan.guard, Guard::None);

        let forced = config(&[
            "-s", "create3", "-f", FACTORY, "-c", CALLER, "--guard", "createx", "--prefix", "00",
        ])
        .unwrap();
        assert_eq!(forced.plan.guard, Guard::CreateX { chain_id: None });
        assert_eq!(forced.plan.factory, parse_fixed::<20>(FACTORY).unwrap());

        assert!(config(&[
            "-s",
            "create3",
            "-f",
            FACTORY,
            "--chain-id",
            "1",
            "--prefix",
            "00",
        ])
        .unwrap_err()
        .contains("--chain-id only applies"));
    }

    #[test]
    fn targets_combine_and_conflicts_are_rejected() {
        let base = ["-s", "create2", "-f", FACTORY, "-i", HASH];
        let with = |extra: &[&str]| config(&[&base[..], extra].concat());

        let combined = with(&["--prefix", "c0ffee", "--hook-flags", "0x00c0"]).unwrap();
        assert_eq!(combined.criteria.pattern.bits(), 24 + 14);

        let zeros = with(&["--leading-zero-bytes", "3", "--total-zero-bytes", "5"]).unwrap();
        assert_eq!(zeros.criteria.pattern.bits(), 24);
        assert_eq!(zeros.criteria.total_zero_bytes, Some(5));

        assert!(with(&[]).unwrap_err().contains("no target given"));
        assert!(with(&["--prefix", "ff", "--leading-zero-bytes", "1"])
            .unwrap_err()
            .contains("conflicting"));
        assert!(with(&["--hook-flags", "0x4000"]).is_err());
        assert!(with(&["--leading-zero-bytes", "21"]).is_err());
        assert!(
            with(&["--prefix", &"1".repeat(40), "--total-zero-bytes", "1"])
                .unwrap_err()
                .contains("no address can meet")
        );
    }

    #[test]
    fn check_needs_no_target() {
        let salt = format!("0x{}", "00".repeat(32));
        let checked =
            config(&["-s", "create2", "-f", FACTORY, "-i", HASH, "--check", &salt]).unwrap();
        assert_eq!(checked.check, Some([0u8; 32]));
        assert!(checked.criteria.is_empty());
    }

    #[test]
    fn compute_flags_are_validated() {
        let base = ["-s", "create2", "-f", FACTORY, "-i", HASH, "--prefix", "00"];
        let with = |extra: &[&str]| config(&[&base[..], extra].concat());

        let tuned = with(&[
            "--compute",
            "hybrid",
            "--threads",
            "4",
            "--cpu-kernel",
            "scalar",
            "--backend",
            "metal",
            "--gpu-batch-bits",
            "24",
            "--seed",
            "0xabc",
            "-n",
            "0",
        ])
        .unwrap();
        assert_eq!(tuned.compute, Compute::Hybrid);
        assert_eq!(tuned.threads, Some(4));
        assert_eq!(tuned.cpu_kernel, CpuKernel::Scalar);
        assert_eq!(tuned.backend, Backend::Metal);
        assert_eq!(
            (tuned.gpu_batch_bits, tuned.seed, tuned.count),
            (24, Some(0xabc), 0)
        );

        assert!(with(&["--threads", "0"]).is_err());
        assert!(with(&["--gpu-batch-bits", "31"]).is_err());
        assert!(with(&["--seed", "123456789"]).is_err());
        // inputs are flags, never positionals
        assert!(config(&[FACTORY]).is_err());
    }
}
