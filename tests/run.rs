//! End-to-end runs through the same entry point as the binary.

use clap::Parser;
use saltmill::cli::{Args, Config};
use saltmill::keccak::{checksum, parse_fixed};
use saltmill::session::Hit;

const CALLER: &str = "0x1234567890abcdef1234567890abcdef12345678";

fn run(flags: &[&str]) -> Result<Vec<Hit>, String> {
    let args = Args::try_parse_from(["saltmill"].iter().chain(flags)).map_err(|e| e.to_string())?;
    Config::new(args).and_then(saltmill::run)
}

#[test]
fn cpu_run_returns_the_requested_number_of_hits() {
    let hits = run(&[
        "--scheme",
        "create3",
        "--caller",
        CALLER,
        "--prefix",
        "abc",
        "--count",
        "2",
        "--compute",
        "cpu",
        "--threads",
        "2",
        "--quiet",
    ])
    .unwrap();
    assert_eq!(hits.len(), 2);
    for hit in hits {
        assert!(checksum(&hit.address).to_lowercase().starts_with("0xabc"));
        // a permissioned CreateX salt: the caller, then the zero flag byte
        assert_eq!(hit.salt[..20], parse_fixed::<20>(CALLER).unwrap());
        assert_eq!(hit.salt[20], 0);
    }
}

/// The default compute mode: the GPU joins when the machine has one, and
/// the run succeeds on the CPU alone when it does not.
#[test]
fn default_run_mines_create2_hook_flags() {
    let hits = run(&[
        "--scheme",
        "create2",
        "--factory",
        "0x4e59b44847b379578588920cA78FbF26c0B4956C",
        "--init-code-hash",
        "0x2222222222222222222222222222222222222222222222222222222222222222",
        "--hook-flags",
        "0x00c0",
        "--prefix",
        "c0",
        "--count",
        "3",
        "--quiet",
    ])
    .unwrap();
    assert_eq!(hits.len(), 3);
    for hit in hits {
        assert_eq!(hit.address[0], 0xc0);
        assert_eq!(
            u16::from_be_bytes([hit.address[18], hit.address[19]]) & 0x3fff,
            0x00c0
        );
        assert_eq!(hit.salt[..20], [0u8; 20]);
    }
}

#[test]
fn hits_are_appended_to_the_output_file() {
    let path = std::env::temp_dir().join(format!("saltmill-test-{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let hits = run(&[
        "-s",
        "create3",
        "-c",
        CALLER,
        "--suffix",
        "f00",
        "-n",
        "2",
        "--compute",
        "cpu",
        "-q",
        "--output",
        path.to_str().unwrap(),
    ])
    .unwrap();
    let written = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let lines: Vec<String> = hits.iter().map(Hit::line).collect();
    assert_eq!(written.lines().collect::<Vec<_>>(), lines);
}

#[test]
fn check_derives_the_address_of_a_given_salt() {
    // reproduced on mainnet CreateX by simulating deployCreate3 (2026-11-19)
    let salt = "0x1234567890abcdef1234567890abcdef12345678000102030405060708090a0b";
    let hits = run(&["-s", "create3", "-c", CALLER, "--check", salt]).unwrap();
    assert_eq!(
        checksum(&hits[0].address),
        "0x605AD0Dc14138FEE80f142f31c7dE12a48269FF0"
    );

    // with a target, a salt that misses it is an error
    assert!(run(&["-s", "create3", "-c", CALLER, "--check", salt, "--prefix", "605a"]).is_ok());
    let miss = run(&[
        "-s", "create3", "-c", CALLER, "--check", salt, "--prefix", "dead",
    ]);
    assert!(miss.unwrap_err().contains("does not meet the target"));

    // an unguarded factory takes any salt (address from `cast create2`)
    let hits = run(&[
        "-s",
        "create2",
        "-f",
        "0xdeadbeef00000000000000000000000000000000",
        "-i",
        "0xbc36789e7a1e281436464229828f817d6612f7b477d66591ff96a9e064bcc98a",
        "--check",
        "0x000000000000000000000000feed000000000000000000000000000000000000",
    ])
    .unwrap();
    assert_eq!(
        checksum(&hits[0].address),
        "0xD04116cDd17beBE565EB2422F2497E06cC1C9833"
    );

    // a salt CreateX would guard differently is refused, not mis-derived
    let foreign = "0xffffffffffffffffffffffffffffffffffffffff000102030405060708090a0b";
    let refused = run(&["-s", "create3", "-c", CALLER, "--check", foreign]);
    assert!(refused.unwrap_err().contains("does not start with"));
}

#[test]
fn time_limit_fails_a_run_that_finds_too_few_hits() {
    let outcome = run(&[
        "-s",
        "create3",
        "-c",
        CALLER,
        "--prefix",
        "0123456789abcdef0123",
        "--compute",
        "cpu",
        "--threads",
        "1",
        "--time-limit",
        "1",
        "-q",
    ]);
    assert!(outcome
        .unwrap_err()
        .contains("time limit reached with 0 of 1"));
}
