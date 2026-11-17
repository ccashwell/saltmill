//! saltmill: a vanity salt miner for CREATE2 and CREATE3 addresses.

#![deny(unused_must_use, rust_2018_idioms)]

pub mod cpu;
pub mod keccak;
#[cfg(target_arch = "aarch64")]
mod keccak_x2;
pub mod pattern;
pub mod plan;
pub mod session;
