//! Address derivation plans: how a 32-byte salt becomes a contract address
//! under each supported deployment scheme, and where the mined bytes of that
//! salt sit in the first message hashed.
//!
//! Every scheme is a chain of one to three single-block keccaks:
//!
//! ```text
//! guard   (optional)  guarded = keccak256(<factory-specific preimage of the salt>)
//! create2             address = keccak256(0xff ++ factory ++ salt ++ initCodeHash)[12..]
//! create  (CREATE3)   address = keccak256(0xd6 0x94 ++ proxy ++ 0x01)[12..]
//! ```
//!
//! CREATE2 is the middle stage alone. CREATE3 deploys a fixed proxy with
//! CREATE2 and the proxy deploys the contract with CREATE at nonce 1, so the
//! address never depends on the contract's init code. CreateX hashes the salt
//! together with the caller (and optionally the chain id) before either, which
//! is the guard stage.

use crate::keccak::{block_lanes, keccak256};

/// CreateX (github.com/pcaversaccio/createx): the same address on every chain
/// it is deployed to. Reproduced against mainnet and Base on 2026-11-19 by
/// simulating `deployCreate3` and comparing with this crate's derivation.
pub const CREATEX: [u8; 20] = [
    0xba, 0x5e, 0xd0, 0x99, 0x63, 0x3d, 0x3b, 0x31, 0x3e, 0x4d, 0x5f, 0x7b, 0xdc, 0x13, 0x05, 0xd3,
    0xc2, 0x8b, 0xa5, 0xed,
];

/// The CREATE3 proxy init code used by CreateX, solmate and Solady:
/// `67363d3d37363d34f03d5260086018f3`. Its hash is pinned by a test.
pub const CREATE3_PROXY_INIT_CODE: [u8; 16] = [
    0x67, 0x36, 0x3d, 0x3d, 0x37, 0x36, 0x3d, 0x34, 0xf0, 0x3d, 0x52, 0x60, 0x08, 0x60, 0x18, 0xf3,
];

/// The salt's last eight bytes are the search counter, in every scheme.
const COUNTER_START: usize = 24;

/// How the final address is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Scheme {
    /// `keccak256(0xff ++ factory ++ salt ++ initCodeHash)`
    Create2,
    /// CREATE2 of a fixed proxy, then CREATE from the proxy at nonce 1
    Create3,
}

/// What the factory does to the salt before using it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guard {
    /// The salt goes into CREATE2 as given. Its first 20 bytes are the caller
    /// (or zero), for factories that check the prefix against `msg.sender`.
    None,
    /// CreateX's `_guard` for a permissioned salt: bytes 0..20 are the caller
    /// and byte 20 selects cross-chain redeploy protection. With a chain id
    /// the guard also hashes it, so the address differs per chain.
    CreateX { chain_id: Option<u64> },
}

/// The first message hashed for a candidate, as sponge lanes, plus where the
/// 64-bit counter is spliced into it. Consumed by the lane-based backends
/// (2-way NEON and the GPU kernel).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneJob {
    /// Rate lanes of the first message, with the counter bytes zeroed.
    pub first: [u64; 17],
    /// The lane holding the counter's low bits.
    pub ctr_lane: usize,
    /// The counter's bit offset in that lane (a multiple of 8). When it is
    /// non-zero the counter's high bits spill into the next lane.
    pub ctr_shift: u32,
    /// Rate lanes of the CREATE2 message with the salt zeroed, present when a
    /// guard stage feeds it.
    pub guarded_create2: Option<[u64; 17]>,
    /// Whether the CREATE hop of CREATE3 follows.
    pub create3: bool,
}

/// A fully resolved derivation: scheme, guard, factory and the fixed part of
/// the salt.
#[derive(Clone, Debug)]
pub struct Plan {
    pub scheme: Scheme,
    pub guard: Guard,
    pub factory: [u8; 20],
    pub caller: [u8; 20],
    /// The hash CREATE2 commits to: the contract's init code hash for
    /// CREATE2, the proxy's for CREATE3.
    pub init_code_hash: [u8; 32],
    salt_base: [u8; 32],
    free_start: usize,
}

impl Plan {
    pub fn new(
        scheme: Scheme,
        guard: Guard,
        factory: [u8; 20],
        caller: [u8; 20],
        init_code_hash: [u8; 32],
    ) -> Self {
        let mut salt_base = [0u8; 32];
        salt_base[..20].copy_from_slice(&caller);
        let free_start = match guard {
            Guard::None => 20,
            Guard::CreateX { chain_id } => {
                salt_base[20] = chain_id.is_some() as u8;
                21
            }
        };
        Self {
            scheme,
            guard,
            factory,
            caller,
            init_code_hash,
            salt_base,
            free_start,
        }
    }

    /// Number of salt bytes between the fixed prefix and the counter. Each
    /// miner fills them with its own tag so no two search the same salts.
    pub fn tag_len(&self) -> usize {
        COUNTER_START - self.free_start
    }

    /// Number of salt bytes the search controls.
    pub fn mined_bytes(&self) -> usize {
        32 - self.free_start
    }

    /// The salt for a tag and counter value.
    pub fn salt(&self, tag: &[u8], counter: u64) -> [u8; 32] {
        assert_eq!(tag.len(), self.tag_len());
        let mut salt = self.salt_base;
        salt[self.free_start..COUNTER_START].copy_from_slice(tag);
        salt[COUNTER_START..].copy_from_slice(&counter.to_le_bytes());
        salt
    }

    /// Whether `salt` has the fixed prefix this plan derives addresses for.
    /// A CreateX salt with another prefix takes a different guard branch, so
    /// its address is not the one `address` computes.
    pub fn accepts(&self, salt: &[u8; 32]) -> bool {
        salt[..self.free_start] == self.salt_base[..self.free_start]
    }

    /// The guard preimage for `salt` and the salt's offset in it.
    fn guard_message(&self, salt: &[u8; 32]) -> Option<(Vec<u8>, usize)> {
        let Guard::CreateX { chain_id } = self.guard else {
            return None;
        };
        // keccak256(abi.encode(caller, salt)) or abi.encode(caller, chainId, salt)
        let mut message = vec![0u8; 12];
        message.extend_from_slice(&self.caller);
        if let Some(chain_id) = chain_id {
            message.extend_from_slice(&[0u8; 24]);
            message.extend_from_slice(&chain_id.to_be_bytes());
        }
        let offset = message.len();
        message.extend_from_slice(salt);
        Some((message, offset))
    }

    fn create2_message(&self, salt: &[u8; 32]) -> [u8; 85] {
        let mut message = [0u8; 85];
        message[0] = 0xff;
        message[1..21].copy_from_slice(&self.factory);
        message[21..53].copy_from_slice(salt);
        message[53..].copy_from_slice(&self.init_code_hash);
        message
    }

    /// The salt CREATE2 receives: `salt` itself, or its guard hash.
    pub fn guarded_salt(&self, salt: &[u8; 32]) -> [u8; 32] {
        match self.guard_message(salt) {
            Some((message, _)) => keccak256(&message),
            None => *salt,
        }
    }

    /// The address `salt` deploys to. This is the reference derivation, built
    /// from whole-message digests; the lane-based backends are checked
    /// against it.
    pub fn address(&self, salt: &[u8; 32]) -> [u8; 20] {
        let digest = keccak256(&self.create2_message(&self.guarded_salt(salt)));
        let mut address = [0u8; 20];
        address.copy_from_slice(&digest[12..]);
        if self.scheme == Scheme::Create2 {
            return address;
        }

        // rlp([proxy, 1]): list prefix 0xd6, 20-byte string prefix 0x94, the
        // proxy, nonce 1
        let mut message = [0u8; 23];
        message[0] = 0xd6;
        message[1] = 0x94;
        message[2..22].copy_from_slice(&address);
        message[22] = 0x01;
        address.copy_from_slice(&keccak256(&message)[12..]);
        address
    }

    /// The lane view of this plan for a miner's tag.
    pub fn lane_job(&self, tag: &[u8]) -> LaneJob {
        let salt = self.salt(tag, 0);
        let (first, salt_offset, guarded_create2) = match self.guard_message(&salt) {
            Some((message, offset)) => (
                block_lanes(&message),
                offset,
                Some(block_lanes(&self.create2_message(&[0u8; 32]))),
            ),
            None => (block_lanes(&self.create2_message(&salt)), 21, None),
        };
        let counter_offset = salt_offset + COUNTER_START;
        LaneJob {
            first,
            ctr_lane: counter_offset / 8,
            ctr_shift: (counter_offset % 8) as u32 * 8,
            guarded_create2,
            create3: self.scheme == Scheme::Create3,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::keccak::{checksum, parse_fixed};

    pub(crate) const CALLER: &str = "0x1234567890abcdef1234567890abcdef12345678";

    pub(crate) fn proxy_hash() -> [u8; 32] {
        keccak256(&CREATE3_PROXY_INIT_CODE)
    }

    /// One plan per derivation shape, for backends to check themselves on.
    pub(crate) fn all_plans() -> Vec<Plan> {
        let caller = parse_fixed::<20>(CALLER).unwrap();
        let factory = [0xaa; 20];
        let init = [0x33; 32];
        vec![
            Plan::new(Scheme::Create2, Guard::None, factory, caller, init),
            Plan::new(Scheme::Create2, Guard::None, factory, [0; 20], init),
            Plan::new(
                Scheme::Create2,
                Guard::CreateX { chain_id: None },
                CREATEX,
                caller,
                init,
            ),
            Plan::new(
                Scheme::Create2,
                Guard::CreateX {
                    chain_id: Some(8453),
                },
                CREATEX,
                caller,
                init,
            ),
            Plan::new(Scheme::Create3, Guard::None, factory, caller, proxy_hash()),
            Plan::new(
                Scheme::Create3,
                Guard::CreateX { chain_id: None },
                CREATEX,
                caller,
                proxy_hash(),
            ),
            Plan::new(
                Scheme::Create3,
                Guard::CreateX {
                    chain_id: Some(u64::MAX),
                },
                CREATEX,
                caller,
                proxy_hash(),
            ),
        ]
    }

    #[test]
    fn proxy_init_code_hash_is_the_published_one() {
        assert_eq!(
            proxy_hash(),
            parse_fixed::<32>("0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f")
                .unwrap()
        );
    }

    #[test]
    fn createx_constant_is_the_canonical_address() {
        assert_eq!(
            checksum(&CREATEX),
            "0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed"
        );
    }

    // The expected addresses below come from `cast create2` and
    // `cast compute-address` (foundry 1.9.0), not from this crate.

    #[test]
    fn create2_matches_cast() {
        let init = keccak256(&[0x00]);
        let plan = Plan::new(Scheme::Create2, Guard::None, [0; 20], [0; 20], init);
        assert_eq!(
            checksum(&plan.address(&[0; 32])),
            "0x4D1A2e2bB4F88F0250f26Ffff098B0b30B26BF38"
        );

        let factory = parse_fixed::<20>("0xdeadbeef00000000000000000000000000000000").unwrap();
        let salt =
            parse_fixed::<32>("0x000000000000000000000000feed000000000000000000000000000000000000")
                .unwrap();
        let plan = Plan::new(Scheme::Create2, Guard::None, factory, [0; 20], init);
        assert_eq!(
            checksum(&plan.address(&salt)),
            "0xD04116cDd17beBE565EB2422F2497E06cC1C9833"
        );
    }

    /// Also the address a simulated `deployCreate3` from this caller returned
    /// on mainnet CreateX (2026-11-19).
    #[test]
    fn createx_create3_matches_cast_and_mainnet() {
        let caller = parse_fixed::<20>(CALLER).unwrap();
        let plan = Plan::new(
            Scheme::Create3,
            Guard::CreateX { chain_id: None },
            CREATEX,
            caller,
            proxy_hash(),
        );
        let salt =
            parse_fixed::<32>("0x1234567890abcdef1234567890abcdef12345678000102030405060708090a0b")
                .unwrap();
        assert!(plan.accepts(&salt));
        assert_eq!(
            plan.guarded_salt(&salt),
            parse_fixed::<32>("0x09565892ae2588b7ce551a5814086b8db3fe0c4656353ad1a43d29ac92e6a792")
                .unwrap()
        );
        assert_eq!(
            checksum(&plan.address(&salt)),
            "0x605AD0Dc14138FEE80f142f31c7dE12a48269FF0"
        );
    }

    /// Also the address a simulated `deployCreate3` returned on Base (chain
    /// id 8453) CreateX (2026-11-19).
    #[test]
    fn createx_create3_with_chain_id_matches_cast_and_base() {
        let caller = parse_fixed::<20>(CALLER).unwrap();
        let plan = Plan::new(
            Scheme::Create3,
            Guard::CreateX {
                chain_id: Some(8453),
            },
            CREATEX,
            caller,
            proxy_hash(),
        );
        let salt =
            parse_fixed::<32>("0x1234567890abcdef1234567890abcdef12345678010102030405060708090a0b")
                .unwrap();
        assert!(plan.accepts(&salt));
        assert_eq!(
            checksum(&plan.address(&salt)),
            "0xe28485d7fA9F22057DE26964c5e7bB1b850B5564"
        );

        // the same salt without the protection byte belongs to another plan
        let unprotected = Plan::new(
            Scheme::Create3,
            Guard::CreateX { chain_id: None },
            CREATEX,
            caller,
            proxy_hash(),
        );
        assert!(!unprotected.accepts(&salt));
    }

    #[test]
    fn createx_create2_matches_cast() {
        let caller = parse_fixed::<20>(CALLER).unwrap();
        let plan = Plan::new(
            Scheme::Create2,
            Guard::CreateX { chain_id: None },
            CREATEX,
            caller,
            keccak256(&[0x00]),
        );
        let salt =
            parse_fixed::<32>("0x1234567890abcdef1234567890abcdef12345678000102030405060708090a0b")
                .unwrap();
        assert_eq!(
            checksum(&plan.address(&salt)),
            "0x855902E7AB5408B18071eEE343Bec1CfA0b44E0f"
        );
    }

    #[test]
    fn unguarded_create3_matches_cast() {
        let mut factory = [0u8; 20];
        factory[19] = 0xaa;
        let plan = Plan::new(Scheme::Create3, Guard::None, factory, [0; 20], proxy_hash());
        let salt =
            parse_fixed::<32>("0x00000000000000000000000000000000000000000102030405060708090a0b0c")
                .unwrap();
        assert_eq!(
            checksum(&plan.address(&salt)),
            "0xe1A88be7D5a6B95bf59715171AD9d49619755eF8"
        );
    }

    #[test]
    fn salt_layout_is_prefix_tag_counter() {
        let plans = all_plans();
        let raw = &plans[0];
        assert_eq!((raw.tag_len(), raw.mined_bytes()), (4, 12));
        let salt = raw.salt(&[0xa1, 0xa2, 0xa3, 0xa4], 0x0807_0605_0403_0201);
        assert_eq!(salt[..20], parse_fixed::<20>(CALLER).unwrap());
        assert_eq!(salt[20..24], [0xa1, 0xa2, 0xa3, 0xa4]);
        assert_eq!(salt[24..], [1, 2, 3, 4, 5, 6, 7, 8]);

        let createx = &plans[5];
        assert_eq!((createx.tag_len(), createx.mined_bytes()), (3, 11));
        let salt = createx.salt(&[0xb1, 0xb2, 0xb3], 1);
        assert_eq!(salt[20], 0x00);
        assert_eq!(salt[21..24], [0xb1, 0xb2, 0xb3]);
        assert!(createx.accepts(&salt));

        let protected = &plans[6];
        assert_eq!(protected.salt(&[0; 3], 0)[20], 0x01);
    }

    #[test]
    fn lane_job_locates_the_counter() {
        let plans = all_plans();
        // raw CREATE2: the counter is message bytes 45..53
        let job = plans[0].lane_job(&[0; 4]);
        assert_eq!((job.ctr_lane, job.ctr_shift), (5, 40));
        assert!(job.guarded_create2.is_none() && !job.create3);

        // CreateX without chain id: 64-byte guard preimage, counter 56..64
        let job = plans[5].lane_job(&[0; 3]);
        assert_eq!((job.ctr_lane, job.ctr_shift), (7, 0));
        assert!(job.guarded_create2.is_some() && job.create3);

        // CreateX with chain id: 96-byte guard preimage, counter 88..96
        let job = plans[6].lane_job(&[0; 3]);
        assert_eq!((job.ctr_lane, job.ctr_shift), (11, 0));
        // the chain id is big-endian in bytes 56..64, i.e. lane 7
        assert_eq!(job.first[7], u64::MAX);
    }
}
