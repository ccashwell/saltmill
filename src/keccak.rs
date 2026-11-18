//! Keccak-256 primitives shared by every backend: the reference digest, the
//! sponge-lane view of a single-block message, and the bit-interleaved lane
//! representation the GPU kernel works in.

use keccak_asm::Keccak256;

/// Keccak-256 rate in bytes. Every message the miner hashes fits in one block.
pub const RATE: usize = 136;

/// Keccak-f[1600] round constants.
pub const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// The reference digest (CRYPTOGAMS assembly). Every hit a vectorized or GPU
/// backend reports is recomputed through this before it is shown.
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    Keccak256::digest(data).into()
}

/// The 17 rate lanes of the sponge after absorbing a padded single-block
/// message (capacity lanes 17..=24 stay zero).
pub fn block_lanes(message: &[u8]) -> [u64; 17] {
    assert!(message.len() < RATE, "message must fit in one keccak block");
    let mut block = [0u8; RATE];
    block[..message.len()].copy_from_slice(message);
    block[message.len()] ^= 0x01;
    block[RATE - 1] ^= 0x80;

    let mut lanes = [0u64; 17];
    for (j, lane) in lanes.iter_mut().enumerate() {
        *lane = u64::from_le_bytes(block[8 * j..8 * j + 8].try_into().unwrap());
    }
    lanes
}

/// Split a 64-bit lane into its even bits and its odd bits, each packed into
/// a 32-bit word. In this form a 64-bit rotation costs one or two native
/// 32-bit rotations, which is what makes keccak fast on GPUs without 64-bit
/// integer ALUs (Apple silicon).
pub fn bit_interleave(lane: u64) -> (u32, u32) {
    let mut even = 0u32;
    let mut odd = 0u32;
    for i in 0..32 {
        even |= (((lane >> (2 * i)) & 1) as u32) << i;
        odd |= (((lane >> (2 * i + 1)) & 1) as u32) << i;
    }
    (even, odd)
}

/// Lowercase hex without a prefix.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// EIP-55 checksummed address with a `0x` prefix.
pub fn checksum(address: &[u8; 20]) -> String {
    let lower = hex(address);
    let hash = keccak256(lower.as_bytes());
    let mut out = String::with_capacity(42);
    out.push_str("0x");
    for (i, c) in lower.chars().enumerate() {
        let nibble = if i % 2 == 0 {
            hash[i / 2] >> 4
        } else {
            hash[i / 2] & 0x0f
        };
        out.push(if nibble >= 8 {
            c.to_ascii_uppercase()
        } else {
            c
        });
    }
    out
}

/// Parse an optionally `0x`-prefixed hex string of exactly `N` bytes.
pub fn parse_fixed<const N: usize>(s: &str) -> Result<[u8; N], String> {
    let digits = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    if digits.len() != N * 2 {
        return Err(format!(
            "expected {N} bytes ({} hex characters), got {} characters",
            N * 2,
            digits.len()
        ));
    }
    let mut out = [0u8; N];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&digits[2 * i..2 * i + 2], 16)
            .map_err(|_| format!("{:?} is not hex", &digits[2 * i..2 * i + 2]))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keccak256_matches_known_digest() {
        // keccak256("") from the Keccak team's test vectors
        assert_eq!(
            hex(&keccak256(b"")),
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        );
    }

    #[test]
    fn checksum_matches_eip55_vectors() {
        for expected in [
            "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
            "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
            "0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB",
            "0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb",
        ] {
            let address = parse_fixed::<20>(expected).unwrap();
            assert_eq!(checksum(&address), expected);
        }
    }

    #[test]
    fn block_lanes_place_padding_and_message() {
        let lanes = block_lanes(&[0xff; 85]);
        assert_eq!(lanes[0], u64::MAX);
        // byte 85 is the padding start, in lane 10 at byte 5
        assert_eq!(lanes[10], 0x0000_01ff_ffff_ffff);
        assert_eq!(lanes[16], 0x8000_0000_0000_0000);

        // a 135-byte message shares its last block byte between both pads
        assert_eq!(block_lanes(&[0u8; 135])[16], 0x8100_0000_0000_0000);
    }

    #[test]
    fn bit_interleave_separates_even_and_odd_bits() {
        assert_eq!(bit_interleave(0), (0, 0));
        assert_eq!(bit_interleave(u64::MAX), (u32::MAX, u32::MAX));
        assert_eq!(bit_interleave(0x5555_5555_5555_5555), (u32::MAX, 0));
        assert_eq!(bit_interleave(0b1110), (0b10, 0b11));
    }

    #[test]
    fn parse_fixed_rejects_wrong_lengths_and_non_hex() {
        assert!(parse_fixed::<20>("0x1234").is_err());
        assert!(parse_fixed::<2>("0xzz00").is_err());
        assert_eq!(parse_fixed::<2>("0XAbcd").unwrap(), [0xab, 0xcd]);
        assert_eq!(parse_fixed::<2>("abcd").unwrap(), [0xab, 0xcd]);
    }
}
