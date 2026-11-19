//! What makes an address a hit: a bit pattern over its 20 bytes, optionally
//! with a minimum number of zero bytes.

use std::fmt;

/// Uniswap v4 `Hooks.ALL_HOOK_MASK`: hook permissions occupy the lowest 14
/// bits of the hook address and must match the declared permissions exactly.
pub const HOOK_FLAG_MASK: u16 = (1 << 14) - 1;

/// A bit-level constraint over a 20-byte address: an address matches when
/// `address & mask == value` for every byte. Built from vanity prefixes,
/// suffixes, leading zero bytes and Uniswap v4 hook permission flags, which
/// may be combined as long as their fixed bits agree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pattern {
    mask: [u8; 20],
    value: [u8; 20],
}

impl Pattern {
    /// Constrain the address to start with the given nibbles.
    pub fn from_prefix(nibbles: &[u8]) -> Self {
        let mut pattern = Self::default();
        for (i, &nibble) in nibbles.iter().enumerate() {
            pattern.set_nibble(i, nibble);
        }
        pattern
    }

    /// Constrain the address to end with the given nibbles.
    pub fn from_suffix(nibbles: &[u8]) -> Self {
        let mut pattern = Self::default();
        for (i, &nibble) in nibbles.iter().enumerate() {
            pattern.set_nibble(40 - nibbles.len() + i, nibble);
        }
        pattern
    }

    /// Constrain the first `count` bytes of the address to zero.
    pub fn from_leading_zero_bytes(count: usize) -> Self {
        let mut pattern = Self::default();
        pattern.mask[..count].fill(0xff);
        pattern
    }

    /// Constrain the lowest 14 bits of the address to exactly equal the given
    /// Uniswap v4 hook permission flags.
    pub fn from_hook_flags(flags: u16) -> Self {
        let mut pattern = Self::default();
        pattern.mask[18] = (HOOK_FLAG_MASK >> 8) as u8;
        pattern.mask[19] = 0xff;
        pattern.value[18] = (flags >> 8) as u8;
        pattern.value[19] = (flags & 0xff) as u8;
        pattern
    }

    /// Constrain every bit: only `address` matches.
    pub fn exact(address: &[u8; 20]) -> Self {
        Self {
            mask: [0xff; 20],
            value: *address,
        }
    }

    fn set_nibble(&mut self, position: usize, nibble: u8) {
        let byte = position / 2;
        if position.is_multiple_of(2) {
            self.mask[byte] |= 0xf0;
            self.value[byte] |= nibble << 4;
        } else {
            self.mask[byte] |= 0x0f;
            self.value[byte] |= nibble;
        }
    }

    /// Combine two patterns, requiring agreement on any overlapping bits.
    pub fn merge(self, other: Self) -> Result<Self, String> {
        let mut merged = Self::default();
        for i in 0..20 {
            let overlap = self.mask[i] & other.mask[i];
            if (self.value[i] ^ other.value[i]) & overlap != 0 {
                return Err(format!(
                    "conflicting patterns: the constraints disagree on address byte {i}"
                ));
            }
            merged.mask[i] = self.mask[i] | other.mask[i];
            merged.value[i] = self.value[i] | other.value[i];
        }
        Ok(merged)
    }

    /// Whether the address satisfies this pattern.
    #[inline]
    pub fn matches(&self, address: &[u8; 20]) -> bool {
        // three word compares instead of twenty byte compares: this runs
        // once per candidate on the CPU path
        let head = |b: &[u8; 20]| u64::from_le_bytes(b[..8].try_into().unwrap());
        let body = |b: &[u8; 20]| u64::from_le_bytes(b[8..16].try_into().unwrap());
        let tail = |b: &[u8; 20]| u32::from_le_bytes(b[16..].try_into().unwrap());
        (head(address) & head(&self.mask)) == head(&self.value)
            && (body(address) & body(&self.mask)) == body(&self.value)
            && (tail(address) & tail(&self.mask)) == tail(&self.value)
    }

    /// Number of constrained bits; a random address matches with probability
    /// 2^-bits().
    pub fn bits(&self) -> u32 {
        self.mask.iter().map(|m| m.count_ones()).sum()
    }

    /// The mask and value as five little-endian 32-bit words, the form the
    /// GPU kernel compares the address in.
    pub fn words(&self) -> ([u32; 5], [u32; 5]) {
        let words = |bytes: &[u8; 20]| {
            let mut out = [0u32; 5];
            for (i, word) in out.iter_mut().enumerate() {
                *word = u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
            }
            out
        };
        (words(&self.mask), words(&self.value))
    }

    /// The probability that byte `i` of a matching address is zero.
    fn zero_probability(&self, i: usize) -> f64 {
        if self.value[i] != 0 {
            0.0
        } else {
            0.5f64.powi(8 - self.mask[i].count_ones() as i32)
        }
    }
}

/// Renders the pattern as an address template: fixed nibbles as hex digits,
/// free nibbles as `x`, and partially-constrained nibbles (e.g. the top two
/// bits of the 14-bit hook flags) as `?`.
impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x")?;
        for position in 0..40 {
            let byte = position / 2;
            let (mask, value) = if position % 2 == 0 {
                (self.mask[byte] >> 4, self.value[byte] >> 4)
            } else {
                (self.mask[byte] & 0xf, self.value[byte] & 0xf)
            };
            match mask {
                0x0 => write!(f, "x")?,
                0xf => write!(f, "{value:x}")?,
                _ => write!(f, "?")?,
            }
        }
        Ok(())
    }
}

/// The full hit test: the pattern, and optionally at least this many zero
/// bytes anywhere in the address.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Criteria {
    pub pattern: Pattern,
    pub total_zero_bytes: Option<u8>,
}

impl Criteria {
    /// Whether nothing is constrained (every address would be a hit).
    pub fn is_empty(&self) -> bool {
        self.pattern.bits() == 0 && self.total_zero_bytes.unwrap_or(0) == 0
    }

    #[inline]
    pub fn matches(&self, address: &[u8; 20]) -> bool {
        if !self.pattern.matches(address) {
            return false;
        }
        match self.total_zero_bytes {
            Some(min) => zero_bytes(address) >= min as usize,
            None => true,
        }
    }

    /// -log2 of the probability that a uniformly random address is a hit.
    pub fn difficulty_bits(&self) -> f64 {
        let mut bits = self.pattern.bits() as f64;
        if let Some(min) = self.total_zero_bytes {
            // P(at least `min` zero bytes | pattern holds): each byte is zero
            // independently with its own probability, so walk the
            // distribution of the zero count byte by byte
            let mut ways = [0f64; 21];
            ways[0] = 1.0;
            for i in 0..20 {
                let p = self.pattern.zero_probability(i);
                for k in (0..=i).rev() {
                    ways[k + 1] += ways[k] * p;
                    ways[k] *= 1.0 - p;
                }
            }
            let tail: f64 = ways[(min as usize).min(20)..].iter().sum();
            bits -= tail.log2();
        }
        bits
    }
}

impl fmt::Display for Criteria {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pattern)?;
        if let Some(min) = self.total_zero_bytes {
            write!(f, " with \u{2265}{min} zero bytes")?;
        }
        Ok(())
    }
}

/// Number of zero bytes in the address.
pub fn zero_bytes(address: &[u8; 20]) -> usize {
    address.iter().filter(|&&b| b == 0).count()
}

/// Number of leading zero bytes in the address.
pub fn leading_zero_bytes(address: &[u8; 20]) -> usize {
    address.iter().take_while(|&&b| b == 0).count()
}

/// Parse a hex string (optionally 0x-prefixed, odd lengths allowed) into
/// individual nibble values.
pub fn parse_nibbles(s: &str, what: &str) -> Result<Vec<u8>, String> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if s.is_empty() {
        return Err(format!("{what} pattern is empty"));
    }
    if s.len() > 40 {
        return Err(format!("{what} pattern is longer than 40 hex characters"));
    }
    s.chars()
        .map(|c| {
            c.to_digit(16)
                .map(|d| d as u8)
                .ok_or_else(|| format!("{what} pattern contains non-hex character {c:?}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hook_flags_constrain_lowest_14_bits() {
        // BEFORE_SWAP (1 << 7) | AFTER_SWAP (1 << 6) | BEFORE_INITIALIZE (1 << 13)
        let flags = (1u16 << 7) | (1 << 6) | (1 << 13);
        let pattern = Pattern::from_hook_flags(flags);
        assert_eq!(pattern.bits(), 14);

        let mut address = [0u8; 20];
        address[18] = (flags >> 8) as u8;
        address[19] = (flags & 0xff) as u8;
        assert!(pattern.matches(&address));

        // the top two bits of byte 18 are outside the 14-bit mask
        address[18] |= 0xc0;
        assert!(pattern.matches(&address));

        // any flipped flag bit must reject
        address[19] ^= 1;
        assert!(!pattern.matches(&address));
    }

    #[test]
    fn prefix_and_suffix_handle_odd_nibble_counts() {
        let prefix = Pattern::from_prefix(&parse_nibbles("c0ffe", "prefix").unwrap());
        let suffix = Pattern::from_suffix(&parse_nibbles("abc", "suffix").unwrap());

        let mut address = [0u8; 20];
        address[0] = 0xc0;
        address[1] = 0xff;
        address[2] = 0xe7; // low nibble free
        address[18] = 0x0a; // high nibble free
        address[19] = 0xbc;
        assert!(prefix.matches(&address));
        assert!(suffix.matches(&address));

        address[2] = 0xd7;
        assert!(!prefix.matches(&address));
        address[2] = 0xe7;
        address[18] = 0x0b;
        assert!(!suffix.matches(&address));
    }

    #[test]
    fn matches_checks_every_byte_position() {
        let target: [u8; 20] = core::array::from_fn(|i| 0x10 + i as u8);
        let pattern = Pattern::exact(&target);
        assert!(pattern.matches(&target));
        for i in 0..20 {
            let mut other = target;
            other[i] ^= 0x01;
            assert!(!pattern.matches(&other), "byte {i} is not compared");
        }
    }

    #[test]
    fn merge_rejects_conflicting_bits_and_accepts_agreement() {
        let suffix = Pattern::from_suffix(&parse_nibbles("ff", "suffix").unwrap());
        assert!(suffix.merge(Pattern::from_hook_flags(0)).is_err());
        assert_eq!(
            suffix.merge(Pattern::from_hook_flags(0xff)).unwrap().bits(),
            14
        );

        let zeros = Pattern::from_leading_zero_bytes(2);
        let prefix = Pattern::from_prefix(&parse_nibbles("0000c0", "prefix").unwrap());
        assert_eq!(zeros.merge(prefix).unwrap().bits(), 24);
        assert!(zeros
            .merge(Pattern::from_prefix(
                &parse_nibbles("c0", "prefix").unwrap()
            ))
            .is_err());
    }

    #[test]
    fn display_marks_free_and_partial_nibbles() {
        let pattern = Pattern::from_prefix(&parse_nibbles("0xc0ffee", "prefix").unwrap())
            .merge(Pattern::from_hook_flags(0x00c0))
            .unwrap();
        assert_eq!(pattern.bits(), 24 + 14);
        assert_eq!(
            pattern.to_string(),
            "0xc0ffeexxxxxxxxxxxxxxxxxxxxxxxxxxxxxx?0c0"
        );
    }

    #[test]
    fn words_are_little_endian_address_words() {
        let (mask, value) = Pattern::from_prefix(&[0xa, 0xb, 0xc]).words();
        assert_eq!(mask, [0x0000_f0ff, 0, 0, 0, 0]);
        assert_eq!(value, [0x0000_c0ab, 0, 0, 0, 0]);

        let (mask, value) = Pattern::from_hook_flags(0x2400).words();
        assert_eq!(mask[4], 0xff3f_0000);
        assert_eq!(value[4], 0x0024_0000);
    }

    #[test]
    fn criteria_combine_pattern_and_zero_bytes() {
        let criteria = Criteria {
            pattern: Pattern::from_leading_zero_bytes(2),
            total_zero_bytes: Some(4),
        };
        let mut address = [0x11u8; 20];
        address[0] = 0;
        address[1] = 0;
        assert!(!criteria.matches(&address));
        address[7] = 0;
        address[19] = 0;
        assert!(criteria.matches(&address));
        address[0] = 1;
        assert!(!criteria.matches(&address));

        assert_eq!(zero_bytes(&address), 3);
        assert_eq!(leading_zero_bytes(&address), 0);
        assert_eq!(leading_zero_bytes(&[0u8; 20]), 20);
    }

    #[test]
    fn difficulty_accounts_for_zero_bytes_the_pattern_already_fixes() {
        let bits = |pattern, total_zero_bytes| {
            Criteria {
                pattern,
                total_zero_bytes,
            }
            .difficulty_bits()
        };
        assert_eq!(bits(Pattern::from_hook_flags(0), None), 14.0);

        // four leading zero bytes already are four zero bytes
        let zeros = Pattern::from_leading_zero_bytes(4);
        assert!((bits(zeros, Some(4)) - 32.0).abs() < 1e-9);
        assert!(bits(zeros, Some(5)) > 36.0);

        // one zero byte anywhere among 20: 1 - (255/256)^20
        let expected = -(1.0 - (255.0f64 / 256.0).powi(20)).log2();
        assert!((bits(Pattern::default(), Some(1)) - expected).abs() < 1e-9);

        // a byte fixed to a non-zero value can never be one of the zeros
        let fixed = Pattern::exact(&[0x11; 20]);
        assert_eq!(bits(fixed, Some(1)), f64::INFINITY);
    }

    #[test]
    fn parse_nibbles_validates_input() {
        assert_eq!(parse_nibbles("0xAb1", "prefix").unwrap(), vec![0xa, 0xb, 1]);
        assert!(parse_nibbles("", "prefix").is_err());
        assert!(parse_nibbles("xyz", "prefix").is_err());
        assert!(parse_nibbles(&"0".repeat(41), "prefix").is_err());
    }
}
