//! 2-way parallel keccak-f[1600] using the ARMv8.4 SHA3 NEON instructions.
//!
//! The SHA3 extension's EOR3 (three-way xor), RAX1 (xor-rotate-by-1), XAR
//! (xor-rotate) and BCAX (bit-clear-xor) instructions each operate on a full
//! 128-bit vector, i.e. two independent 64-bit keccak lanes at once. A single
//! hash stream cannot use that width - Apple's scalar pipeline already
//! saturates - so instead each vector register holds the same lane of two
//! *independent* sponge states, hashing two candidate salts per core
//! simultaneously. A CREATE3 candidate chains up to three permutations, and
//! the bytes carried from one digest into the next message stay in the
//! vector registers the whole way.
//!
//! Callers must ensure the `sha3` target feature is present at runtime
//! (`std::arch::is_aarch64_feature_detected!("sha3")`); all Apple M-series
//! chips have it.

use crate::keccak::RC;
use crate::plan::LaneJob;
use core::arch::aarch64::*;

/// Destination index of the combined rho+pi step for each source lane:
/// lane (x, y) at flat index `x + 5y` moves to `(y, 2x + 3y)`.
const PI: [usize; 25] = [
    0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4,
];

/// Keccak-f[1600] on two interleaved states.
#[inline]
#[target_feature(enable = "sha3")]
unsafe fn permute(a: &mut [uint64x2_t; 25]) {
    for &rc in &RC {
        // theta: column parities and D values
        let c0 = veor3q_u64(veor3q_u64(a[0], a[5], a[10]), a[15], a[20]);
        let c1 = veor3q_u64(veor3q_u64(a[1], a[6], a[11]), a[16], a[21]);
        let c2 = veor3q_u64(veor3q_u64(a[2], a[7], a[12]), a[17], a[22]);
        let c3 = veor3q_u64(veor3q_u64(a[3], a[8], a[13]), a[18], a[23]);
        let c4 = veor3q_u64(veor3q_u64(a[4], a[9], a[14]), a[19], a[24]);
        let d0 = vrax1q_u64(c4, c1);
        let d1 = vrax1q_u64(c0, c2);
        let d2 = vrax1q_u64(c1, c3);
        let d3 = vrax1q_u64(c2, c4);
        let d4 = vrax1q_u64(c3, c0);

        // theta + rho + pi: b[PI[i]] = rol(a[i] ^ d[i % 5], RHO[i]), where
        // XAR's immediate is the right-rotation 64 - RHO[i]
        let mut b = [vdupq_n_u64(0); 25];
        b[PI[0]] = veorq_u64(a[0], d0);
        b[PI[1]] = vxarq_u64::<63>(a[1], d1);
        b[PI[2]] = vxarq_u64::<2>(a[2], d2);
        b[PI[3]] = vxarq_u64::<36>(a[3], d3);
        b[PI[4]] = vxarq_u64::<37>(a[4], d4);
        b[PI[5]] = vxarq_u64::<28>(a[5], d0);
        b[PI[6]] = vxarq_u64::<20>(a[6], d1);
        b[PI[7]] = vxarq_u64::<58>(a[7], d2);
        b[PI[8]] = vxarq_u64::<9>(a[8], d3);
        b[PI[9]] = vxarq_u64::<44>(a[9], d4);
        b[PI[10]] = vxarq_u64::<61>(a[10], d0);
        b[PI[11]] = vxarq_u64::<54>(a[11], d1);
        b[PI[12]] = vxarq_u64::<21>(a[12], d2);
        b[PI[13]] = vxarq_u64::<39>(a[13], d3);
        b[PI[14]] = vxarq_u64::<25>(a[14], d4);
        b[PI[15]] = vxarq_u64::<23>(a[15], d0);
        b[PI[16]] = vxarq_u64::<19>(a[16], d1);
        b[PI[17]] = vxarq_u64::<49>(a[17], d2);
        b[PI[18]] = vxarq_u64::<43>(a[18], d3);
        b[PI[19]] = vxarq_u64::<56>(a[19], d4);
        b[PI[20]] = vxarq_u64::<46>(a[20], d0);
        b[PI[21]] = vxarq_u64::<62>(a[21], d1);
        b[PI[22]] = vxarq_u64::<3>(a[22], d2);
        b[PI[23]] = vxarq_u64::<8>(a[23], d3);
        b[PI[24]] = vxarq_u64::<50>(a[24], d4);

        // chi: a[x] = b[x] ^ (~b[x+1] & b[x+2]) per row
        for y in 0..5 {
            let r = 5 * y;
            a[r] = vbcaxq_u64(b[r], b[r + 2], b[r + 1]);
            a[r + 1] = vbcaxq_u64(b[r + 1], b[r + 3], b[r + 2]);
            a[r + 2] = vbcaxq_u64(b[r + 2], b[r + 4], b[r + 3]);
            a[r + 3] = vbcaxq_u64(b[r + 3], b[r], b[r + 4]);
            a[r + 4] = vbcaxq_u64(b[r + 4], b[r + 1], b[r]);
        }

        // iota
        a[0] = veorq_u64(a[0], vdupq_n_u64(rc));
    }
}

/// A fresh sponge state holding the same rate lanes in both vector lanes.
#[inline]
#[target_feature(enable = "sha3")]
unsafe fn absorb(lanes: &[u64; 17]) -> [uint64x2_t; 25] {
    let mut a = [vdupq_n_u64(0); 25];
    for (slot, &lane) in a.iter_mut().zip(lanes) {
        *slot = vdupq_n_u64(lane);
    }
    a
}

/// Derive the addresses of two candidates that differ only in their 64-bit
/// counters, following every stage of the job.
///
/// # Safety
/// Requires the aarch64 `sha3` target feature at runtime.
#[target_feature(enable = "sha3")]
pub(crate) unsafe fn address_pair(
    job: &LaneJob,
    counter_a: u64,
    counter_b: u64,
) -> ([u8; 20], [u8; 20]) {
    // candidate A lives in vector lane 0 of every register, B in lane 1
    let counters = vcombine_u64(vcreate_u64(counter_a), vcreate_u64(counter_b));

    // splice the counters into the first message; a negative NEON shift
    // count shifts right, which carries the high bits into the next lane
    let mut a = absorb(&job.first);
    let shift = job.ctr_shift as i64;
    a[job.ctr_lane] = vorrq_u64(a[job.ctr_lane], vshlq_u64(counters, vdupq_n_s64(shift)));
    if shift != 0 {
        a[job.ctr_lane + 1] = vorrq_u64(
            a[job.ctr_lane + 1],
            vshlq_u64(counters, vdupq_n_s64(shift - 64)),
        );
    }
    permute(&mut a);

    if let Some(create2) = &job.guarded_create2 {
        // the 32-byte guard digest (lanes 0..4) becomes the CREATE2 salt at
        // message bytes 21..53: lane 2 from byte 5 on, through lane 6 byte 4
        let (d0, d1, d2, d3) = (a[0], a[1], a[2], a[3]);
        a = absorb(create2);
        a[2] = vorrq_u64(a[2], vshlq_n_u64::<40>(d0));
        a[3] = vorrq_u64(vshrq_n_u64::<24>(d0), vshlq_n_u64::<40>(d1));
        a[4] = vorrq_u64(vshrq_n_u64::<24>(d1), vshlq_n_u64::<40>(d2));
        a[5] = vorrq_u64(vshrq_n_u64::<24>(d2), vshlq_n_u64::<40>(d3));
        a[6] = vorrq_u64(a[6], vshrq_n_u64::<24>(d3));
        permute(&mut a);
    }

    if job.create3 {
        // rlp([proxy, 1]) = 0xd6 0x94 ++ proxy ++ 0x01, where the proxy is
        // digest bytes 12..32 (the high half of lane 1, then lanes 2 and 3),
        // followed directly by the keccak padding byte
        let (l1, l2, l3) = (a[1], a[2], a[3]);
        a = [vdupq_n_u64(0); 25];
        a[0] = vorrq_u64(
            vorrq_u64(
                vdupq_n_u64(0x94d6),
                vshlq_n_u64::<16>(vshrq_n_u64::<32>(l1)),
            ),
            vshlq_n_u64::<48>(l2),
        );
        a[1] = vorrq_u64(vshrq_n_u64::<16>(l2), vshlq_n_u64::<48>(l3));
        a[2] = vorrq_u64(vshrq_n_u64::<16>(l3), vdupq_n_u64(0x0101 << 48));
        a[16] = vdupq_n_u64(0x80 << 56);
        permute(&mut a);
    }

    // the address is digest bytes 12..32
    fn address(lane1: u64, lane2: u64, lane3: u64) -> [u8; 20] {
        let mut out = [0u8; 20];
        out[..4].copy_from_slice(&lane1.to_le_bytes()[4..]);
        out[4..12].copy_from_slice(&lane2.to_le_bytes());
        out[12..].copy_from_slice(&lane3.to_le_bytes());
        out
    }
    (
        address(
            vgetq_lane_u64::<0>(a[1]),
            vgetq_lane_u64::<0>(a[2]),
            vgetq_lane_u64::<0>(a[3]),
        ),
        address(
            vgetq_lane_u64::<1>(a[1]),
            vgetq_lane_u64::<1>(a[2]),
            vgetq_lane_u64::<1>(a[3]),
        ),
    )
}

#[cfg(test)]
mod tests {
    use crate::plan::tests::all_plans;

    /// Pin the vectorized pipeline against the reference derivation on every
    /// plan shape, with counters that exercise both lanes the counter can
    /// straddle.
    #[test]
    fn address_pair_matches_reference_on_every_plan() {
        if !std::arch::is_aarch64_feature_detected!("sha3") {
            eprintln!("skipping: no sha3 target feature at runtime");
            return;
        }

        let counter_pairs = [
            (0u64, 1u64),
            (2, 3),
            (0xdead_beef, 0x00ca_fef0_0d12),
            (u64::MAX - 1, u64::MAX),
            (0x0102_0304_0506_0708, 0x8070_6050_4030_2010),
        ];
        for (index, plan) in all_plans().iter().enumerate() {
            let tag: Vec<u8> = (0..plan.tag_len() as u8).map(|i| 0xc1 + i).collect();
            let job = plan.lane_job(&tag);
            for (counter_a, counter_b) in counter_pairs {
                let (address_a, address_b) =
                    unsafe { super::address_pair(&job, counter_a, counter_b) };
                for (counter, address) in [(counter_a, address_a), (counter_b, address_b)] {
                    assert_eq!(
                        address,
                        plan.address(&plan.salt(&tag, counter)),
                        "plan {index}, counter {counter:#x}"
                    );
                }
            }
        }
    }
}
