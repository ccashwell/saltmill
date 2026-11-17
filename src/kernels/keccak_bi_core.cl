/******** Keccak-f[1600], bit-interleaved, shared OpenCL/Metal core ********/

/**
* Bit-interleaved keccak-256 salt miner core for GPUs without native 64-bit
* integer ALUs (notably Apple silicon, where every ulong rotate/xor of a
* 64-bit kernel is emulated over multiple 32-bit operations).
*
* Each 64-bit keccak lane is stored as two 32-bit words: one packing the
* lane's even-numbered bits, one the odd-numbered bits. XOR/AND/NOT apply
* componentwise, and a 64-bit rotation by n becomes one 32-bit rotation of
* each component (with the components swapping places when n is odd) - all
* native single-cycle operations on a 32-bit ALU. A shift by an even number
* of bits 2k is a shift of each component by k, which is how the bytes of
* one digest are carried into the next message without ever leaving the
* interleaved form.
*
* One candidate is up to three chained single-block hashes:
*
*   guard   (HAS_GUARD)   guarded = keccak256(first message)
*   create2               address = keccak256(0xff ++ factory ++ salt ++ hash)[12..]
*   create  (HAS_CREATE)  address = keccak256(0xd6 0x94 ++ address ++ 0x01)[12..]
*
* This file is pure C, compiled as either OpenCL C or Metal Shading
* Language; the per-API wrapper defines the entry point.
*
* The host defines, all bit-interleaved (_E even bits, _O odd bits):
*   M{i}_E/_O, i in 0..17   rate lanes of the first message, counter bytes zero
*   CTR_LANE, CTR_SHIFT     lane and bit offset (multiple of 8) of the 64-bit
*                           counter (gid | nonce << 32) in the first message
*   HAS_GUARD               0/1; when 1 also P{i}_E/_O, i in 0..17: rate lanes
*                           of the CREATE2 message with the salt bytes zero
*   HAS_CREATE              0/1; when 1 also C0, C2, C16: the constant parts
*                           of the CREATE message lanes
*   RC{i}_E/_O, i in 0..24  iota round constants
*   hasPattern(w)           the address test over five little-endian words
*   TOTAL_ZEROES            optional minimum number of zero address bytes
*/

#define rotl32(x, s) rotate((uint)(x), (uint)(s))

// rotate a bit-interleaved lane (e, o) left by the constant n
#define ROLE(e, o, n) (((n) & 1) ? rotl32((o), (((n) + 1) >> 1)) : rotl32((e), ((n) >> 1)))
#define ROLO(e, o, n) (((n) & 1) ? rotl32((e), (((n) - 1) >> 1)) : rotl32((o), ((n) >> 1)))

// perfect unshuffle: gather the even bits of x into the low half-word and
// the odd bits into the high half-word
static inline uint unshuffle(uint x)
{
  uint t;
  t = (x ^ (x >> 1)) & 0x22222222u; x ^= t ^ (t << 1);
  t = (x ^ (x >> 2)) & 0x0c0c0c0cu; x ^= t ^ (t << 2);
  t = (x ^ (x >> 4)) & 0x00f000f0u; x ^= t ^ (t << 4);
  t = (x ^ (x >> 8)) & 0x0000ff00u; x ^= t ^ (t << 8);
  return x;
}

// inverse of unshuffle
static inline uint shuffle(uint x)
{
  uint t;
  t = (x ^ (x >> 8)) & 0x0000ff00u; x ^= t ^ (t << 8);
  t = (x ^ (x >> 4)) & 0x00f000f0u; x ^= t ^ (t << 4);
  t = (x ^ (x >> 2)) & 0x0c0c0c0cu; x ^= t ^ (t << 2);
  t = (x ^ (x >> 1)) & 0x22222222u; x ^= t ^ (t << 1);
  return x;
}

// interleave the 64-bit value (lo, hi) and OR it into lane i
#define OR_INTERLEAVED(i, lo, hi) \
lo = unshuffle(lo); \
hi = unshuffle(hi); \
ae[i] |= (lo & 0xffffu) | (hi << 16); \
ao[i] |= (lo >> 16) | (hi & 0xffff0000u);

#define parities() \
be[0] = ae[0] ^ ae[5] ^ ae[10] ^ ae[15] ^ ae[20]; \
bo[0] = ao[0] ^ ao[5] ^ ao[10] ^ ao[15] ^ ao[20]; \
be[1] = ae[1] ^ ae[6] ^ ae[11] ^ ae[16] ^ ae[21]; \
bo[1] = ao[1] ^ ao[6] ^ ao[11] ^ ao[16] ^ ao[21]; \
be[2] = ae[2] ^ ae[7] ^ ae[12] ^ ae[17] ^ ae[22]; \
bo[2] = ao[2] ^ ao[7] ^ ao[12] ^ ao[17] ^ ao[22]; \
be[3] = ae[3] ^ ae[8] ^ ae[13] ^ ae[18] ^ ae[23]; \
bo[3] = ao[3] ^ ao[8] ^ ao[13] ^ ao[18] ^ ao[23]; \
be[4] = ae[4] ^ ae[9] ^ ae[14] ^ ae[19] ^ ae[24]; \
bo[4] = ao[4] ^ ao[9] ^ ao[14] ^ ao[19] ^ ao[24];

#define theta_(m, n, x) \
te = be[m] ^ rotl32(bo[n], 1); \
to = bo[m] ^ be[n]; \
ae[x + 0] ^= te; \
ae[x + 5] ^= te; \
ae[x + 10] ^= te; \
ae[x + 15] ^= te; \
ae[x + 20] ^= te; \
ao[x + 0] ^= to; \
ao[x + 5] ^= to; \
ao[x + 10] ^= to; \
ao[x + 15] ^= to; \
ao[x + 20] ^= to;

#define theta() \
parities(); \
theta_(4, 1, 0); \
theta_(0, 2, 1); \
theta_(1, 3, 2); \
theta_(2, 4, 3); \
theta_(3, 0, 4);

#define rhoPi_(m, n) \
te = be[0]; to = bo[0]; \
be[0] = ae[m]; bo[0] = ao[m]; \
ae[m] = ROLE(te, to, n); \
ao[m] = ROLO(te, to, n);

#define rhoPi() \
te = ae[1]; to = ao[1]; \
be[0] = ae[10]; bo[0] = ao[10]; \
ae[10] = rotl32(to, 1); ao[10] = te; \
rhoPi_(7, 3); \
rhoPi_(11, 6); \
rhoPi_(17, 10); \
rhoPi_(18, 15); \
rhoPi_(3, 21); \
rhoPi_(5, 28); \
rhoPi_(16, 36); \
rhoPi_(8, 45); \
rhoPi_(21, 55); \
rhoPi_(24, 2); \
rhoPi_(4, 14); \
rhoPi_(15, 27); \
rhoPi_(23, 41); \
rhoPi_(19, 56); \
rhoPi_(13, 8); \
rhoPi_(12, 25); \
rhoPi_(2, 43); \
rhoPi_(20, 62); \
rhoPi_(14, 18); \
rhoPi_(22, 39); \
rhoPi_(9, 61); \
rhoPi_(6, 20); \
rhoPi_(1, 44);

#define chi_(n) \
be[0] = ae[n + 0]; be[1] = ae[n + 1]; be[2] = ae[n + 2]; be[3] = ae[n + 3]; be[4] = ae[n + 4]; \
bo[0] = ao[n + 0]; bo[1] = ao[n + 1]; bo[2] = ao[n + 2]; bo[3] = ao[n + 3]; bo[4] = ao[n + 4]; \
ae[n + 0] = be[0] ^ ((~be[1]) & be[2]); \
ae[n + 1] = be[1] ^ ((~be[2]) & be[3]); \
ae[n + 2] = be[2] ^ ((~be[3]) & be[4]); \
ae[n + 3] = be[3] ^ ((~be[4]) & be[0]); \
ae[n + 4] = be[4] ^ ((~be[0]) & be[1]); \
ao[n + 0] = bo[0] ^ ((~bo[1]) & bo[2]); \
ao[n + 1] = bo[1] ^ ((~bo[2]) & bo[3]); \
ao[n + 2] = bo[2] ^ ((~bo[3]) & bo[4]); \
ao[n + 3] = bo[3] ^ ((~bo[4]) & bo[0]); \
ao[n + 4] = bo[4] ^ ((~bo[0]) & bo[1]);

#define chi() chi_(0); chi_(5); chi_(10); chi_(15); chi_(20);

#define iteration(re, ro) theta(); rhoPi(); chi(); ae[0] ^= re; ao[0] ^= ro;

// Rounds 1..23 in full, then only as much of round 24 as the digest needs:
// its first 32 bytes are lanes 0..3, all in chi row 0, whose inputs are the
// rho+pi images of lanes 0, 6, 12, 18 and 24. Leaves the digest lanes in
// l0..l3; a stage that only wants the address ignores l0 and the low half
// of l1, and the compiler drops their computation.
#define permuteToDigest() \
iteration(RC0_E, RC0_O); \
iteration(RC1_E, RC1_O); \
iteration(RC2_E, RC2_O); \
iteration(RC3_E, RC3_O); \
iteration(RC4_E, RC4_O); \
iteration(RC5_E, RC5_O); \
iteration(RC6_E, RC6_O); \
iteration(RC7_E, RC7_O); \
iteration(RC8_E, RC8_O); \
iteration(RC9_E, RC9_O); \
iteration(RC10_E, RC10_O); \
iteration(RC11_E, RC11_O); \
iteration(RC12_E, RC12_O); \
iteration(RC13_E, RC13_O); \
iteration(RC14_E, RC14_O); \
iteration(RC15_E, RC15_O); \
iteration(RC16_E, RC16_O); \
iteration(RC17_E, RC17_O); \
iteration(RC18_E, RC18_O); \
iteration(RC19_E, RC19_O); \
iteration(RC20_E, RC20_O); \
iteration(RC21_E, RC21_O); \
iteration(RC22_E, RC22_O); \
parities(); \
{ \
  uint a0e = ae[0] ^ be[4] ^ rotl32(bo[1], 1); \
  uint a0o = ao[0] ^ bo[4] ^ be[1]; \
  uint a6e = ae[6] ^ be[0] ^ rotl32(bo[2], 1); \
  uint a6o = ao[6] ^ bo[0] ^ be[2]; \
  uint a12e = ae[12] ^ be[1] ^ rotl32(bo[3], 1); \
  uint a12o = ao[12] ^ bo[1] ^ be[3]; \
  uint a18e = ae[18] ^ be[2] ^ rotl32(bo[4], 1); \
  uint a18o = ao[18] ^ bo[2] ^ be[4]; \
  uint a24e = ae[24] ^ be[3] ^ rotl32(bo[0], 1); \
  uint a24o = ao[24] ^ bo[3] ^ be[0]; \
  uint b1e = ROLE(a6e, a6o, 44), b1o = ROLO(a6e, a6o, 44); \
  uint b2e = ROLE(a12e, a12o, 43), b2o = ROLO(a12e, a12o, 43); \
  uint b3e = ROLE(a18e, a18o, 21), b3o = ROLO(a18e, a18o, 21); \
  uint b4e = ROLE(a24e, a24o, 14), b4o = ROLO(a24e, a24o, 14); \
  l0e = a0e ^ ((~b1e) & b2e) ^ RC23_E; l0o = a0o ^ ((~b1o) & b2o) ^ RC23_O; \
  l1e = b1e ^ ((~b2e) & b3e); l1o = b1o ^ ((~b2o) & b3o); \
  l2e = b2e ^ ((~b3e) & b4e); l2o = b2o ^ ((~b3o) & b4o); \
  l3e = b3e ^ ((~b4e) & a0e); l3o = b3o ^ ((~b4o) & a0o); \
}

#define clearCapacity() \
ae[17] = 0; ao[17] = 0; \
ae[18] = 0; ao[18] = 0; \
ae[19] = 0; ao[19] = 0; \
ae[20] = 0; ao[20] = 0; \
ae[21] = 0; ao[21] = 0; \
ae[22] = 0; ao[22] = 0; \
ae[23] = 0; ao[23] = 0; \
ae[24] = 0; ao[24] = 0;

// number of zero bytes in a 32-bit word
#define zeroBytes(w) ( \
  (uint)(((w) & 0x000000ffu) == 0u) + (uint)(((w) & 0x0000ff00u) == 0u) + \
  (uint)(((w) & 0x00ff0000u) == 0u) + (uint)(((w) & 0xff000000u) == 0u))

// derive one candidate's address and report whether it meets the target
static inline bool hashAndCheck(uint gid, uint dn)
{
  uint ae[25];
  uint ao[25];
  uint be[5];
  uint bo[5];
  uint te, to;
  uint lo, hi;
  uint l0e, l0o, l1e, l1o, l2e, l2o, l3e, l3o;

  // the first message, interleaved at kernel build time
  ae[0] = M0_E; ao[0] = M0_O;
  ae[1] = M1_E; ao[1] = M1_O;
  ae[2] = M2_E; ao[2] = M2_O;
  ae[3] = M3_E; ao[3] = M3_O;
  ae[4] = M4_E; ao[4] = M4_O;
  ae[5] = M5_E; ao[5] = M5_O;
  ae[6] = M6_E; ao[6] = M6_O;
  ae[7] = M7_E; ao[7] = M7_O;
  ae[8] = M8_E; ao[8] = M8_O;
  ae[9] = M9_E; ao[9] = M9_O;
  ae[10] = M10_E; ao[10] = M10_O;
  ae[11] = M11_E; ao[11] = M11_O;
  ae[12] = M12_E; ao[12] = M12_O;
  ae[13] = M13_E; ao[13] = M13_O;
  ae[14] = M14_E; ao[14] = M14_O;
  ae[15] = M15_E; ao[15] = M15_O;
  ae[16] = M16_E; ao[16] = M16_O;
  clearCapacity();

  // splice in the counter gid | dn << 32, little-endian at bit CTR_SHIFT of
  // lane CTR_LANE; a non-zero shift spills its high bits into the next lane
#if CTR_SHIFT == 0
  lo = gid;
  hi = dn;
  OR_INTERLEAVED(CTR_LANE, lo, hi);
#elif CTR_SHIFT < 32
  lo = gid << CTR_SHIFT;
  hi = (dn << CTR_SHIFT) | (gid >> (32 - CTR_SHIFT));
  OR_INTERLEAVED(CTR_LANE, lo, hi);
  lo = dn >> (32 - CTR_SHIFT);
  hi = 0;
  OR_INTERLEAVED(CTR_LANE + 1, lo, hi);
#elif CTR_SHIFT == 32
  lo = 0;
  hi = gid;
  OR_INTERLEAVED(CTR_LANE, lo, hi);
  lo = dn;
  hi = 0;
  OR_INTERLEAVED(CTR_LANE + 1, lo, hi);
#else
  lo = 0;
  hi = gid << (CTR_SHIFT - 32);
  OR_INTERLEAVED(CTR_LANE, lo, hi);
  lo = (gid >> (64 - CTR_SHIFT)) | (dn << (CTR_SHIFT - 32));
  hi = dn >> (64 - CTR_SHIFT);
  OR_INTERLEAVED(CTR_LANE + 1, lo, hi);
#endif

  permuteToDigest();

#if HAS_GUARD
  // the 32-byte guard digest becomes the CREATE2 salt at message bytes
  // 21..53: a left shift by 40 bits from lane 2 on, i.e. by 20 bits in each
  // interleaved component, with the 12 bits shifted out carried forward
  ae[0] = P0_E; ao[0] = P0_O;
  ae[1] = P1_E; ao[1] = P1_O;
  ae[2] = P2_E | (l0e << 20); ao[2] = P2_O | (l0o << 20);
  ae[3] = (l0e >> 12) | (l1e << 20); ao[3] = (l0o >> 12) | (l1o << 20);
  ae[4] = (l1e >> 12) | (l2e << 20); ao[4] = (l1o >> 12) | (l2o << 20);
  ae[5] = (l2e >> 12) | (l3e << 20); ao[5] = (l2o >> 12) | (l3o << 20);
  ae[6] = P6_E | (l3e >> 12); ao[6] = P6_O | (l3o >> 12);
  ae[7] = P7_E; ao[7] = P7_O;
  ae[8] = P8_E; ao[8] = P8_O;
  ae[9] = P9_E; ao[9] = P9_O;
  ae[10] = P10_E; ao[10] = P10_O;
  ae[11] = P11_E; ao[11] = P11_O;
  ae[12] = P12_E; ao[12] = P12_O;
  ae[13] = P13_E; ao[13] = P13_O;
  ae[14] = P14_E; ao[14] = P14_O;
  ae[15] = P15_E; ao[15] = P15_O;
  ae[16] = P16_E; ao[16] = P16_O;
  clearCapacity();

  permuteToDigest();
#endif

#if HAS_CREATE
  // rlp([proxy, 1]) = 0xd6 0x94 ++ proxy ++ 0x01, then the keccak padding.
  // The proxy is digest bytes 12..32 (the high half of lane 1, then lanes 2
  // and 3) and lands at message bytes 2..22.
  ae[0] = C0_E | ((l1e >> 16) << 8) | (l2e << 24);
  ao[0] = C0_O | ((l1o >> 16) << 8) | (l2o << 24);
  ae[1] = (l2e >> 8) | (l3e << 24);
  ao[1] = (l2o >> 8) | (l3o << 24);
  ae[2] = C2_E | (l3e >> 8);
  ao[2] = C2_O | (l3o >> 8);
  ae[3] = 0; ao[3] = 0;
  ae[4] = 0; ao[4] = 0;
  ae[5] = 0; ao[5] = 0;
  ae[6] = 0; ao[6] = 0;
  ae[7] = 0; ao[7] = 0;
  ae[8] = 0; ao[8] = 0;
  ae[9] = 0; ao[9] = 0;
  ae[10] = 0; ao[10] = 0;
  ae[11] = 0; ao[11] = 0;
  ae[12] = 0; ao[12] = 0;
  ae[13] = 0; ao[13] = 0;
  ae[14] = 0; ao[14] = 0;
  ae[15] = 0; ao[15] = 0;
  ae[16] = C16_E; ao[16] = C16_O;
  clearCapacity();

  permuteToDigest();
#endif

  // de-interleave the address (digest bytes 12..32) into five
  // little-endian words
  uint w[5];
  w[0] = shuffle((l1e >> 16) | (l1o & 0xffff0000u));
  w[1] = shuffle((l2e & 0xffffu) | ((l2o & 0xffffu) << 16));
  w[2] = shuffle((l2e >> 16) | (l2o & 0xffff0000u));
  w[3] = shuffle((l3e & 0xffffu) | ((l3o & 0xffffu) << 16));
  w[4] = shuffle((l3e >> 16) | (l3o & 0xffff0000u));

  return hasPattern(w)
#ifdef TOTAL_ZEROES
    && (zeroBytes(w[0]) + zeroBytes(w[1]) + zeroBytes(w[2]) + zeroBytes(w[3]) + zeroBytes(w[4])
        >= TOTAL_ZEROES)
#endif
  ;
}
