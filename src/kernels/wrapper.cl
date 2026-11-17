/******** OpenCL wrapper for the bit-interleaved keccak core ********/

//__CORE__//

// One work item per candidate. A hit claims the next slot of the solutions
// buffer through an atomic counter, so several hits in one dispatch are all
// reported (up to MAX_SOLUTIONS; the count keeps running past it).
__kernel void mine(
  __constant uint const *d_nonce,
  __global volatile uint *d_count,
  __global uint *d_slots
) {
  uint gid = (uint)get_global_id(0);

  if (hashAndCheck(gid, d_nonce[0])) {
    uint slot = atomic_inc(d_count);
    if (slot < MAX_SOLUTIONS) {
      d_slots[slot] = gid;
    }
  }
}
