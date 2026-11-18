/******** OpenCL wrapper for the bit-interleaved keccak core ********/

#define SPACE_CONSTANT __constant

//__CORE__//

// One work item per candidate. A hit claims the next slot of the solutions
// buffer through an atomic counter, so several hits in one dispatch are all
// reported (up to MAX_SOLUTIONS; the count keeps running past it).
// d_input is the dispatch nonce followed by the eleven target words.
__kernel void mine(
  __constant uint const *d_input,
  __global volatile uint *d_count,
  __global uint *d_slots
) {
  uint gid = (uint)get_global_id(0);

  if (hashAndCheck(gid, d_input[0], d_input + 1)) {
    uint slot = atomic_inc(d_count);
    if (slot < MAX_SOLUTIONS) {
      d_slots[slot] = gid;
    }
  }
}
