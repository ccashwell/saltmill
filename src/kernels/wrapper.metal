/******** Metal wrapper for the bit-interleaved keccak core ********/

#include <metal_stdlib>
using namespace metal;

//__CORE__//

// One thread per candidate. A hit claims the next slot of the solutions
// buffer through an atomic counter, so several hits in one dispatch are all
// reported (up to MAX_SOLUTIONS; the count keeps running past it).
kernel void mine(
  constant uint *d_nonce [[buffer(0)]],
  device atomic_uint *d_count [[buffer(1)]],
  device uint *d_slots [[buffer(2)]],
  uint gid [[thread_position_in_grid]]
) {
  if (hashAndCheck(gid, d_nonce[0])) {
    uint slot = atomic_fetch_add_explicit(d_count, 1u, memory_order_relaxed);
    if (slot < MAX_SOLUTIONS) {
      d_slots[slot] = gid;
    }
  }
}
