# saltmill

A salt miner for vanity contract addresses, for both CREATE2 and CREATE3, that uses the GPU and every CPU core at once. It is built for Apple silicon first: a native Metal kernel on the GPU and a 2-way SHA3 NEON keccak on the CPU.

It brings together two earlier tools: the Apple silicon work on [create2crunch](https://github.com/0age/create2crunch) (CREATE2 only) and a CPU-only CREATE3 miner for [CreateX](https://github.com/pcaversaccio/createx) deployments.

```sh
cargo install saltmill
saltmill --help
```

That needs Rust 1.89 or later. To build from a checkout instead, run `cargo build --release` and use `./target/release/saltmill`.

## Mining

Every input is a named flag. You give a derivation (how a salt becomes an address) and a target (which addresses you want). Each hit is printed to stdout as `<salt> <checksummed address>`; progress goes to stderr, so the output pipes cleanly.

### CREATE3 through CreateX

The address depends only on CreateX, the account that calls it, and the salt. It does not depend on the init code, so one salt gives the same address on every chain and for any constructor arguments.

```sh
saltmill --scheme create3 --caller 0xYourDeployer --prefix 0000000000
```

The salt is a CreateX permissioned salt: bytes 0-19 are the caller, byte 20 is `0x00`, and the remaining 11 bytes are mined. Only that caller can deploy with it, so nobody else can take the address. Deploy with `ICreateX.deployCreate3(salt, initCode)` from that account.

Add `--chain-id <ID>` to turn on CreateX's cross-chain redeploy protection: byte 20 becomes `0x01`, the chain id enters the hash, and the address is valid on that chain only.

### CREATE2

```sh
saltmill --scheme create2 \
  --factory 0x4e59b44847b379578588920cA78FbF26c0B4956C \
  --init-code-hash 0x<keccak256 of creation code ++ abi-encoded constructor args> \
  --prefix c0ffee
```

The salt is `--caller` (20 bytes, zero when omitted) followed by 12 mined bytes. Pass `--caller` for factories that only accept a salt starting with `msg.sender`. The example factory is the deterministic deployment proxy `forge script` uses for `new Contract{salt: salt}(...)`; it takes the salt as given.

### Other factories

| Flags | Derivation |
|---|---|
| `--scheme create2 --factory <F> --init-code-hash <H>` | plain CREATE2 |
| `--scheme create2 --factory 0xba5E…ba5Ed --caller <C> --init-code-hash <H>` | `CreateX.deployCreate2` (guarded salt) |
| `--scheme create3 --caller <C>` | `CreateX.deployCreate3` (guarded salt) |
| `--scheme create3 --factory <F>` | solmate or Solady `CREATE3` used by contract `F` with the salt as given |

The CreateX guard is applied whenever the factory is CreateX, because CreateX always applies it; `--guard none|createx` overrides that for forks and other deployments. `--proxy-init-code-hash` covers CREATE3 factories with a different proxy.

### Targets

All targets given must hold at once.

| Flag | Address must |
|---|---|
| `--prefix <HEX>` | start with these hex characters (odd lengths allowed) |
| `--suffix <HEX>` | end with these hex characters |
| `--hook-flags <FLAGS>` | have `address & 0x3fff == FLAGS`, the 14 Uniswap v4 hook permission bits |
| `--leading-zero-bytes <N>` | start with at least N zero bytes |
| `--total-zero-bytes <N>` | contain at least N zero bytes anywhere |

A v4 hook with `beforeSwap` and `afterSwap` and a vanity prefix:

```sh
saltmill --scheme create2 --factory $FACTORY --init-code-hash $HASH --hook-flags 0x00c0 --prefix c0ffee
```

### Running

| Flag | Effect |
|---|---|
| `--count <N>` | stop after N hits (default 1; 0 mines until interrupted) |
| `--time-limit <SECONDS>` | give up after this long; fails if fewer than `--count` hits were found |
| `--output <FILE>` | also append each hit to a file |
| `--check <SALT>` | print the address a salt deploys to, and fail if it misses the target |
| `--seed <HEX>` | fix the salt bytes that separate one run's search space from another's |
| `--quiet` | print hits and errors only |

### Verifying a salt

Check a salt independently before you depend on it. `--check` recomputes it here; these recompute it elsewhere:

```sh
# CREATE2
cast create2 --deployer $FACTORY --salt $SALT --init-code-hash $HASH

# CreateX CREATE3: simulate the deployment from the caller, on any chain
cast call --from $CALLER 0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed \
  "deployCreate3(bytes32,bytes)(address)" $SALT 0x600160005360016000f3
```

## Compute

`--compute auto` (the default) mines on the GPU and the CPU together, and on the CPU alone when no GPU can be used. `cpu`, `gpu` and `hybrid` force a choice; `hybrid` fails if the GPU is unusable.

- **GPU, Metal** (macOS). The kernel keeps each 64-bit keccak lane as two 32-bit words of even and odd bits, because Apple GPUs have no 64-bit integer ALU. All three hashes of a CREATE3 candidate run in that form in one kernel. Batches are `2^--gpu-batch-bits` candidates (default 26).
- **GPU, OpenCL** (`cargo build --release --features opencl`, then `--backend opencl`). The same kernel for other GPUs. On macOS OpenCL is a slow shim over Metal; use Metal there.
- **CPU.** One thread per core (`--threads`), one core held back for the GPU's host thread when the GPU mines. With the ARMv8.4 SHA3 extension (every Apple M-series chip) each thread hashes two candidates at once in the NEON registers; other CPUs use scalar CRYPTOGAMS assembly (`--cpu-kernel` forces either).

The CPU starts immediately and the GPU joins after its kernel compiles, about 1.5 seconds. Before mining, the compiled kernel is aimed at a candidate whose address the host already knows and must find exactly that one. Every hit, from any miner, is recomputed on the host through a separate plain keccak before it is printed; a mismatch stops the run.

### Performance

Measured on an Apple M4 Max laptop (14-core CPU, 40-core GPU) that was already warm, in millions of salts per second. A CREATE3 salt costs three hashes, so it runs at about a third of the CREATE2 rate.

| | CPU (2-way SHA3 NEON) | GPU (Metal) | Hybrid |
|---|---|---|---|
| CREATE2 | ~155 | ~660 | ~650-700 |
| CREATE3 via CreateX | ~50 | ~207 | ~140-200 |

The GPU and CPU share one power and thermal budget. Once the chassis is hot the GPU throttles, and adding the CPU can take as much from the GPU as it contributes: in back-to-back 15-second CREATE3 runs on a hot machine, hybrid averaged about 140 M/s against 150-160 M/s for `--compute gpu`. For long runs on a laptop, compare the two on your machine.

## How the search space is laid out

```text
CREATE2 / unguarded     caller (20) | tag (4) | counter (8)
CreateX                 caller (20) | flag (1) | tag (3) | counter (8)
```

The tag is random per run (or `--seed`), and its last bit differs between the CPU and the GPU, so the two never try the same salt. The counter's top byte is the CPU thread index; on the GPU the counter is the position in the batch and the dispatch number.

## Development

```sh
cargo test                      # includes Metal kernel tests on macOS
cargo test --features opencl    # also runs the kernel through OpenCL
cargo clippy --all-targets --features opencl
```

The expected addresses in the tests come from `cast`, and the CreateX ones were also reproduced by simulating `deployCreate3` on mainnet and Base.

## License

MIT. The keccak kernels derive from [create2crunch by 0age](https://github.com/0age/create2crunch).
