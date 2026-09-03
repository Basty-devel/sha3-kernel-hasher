# SHA3-Kernel-Hasher

[![Crates.io](https://img.shields.io/crates/v/sha3-kernel-hasher.svg)](https://crates.io/crates/sha3-kernel-hasher)
[![docs.rs](https://docs.rs/sha3-kernel-hasher/badge.svg)](https://docs.rs/sha3-kernel-hasher)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![NIST CAVP](https://img.shields.io/badge/NIST_CAVP-validated-brightgreen.svg)](#71-nist-cavp-test-vectors)

> **Production-grade SHA3-512 (FIPS 202) implementation in pure Rust with
> dual-mode compilation for userspace and Windows kernel drivers.**

---

## Abstract

`sha3-kernel-hasher` is a self-contained implementation of the SHA3-512
cryptographic hash function as standardised in FIPS PUB 202 (National
Institute of Standards and Technology, 2015).  The crate provides the
complete Keccak-f[1600] permutation in safe Rust, correct SHA3 domain
separation and `pad10*1` padding, and an incremental (streaming) API
suitable for both userspace applications and `no_std` Windows kernel
drivers.

The implementation has been validated against the NIST Cryptographic
Algorithm Validation Program (CAVP) SHA3-512 test vectors and
cross-validated against the RustCrypto `sha3` reference crate (v0.10)
for 20 distinct input lengths covering all sponge-rate boundary
conditions.

---

## Table of Contents

1. [Theoretical Background](#1-theoretical-background)
2. [Architecture](#2-architecture)
3. [Module Reference](#3-module-reference)
4. [Installation and Usage](#4-installation-and-usage)
5. [Feature Flags](#5-feature-flags)
6. [Security Considerations](#6-security-considerations)
7. [Correctness Validation](#7-correctness-validation)
8. [Performance](#8-performance)
9. [Publishing to crates.io](#9-publishing-to-cratesio)
10. [Contributing](#10-contributing)
11. [Licence](#11-licence)
12. [References](#12-references)

---

## 1. Theoretical Background

### 1.1 The Keccak Sponge Construction

SHA3 is built on the *sponge construction* introduced by Bertoni et al.
(2011).  A sponge operates on a fixed-size state of *b* bits, divided
into a *rate* *r* and a *capacity* *c* such that *b = r + c*.  For
SHA3-512, *b* = 1600, *r* = 576 bits (72 bytes), and *c* = 1024 bits,
yielding a security level of 256 bits against collision attacks and
512 bits against preimage attacks (NIST, 2015, Section 4).

The sponge has two phases:

- **Absorb** — The message is padded using the `pad10*1` rule with
  domain separation byte `0x06` (FIPS 202, Section 6.1), then split
  into *r*-bit blocks.  Each block is XORed into the first *r* bits of
  the state, followed by an application of the Keccak-f[1600]
  permutation.
- **Squeeze** — After absorbing all blocks, the first 512 bits of the
  state are extracted as the hash digest.

### 1.2 The Keccak-f[1600] Permutation

The permutation comprises 24 rounds (*n_r* = 12 + 2*l*, where *l* = 6
for *w* = 64).  Each round applies five step mappings to the 5x5 matrix
of 64-bit *lanes* (Bertoni et al., 2011, Section 1.2):

| Step | Symbol | Purpose |
|------|--------|---------|
| Theta (θ) | Column-parity diffusion | XOR each bit with the parity of two adjacent columns |
| Rho (ρ) | Intra-lane rotation | Rotate each lane by a fixed offset (FIPS 202, §3.2.2) |
| Pi (π) | Inter-lane transposition | Permute lane positions according to a linear map |
| Chi (χ) | Non-linear mixing | Apply a degree-2 Boolean function row-wise |
| Iota (ι) | Symmetry breaking | XOR a round constant into lane (0,0) |

The round constants are derived from an LFSR-based generation rule
(FIPS 202, Section 3.2.5) and are pre-computed in `src/keccak.rs`.

### 1.3 Security Properties

SHA3-512 provides the following security guarantees, assuming
Keccak-f[1600] behaves as a random permutation (Bertoni et al., 2011):

| Property | Security (bits) |
|----------|----------------|
| Collision resistance | 256 |
| Preimage resistance | 512 |
| Second preimage resistance | 512 |

SHA3 is structurally distinct from the Merkle-Damgard family (SHA-1,
SHA-2) and is therefore immune to length-extension attacks by design
(Kelsey and Schneier, 2005).

---

## 2. Architecture

### 2.1 Crate Layout

```
sha3-kernel-hasher/
├── src/
│   ├── lib.rs          # Public API, sponge absorb/finalize, hash_many batching, SIMD detection
│   ├── keccak.rs       # Keccak-f[1600] permutation (24-round, all 5 steps) — scalar reference
│   ├── simd_avx512.rs  # Single-message AVX-512 permutation (opt-in, not a measured win — §5.1)
│   ├── simd_parallel.rs # Multi-message AVX-512x8 / AVX2x4 permutation (real win — §5.1)
│   ├── hex.rs          # Sha3Digest wrapper, hex encoding/decoding, Display
│   ├── ct.rs           # Constant-time comparison primitives
│   └── io.rs           # Streaming I/O hashing (std::io::Read adapter)
├── examples/
│   ├── basic.rs        # Minimal one-shot and incremental hashing
│   ├── kernel.rs       # Simulated kernel memory-region hashing
│   └── benchmark.rs    # Standalone throughput measurement
├── benches/
│   └── sha3_bench.rs   # Criterion-based benchmarks
├── Cargo.toml
└── README.md           # This document
```

### 2.2 Compilation Modes

The crate supports two compilation targets, controlled by Cargo features:

| Mode | Feature | `std` | `alloc` | Use Case |
|------|---------|-------|---------|----------|
| Userspace (default) | `std` | Yes | Yes | Applications, services, CLI tools |
| Kernel | `kernel` | No | Yes | Windows kernel-mode drivers (WDM/KMDF) |

In kernel mode, the `io` module is disabled (it depends on `std::io`),
and so are `simd_avx512`/`simd_parallel` (both need
`is_x86_feature_detected!`, a `std`-only macro with no `core`
equivalent) — kernel-mode hashing is always scalar.
`kernel_safe::save_processor_state`/`restore_processor_state` exist for
a driver that needs to touch SIMD registers around *its own* code
safely (they wrap `KeSaveExtendedProcessorState`, Microsoft, 2023).

### 2.3 SIMD Status

Runtime SIMD detection probes for AVX-512F/BW/DQ, AVX2+BMI2, and
SSE4.2 via `is_x86_feature_detected!` and is exposed for callers to query
via `simd_features()`. On `std` + `x86_64` builds, two independent SIMD
paths exist and are genuinely wired up — see §5.1 for full numbers:

- **`hash()`/`update()`/`finalize()`** (single message): `use_avx512`
  dispatches to a real AVX-512 permutation, but it's not a throughput
  win on this crate's own benchmarks — defaults to `false`.
- **`hash_many()`** (a batch of independent messages): `use_avx512`
  dispatches to a real AVX-512×8 batched permutation that *is* a
  substantial win (~5.5x); `use_avx2` gives a modest ~1.2x via an
  AVX2×4 batched permutation.

Read §5.1 before enabling `use_avx512` on the single-message path —
"real" and "faster" are not the same claim here.

---

## 3. Module Reference

### 3.1 `Sha3_512Kernel` — Primary Hasher

The main entry point.  Provides one-shot hashing, incremental
(`update`/`finalize`) streaming, and memory-region hashing.

```rust
use sha3_kernel_hasher::Sha3_512Kernel;

let mut hasher = Sha3_512Kernel::new();

// One-shot hashing
let hash = hasher.hash(b"FIPS 202 compliance test");

// Incremental hashing
hasher.reset();
hasher.update(b"first ");
hasher.update(b"second");
let hash = hasher.finalize();
```

Batch hashing of independent messages (`std` + `x86_64`), where the
AVX-512×8 SIMD path is a real ~5.5x throughput win over a scalar loop
(§5.1). Messages may have different lengths; digests come back in input
order:

```rust
use sha3_kernel_hasher::{PerformanceConfig, Sha3_512Kernel};

let hasher = Sha3_512Kernel::with_config(PerformanceConfig {
    use_avx512: true,
    ..Default::default()
});

let messages: Vec<&[u8]> = vec![b"first", b"second", b"a much longer third message"];
let digests = hasher.hash_many(&messages);
assert_eq!(digests.len(), 3);
```

If the CPU lacks AVX-512 (or the flags are off), `hash_many` falls back
to AVX2×4 batching, then to hashing each message individually — the
output is identical either way.

### 3.2 `Sha3Digest` — Typed Hash Output (`hex` module)

A newtype wrapper over `[u8; 64]` with `Display`, `LowerHex`,
`UpperHex`, `Debug`, `Eq`, `Hash`, and hex encode/decode methods.

```rust
use sha3_kernel_hasher::{Sha3_512Kernel, Sha3Digest};

let mut hasher = Sha3_512Kernel::new();
let digest: Sha3Digest = hasher.hash_digest(b"hello");

println!("lowercase : {}", digest);       // Display
println!("uppercase : {:X}", digest);      // UpperHex
println!("debug     : {:?}", digest);      // Debug

// Round-trip hex encoding
let hex_str = digest.to_hex_lower();
let parsed  = Sha3Digest::from_hex(&hex_str).unwrap();
assert_eq!(parsed, digest);
```

### 3.3 `ct` — Constant-Time Comparison

Timing-safe equality functions to prevent side-channel leakage when
comparing digests (Kocher, 1996).

```rust
use sha3_kernel_hasher::ct::{ct_eq, ct_hash_eq};

let a = [0xABu8; 64];
let b = [0xABu8; 64];
assert!(ct_hash_eq(&a, &b));   // fixed-size, unrolled
assert!(ct_eq(&a, &b));        // variable-length
```

### 3.4 `io` — Streaming File / Reader Hashing (std only)

Hash arbitrary `std::io::Read` sources with an 8 KiB internal buffer.

```rust
use sha3_kernel_hasher::io::{hash_file, hash_reader};
use std::io::Cursor;

// Hash a file by path
let digest = hash_file("firmware.bin").unwrap();

// Hash any reader
let data = b"stream data";
let digest = hash_reader(Cursor::new(data)).unwrap();
println!("{}", digest);
```

### 3.5 `kernel_safe` — Processor State Management

Provides `save_processor_state` / `restore_processor_state` for
safely using SIMD registers in kernel-mode contexts where
floating-point / SIMD state is not automatically preserved.

### 3.6 `performance` — Benchmarking Utilities (std only)

Simple wall-clock throughput measurement for integration testing.

---

## 4. Installation and Usage

### 4.1 Cargo Dependency

```toml
[dependencies]
sha3-kernel-hasher = "0.3.0"
```

For kernel drivers:

```toml
[dependencies]
sha3-kernel-hasher = { version = "0.3.0", default-features = false, features = ["kernel"] }
```

### 4.2 Minimum Supported Rust Version (MSRV)

The crate requires **Rust 1.89** or later — raised from 1.73 for the
AVX-512 SIMD path (§5.1), which uses `_mm512_ternarylogic_epi64`,
`_mm512_permutexvar_epi64`, `_mm512_rolv_epi64`, and
`_mm512_i64gather_epi64`, all stabilized in Rust 1.89.0. The `kernel`
(`no_std`) build does not include the AVX-512 module and has no
additional MSRV requirement beyond the base 1.73 it always had, but the
crate declares one `rust-version` for the whole package, so it now reads
1.89 regardless of which features are enabled.

### 4.3 Running Tests

```bash
cargo test -p sha3-kernel-hasher
```

### 4.4 Running Benchmarks

```bash
cargo bench -p sha3-kernel-hasher
```

### 4.5 Running the Standalone Benchmark Example

```bash
cargo run --example benchmark --release -p sha3-kernel-hasher
```

---

## 5. Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support, enables `io` module and runtime SIMD *detection* |
| `kernel` | No | `no_std` + `alloc` mode for Windows kernel drivers (implies `no_std`) |
| `serde` | No | `Serialize` / `Deserialize` derives for state and hash types |
| `benchmarks` | No | Enables criterion benchmark harness integration |

There is no `avx2`/`avx512` Cargo feature — SIMD is runtime-dispatched
(`is_x86_feature_detected!`) behind `PerformanceConfig`, not
feature-gated. See §5.1 for what's actually accelerated.

### 5.1 Current SIMD status — read before depending on a performance claim

`Sha3_512Kernel` detects AVX-512F/BW/DQ, AVX2+BMI2, and SSE4.2 at runtime
(`simd_features()`). `PerformanceConfig`'s `use_avx512`/`use_avx2` fields
are real and wired up on `std` + `x86_64` builds, but they answer two
different questions depending on which method reads them:

**Single-message (`hash()`/`update()`/`finalize()`) — not a win, off by
default.** `use_avx512` dispatches every Keccak-f[1600] call to a real,
cross-validated AVX-512 permutation (`src/simd_avx512.rs`) — this is not
a stub, it passes 2000 random-state trials plus 50 chained permutations
against the scalar implementation. It just isn't *faster*: a single
Keccak-f[1600] instance has no independent lanes for a 512-bit register
to parallelize within, and π's cross-row mixing costs real shuffle
instructions scalar code doesn't pay. Measured via
`cargo bench --bench sha3_bench sha3_512_throughput` (this container's
Xeon):

```
sha3_512_throughput/1MB_scalar    [180.66 MiB/s 181.96 MiB/s 183.33 MiB/s]
sha3_512_throughput/1MB_auto      [174.25 MiB/s 175.71 MiB/s 177.20 MiB/s]  (~3-4% slower)
```

`use_avx2` has no effect on this path at all — AVX2 has no per-lane
*variable* 64-bit rotate instruction, which ρ needs; only AVX-512F
provides one (`_mm512_rolv_epi64`). `use_avx512` therefore **defaults to
`false`** here. Set it `true` only after benchmarking your own workload.

**Batch (`hash_many()`) — a real win, still off by default pending your
own benchmark.** Hashing a batch of *independent* messages gives the
vector width genuine parallel work, unlike single-message hashing above:
each SIMD lane holds a different message's state, so ρ+π reduce to array
indexing plus a uniform (not gathered) rotate. Measured end to end
(`cargo bench --bench sha3_bench sha3_512_hash_many`, 64 independent
4 KiB messages):

| Path | Throughput | vs. scalar loop |
|------|-----------|------------------|
| scalar (one `hash()` per message) | ~182 MiB/s | 1.0x |
| AVX2×4 batching | ~220 MiB/s | ~1.2x |
| AVX-512×8 batching | ~1004 MiB/s | **~5.5x** |

AVX-512×8 is the real payoff. AVX2×4 is correct but modest — AVX2's
missing 64-bit rotate is emulated as shift-left + shift-right + or,
which eats most of the 4-way parallelism's benefit. Both flags still
default to whatever `PerformanceConfig::default()` sets (`use_avx512:
false`, `use_avx2: true`) — check your own workload's batch size and
CPU before relying on either number.

Single-hash throughput across payload sizes (scalar path), measured via
`cargo run --release --example benchmark`:

| Size | Throughput |
|------|-----------|
| 64 B | ~150-160 MiB/s |
| 1 KiB | ~155-165 MiB/s |
| 64 KiB | ~170-200 MiB/s |
| 1 MiB | ~180-195 MiB/s |

That single-hash number is a correct, competent scalar Rust SHA3-512 —
comparable to other pure-Rust scalar implementations. The real
hardware-accelerated throughput this crate can deliver today is in
`hash_many()`'s AVX-512×8 batching, not single-message `hash()`.

---

## 6. Security Considerations

### 6.1 Constant-Time Operations

The `ct` module provides accumulator-based comparison functions
following the methodology described by Bernstein, Lange and Schwabe
(2012).  All bytes are always visited; no early-exit branches exist.

### 6.2 Memory Safety

The entire crate is written in safe Rust with the sole exception of
`hash_memory_region`, which uses `core::slice::from_raw_parts` to
construct a slice from a raw kernel address.  The caller is responsible
for ensuring the region is valid and readable for the duration of the
call.

### 6.3 Side-Channel Resistance

The Keccak-f[1600] permutation operates exclusively on 64-bit
integers using XOR, AND, NOT, and rotate instructions.  There are no
data-dependent table lookups, conditional branches, or variable-time
multiplications, which mitigates cache-timing and branch-prediction
side channels (Bernstein, 2005).

### 6.4 Zeroisation

Callers requiring post-use state zeroisation should overwrite the
hasher's internal state using `reset()` after finalising.  A future
release may integrate the `zeroize` crate for automatic drop-time
clearing.

---

## 7. Correctness Validation

### 7.1 NIST CAVP Test Vectors

The following NIST CAVP SHA3-512 test vectors are verified in the
crate's test suite:

| Test Case | Input | Reference |
|-----------|-------|-----------|
| `ShortMsg Len=0` | Empty message | NIST CAVP SHA3-512 ShortMsg |
| `ShortMsg Len=8` | `0xCC` | NIST CAVP SHA3-512 ShortMsg |
| `ShortMsg Len=16` | `0x41 0xFB` | NIST CAVP SHA3-512 ShortMsg |
| `LongMsg Len=1600` | `0xA3` x 200 | NIST CAVP SHA3-512 LongMsg |

### 7.2 Keccak-f[1600] Known-Answer Tests

Lanes 0-4 of the permuted zero-state are verified against the XKCP
(eXtended Keccak Code Package) reference values.

### 7.3 Cross-Validation

The implementation is cross-validated against the RustCrypto `sha3`
crate (v0.10) for 20 input lengths: 0, 1, 2, 31, 32, 63, 64, 71, 72
(exact rate), 73 (rate+1), 100, 143, 144, 145, 200, 255, 256, 512,
1000, and 4096 bytes.

### 7.4 Additional Tests

- **Incremental vs. one-shot equivalence** (multi-chunk and byte-at-a-time)
- **Determinism** (repeated hashing of identical input)
- **Avalanche** (single-bit input difference)
- **Boundary conditions** (exact rate, rate+1, 1 MiB)
- **Memory-region hashing** (zero-size, oversized, valid slice)
- **Runtime SIMD detection** (x86_64 tier probing)

---

## 8. Performance

### 8.1 Algorithmic Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| `hash(n)` | O(n) | O(1) — 200-byte state + 200-byte buffer |
| `keccak_f1600` (per block) | O(1) — 24 rounds x 25 lanes | O(1) — 25-word scratch |
| `finalize` | O(1) | O(1) |

### 8.2 Platform Support

| Architecture | Code path actually run | Notes |
|-------------|------------------------|-------|
| x86_64 (`std`) | Scalar by default; real AVX-512 (`hash()`) and AVX-512×8/AVX2×4 (`hash_many()`) available via `PerformanceConfig` | See §5.1 for which flag does what, and which is actually a throughput win |
| x86_64 (`kernel`) | Scalar (always) | `is_x86_feature_detected!` is `std`-only; `simd_avx512`/`simd_parallel` are excluded from `no_std` builds entirely |
| x86 | Scalar (always) | 32-bit |
| AArch64 | Scalar (always) | No NEON/SVE path implemented yet |
| Other | Scalar (always) | Universal fallback |

Every row runs scalar Rust by default. On `std` + `x86_64`, opting into
`PerformanceConfig::use_avx512`/`use_avx2` changes what actually runs —
see §5.1 for exactly which combinations are real speedups versus
correct-but-not-faster.

### 8.3 Benchmarking

Run the criterion benchmarks:

```bash
cargo bench -p sha3-kernel-hasher
```

Or use the standalone example for a quick measurement:

```bash
cargo run --example benchmark --release -p sha3-kernel-hasher
```

---

## 9. Publishing to crates.io

Yes, this crate can be published to [crates.io](https://crates.io).
Follow these steps:

```bash
# 1. Ensure all tests pass
cargo test -p sha3-kernel-hasher

# 2. Dry-run the publish to check for issues
cargo publish -p sha3-kernel-hasher --dry-run

# 3. Log in to crates.io (one-time setup)
cargo login <your-api-token>

# 4. Publish
cargo publish -p sha3-kernel-hasher
```

Before publishing, verify the following in `Cargo.toml`:
- `name` — Must be unique on crates.io
- `version` — Follow SemVer (Semantic Versioning 2.0.0; Preston-Werner, 2013)
- `license` — Dual MIT/Apache-2.0 is standard for the Rust ecosystem
- `repository` — Points to the canonical source repository
- `description` — Concise summary (< 200 characters)
- `readme` — Points to this file

---

## 10. Contributing

Contributions are welcome.  Please see [CONTRIBUTING.md](CONTRIBUTING.md)
for guidelines.

### Development Workflow

```bash
git clone https://github.com/Basty-devel/sha3-kernel-hasher
cd sha3-kernel-hasher
cargo test
cargo bench
cargo clippy -- -D warnings
```

### Responsible Disclosure

For security vulnerabilities, email **sebastian.nestler@tutanota.de**
using the PGP key in [PUBLIC_KEY.asc](PUBLIC_KEY.asc).

**Fingerprint**: `E799967FCCC36C6986AB39423B52B58B17A9F2E5`

---

## 11. Licence

Dual-licensed under:

- **MIT** — [LICENSE-MIT](LICENSE-MIT)
- **Apache 2.0** — [LICENSE-APACHE](LICENSE-APACHE)

at your option.

---

## 12. References

Bernstein, D.J. (2005) *Cache-timing attacks on AES*. Available at:
https://cr.yp.to/antiforgery/cachetiming-20050414.pdf (Accessed: 8
February 2026).

Bernstein, D.J., Lange, T. and Schwabe, P. (2012) 'The Security
Impact of a New Cryptographic Library', in *Progress in Cryptology —
LATINCRYPT 2012*. Lecture Notes in Computer Science, vol. 7533. Berlin:
Springer, pp. 159-176. doi:10.1007/978-3-642-33481-8_9.

Bertoni, G., Daemen, J., Peeters, M. and Van Assche, G. (2011) *The
Keccak Reference*. Version 3.0. Available at:
https://keccak.team/files/Keccak-reference-3.0.pdf (Accessed: 8
February 2026).

Kelsey, J. and Schneier, B. (2005) 'Second Preimages on n-Bit Hash
Functions for Much Less than 2^n Work', in *Advances in Cryptology —
EUROCRYPT 2005*. Lecture Notes in Computer Science, vol. 3494. Berlin:
Springer, pp. 474-490. doi:10.1007/11426639_28.

Kocher, P.C. (1996) 'Timing Attacks on Implementations of
Diffie-Hellman, RSA, DSS, and Other Systems', in *Advances in
Cryptology — CRYPTO '96*. Lecture Notes in Computer Science, vol. 1109.
Berlin: Springer, pp. 104-113. doi:10.1007/3-540-68697-5_9.

Microsoft (2023) *KeSaveExtendedProcessorState function*. Microsoft
Learn. Available at:
https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/nf-wdm-kesaveextendedprocessorstate
(Accessed: 8 February 2026).

National Institute of Standards and Technology (2015) *SHA-3 Standard:
Permutation-Based Hash and Extendable-Output Functions*. FIPS PUB 202.
Gaithersburg, MD: NIST. doi:10.6028/NIST.FIPS.202.

Preston-Werner, T. (2013) *Semantic Versioning 2.0.0*. Available at:
https://semver.org/spec/v2.0.0.html (Accessed: 8 February 2026).

---

*Developed as part of the Bunker security platform by Sebastian
Friedrich Nestler.*
