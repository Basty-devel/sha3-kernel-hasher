# [RELEASE] sha3-kernel-hasher 0.3.0 — pure-Rust SHA3-512, no_std/kernel-mode dual build

Releasing a SHA3-512 (FIPS 202) implementation aimed at a gap I kept
running into: most Rust SHA3 crates assume userspace `std`, and I needed
one that also builds `no_std` + `alloc` for a Windows kernel-mode driver.

## What it does

- Full Keccak-f[1600] permutation, correct FIPS 202 padding/domain
  separation, safe Rust throughout (one `unsafe` block total, for a
  `hash_memory_region` helper that reads a raw kernel address — caller's
  responsibility to guarantee validity, documented as such).
- Passes NIST CAVP's published SHA3-512 test vectors, cross-validated
  against RustCrypto's `sha3` crate across 20 input lengths, 36 tests
  total.
- Constant-time digest comparison (`ct::ct_eq`) — no early-exit branch,
  so verifying a hash doesn't leak which byte differed first.
- `std` for userspace, `kernel` (`no_std` + `alloc`) for Windows
  kernel-mode drivers, plus processor-state save/restore helpers for
  touching SIMD registers safely in a kernel context.
- MIT OR Apache-2.0, PGP-signed releases (public key in the repo).

## What SIMD actually does here

Correcting my own earlier draft of this post here: I'd written up
AVX-512/AVX2 throughput numbers and a "military-grade" pitch before
actually confirming what the code does. Here's what's real, now that
there's an actual SIMD implementation to measure instead of a stub.

The crate detects AVX-512F/BW/DQ, AVX2+BMI2, and SSE4.2 at runtime and
`PerformanceConfig`'s `use_avx512`/`use_avx2` flags are genuinely wired
up — but they answer different questions depending on the call:

**`hash()` (single message):** `use_avx512` dispatches to a real,
cross-validated AVX-512 permutation — not a stub — but it's *not*
faster:

```
sha3_512_throughput/1MB_scalar   [180.66 MiB/s 181.96 MiB/s 183.33 MiB/s]
sha3_512_throughput/1MB_auto     [174.25 MiB/s 175.71 MiB/s 177.20 MiB/s]  (~3-4% slower)
```

A single Keccak-f[1600] instance has no independent lanes for a 512-bit
register to parallelize within, so it defaults to `false`.

**`hash_many()` (a batch of independent messages):** here SIMD is a real
win, because independent messages give the vector width genuine
parallel work:

```
sha3_512_hash_many/64x4KiB_scalar_loop   [178.47 MiB/s 182.12 MiB/s 186.06 MiB/s]
sha3_512_hash_many/64x4KiB_avx2          [217.39 MiB/s 219.85 MiB/s 222.88 MiB/s]  (~1.2x)
sha3_512_hash_many/64x4KiB_avx512        [997.09 MiB/s 1004.4 MiB/s 1011.2 MiB/s]  (~5.5x)
```

No 2.5 GB/s, no OpenSSL comparison — I don't have a benchmark against
OpenSSL in this repo, so I'm not going to claim one. ~1 GB/s on batched
AVX-512 is the real, reproducible number.

Also correcting: there's no completed external security audit and no
formal FIPS 140/CMVP/Common Criteria certification. "Passes NIST's
published test vectors" is true and checkable; "government-certified"
would be a different, false claim.

## Usage

```toml
[dependencies]
sha3-kernel-hasher = "0.3.0"
```

```rust
use sha3_kernel_hasher::Sha3_512Kernel;

let mut hasher = Sha3_512Kernel::new();
let hash = hasher.hash(b"hello");
println!("SHA3-512: {:x}", hash);
```

Kernel-mode build:

```toml
sha3-kernel-hasher = { version = "0.3.0", default-features = false, features = ["kernel"] }
```

## What's next

Closing the small single-message AVX-512 gap, and a faster AVX2 rotate
for batch hashing (AVX2 has no 64-bit rotate instruction, so it's
currently emulated — see CONTRIBUTING.md) are the open performance
items. A NEON (AArch64) path doesn't exist yet either.

## Links

- Crates.io: https://crates.io/crates/sha3-kernel-hasher
- Docs: https://docs.rs/sha3-kernel-hasher
- GitHub: https://github.com/Basty-devel/sha3-kernel-hasher

#rust #cryptography #sha3 #kernel #no-std
