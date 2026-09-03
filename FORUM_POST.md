# [RELEASE] sha3-kernel-hasher 0.2.0 — pure-Rust SHA3-512, no_std/kernel-mode dual build

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

## What it doesn't do (yet)

Correcting my own earlier draft of this post here: I'd written up
AVX-512/AVX2 throughput numbers and a "military-grade" pitch before
actually confirming what the code does. It doesn't hold up, so here's
what's real instead.

The crate detects AVX-512F/BW/DQ, AVX2+BMI2, and SSE4.2 at runtime and
has a `PerformanceConfig` with `use_avx512`/`use_avx2` flags — but
`hash()` doesn't read either. Every call runs the same scalar
Keccak-f[1600] path. I benchmarked it properly instead of guessing:

```
sha3_512_throughput/1MB_auto     [157.34 MiB/s 159.28 MiB/s 161.33 MiB/s]
sha3_512_throughput/1MB_scalar   [163.80 MiB/s 167.20 MiB/s 170.87 MiB/s]
```

(`auto` = SIMD-detected config, `scalar` = both flags forced `false` —
same code, same speed, on a CPU that reports both AVX-512 and AVX2
available.) That's ~150-190 MiB/s across payload sizes from 64 B to
1 MiB — a solid, correct scalar implementation, not an accelerated one.
No 2.5 GB/s, no OpenSSL comparison — I don't have a benchmark against
OpenSSL in this repo, so I'm not going to claim one.

Also correcting: there's no completed external security audit and no
formal FIPS 140/CMVP/Common Criteria certification. "Passes NIST's
published test vectors" is true and checkable; "government-certified"
would be a different, false claim.

## Usage

```toml
[dependencies]
sha3-kernel-hasher = "0.2.0"
```

```rust
use sha3_kernel_hasher::Sha3_512Kernel;

let mut hasher = Sha3_512Kernel::new();
let hash = hasher.hash(b"hello");
println!("SHA3-512: {:x}", hash);
```

Kernel-mode build:

```toml
sha3-kernel-hasher = { version = "0.2.0", default-features = false, features = ["kernel"] }
```

## What's next

A real SIMD-widened permutation is the open item — that's the honest
reason the `PerformanceConfig` fields exist at all, reserved for when
that lands. Until then this is a tested, correct, dual-mode scalar
SHA3-512, which is what I'm actually shipping today.

## Links

- Crates.io: https://crates.io/crates/sha3-kernel-hasher
- Docs: https://docs.rs/sha3-kernel-hasher
- GitHub: https://github.com/Basty-devel/sha3-kernel-hasher

#rust #cryptography #sha3 #kernel #no-std
