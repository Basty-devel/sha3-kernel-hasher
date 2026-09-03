# SHA3-Kernel-Hasher 0.3.0

A pure-Rust SHA3-512 (FIPS 202) implementation, dual-mode: standard `std`
for userspace, `no_std` + `alloc` for Windows kernel-mode drivers.

## What's actually in this release

- **Full Keccak-f[1600] permutation** — all 24 rounds (θ, ρ, π, χ, ι),
  correct SHA3 domain separation byte (`0x06`) and `pad10*1` padding
  (FIPS 202 §6.1).
- **Passes NIST CAVP's published SHA3-512 test vectors** — ShortMsg
  (empty, `0xCC`, `0x41FB`) and LongMsg (`0xA3` × 200). This means the
  output matches NIST's known-answer vectors; it is *not* a formal CAVP
  or CMVP certificate — no accredited lab has evaluated this
  implementation, and none is claimed.
- **Cross-validated against RustCrypto's `sha3` crate** (v0.10) across 20
  input lengths.
- **New in 0.3.0: `hash_many()`** — batch hashing of independent
  messages with a real AVX-512×8 SIMD path (~5.5x over a scalar loop;
  see below). Handles variable-length messages within one batch.
- **65 passing tests** (57 unit + 8 doc), including
  constant-time-comparison correctness, incremental-vs-one-shot
  equivalence, boundary conditions (empty input, exact-rate-block,
  rate+1), and bit-for-bit cross-validation of every SIMD path against
  the scalar implementation.
- **Constant-time digest comparison** (`ct::ct_eq`) — accumulator-based
  XOR over the full length, no early-exit branch, following Bernstein,
  Lange & Schwabe (2012).
- **`no_std` + `alloc` kernel-mode build** for Windows kernel drivers
  (WDM/KMDF), plus processor-state save/restore helpers
  (`kernel_safe::save_processor_state`/`restore_processor_state`) for
  code that needs to touch SIMD registers safely in a kernel context.
- **MIT OR Apache-2.0**, dual-licensed.
- **PGP-signed releases** — public key included (`PUBLIC_KEY.asc`),
  signature accompanies each release artifact.

## What this release is not

Being direct about this because the crate's own earlier docs weren't:

- **Single-message SIMD (`hash()`) is real but not a speedup.** The
  crate detects AVX-512F/BW/DQ, AVX2+BMI2, and SSE4.2 at runtime
  (`is_x86_feature_detected!`, exposed via `simd_features()`), and
  `PerformanceConfig::use_avx512` genuinely dispatches to a real,
  cross-validated AVX-512 Keccak-f[1600] permutation — this is not a
  stub. It measures ~3-4% *slower* than scalar at 1 MiB, though, because
  a single hash has no independent lanes for a wide register to exploit.
  It defaults to `false` for exactly that reason. See README §5.1 for
  the numbers, or `cargo bench --bench sha3_bench sha3_512_throughput`
  to reproduce on your own hardware.
- **Batch SIMD (`hash_many()`) is a real, substantial speedup.** Hashing
  a batch of independent messages via `PerformanceConfig::use_avx512`
  dispatches to an AVX-512×8 batched permutation measuring ~5.5x over a
  scalar loop (64 independent 4 KiB messages, this container's Xeon);
  `use_avx2` gives a more modest ~1.2x via AVX2×4 (AVX2 has no 64-bit
  rotate instruction). See README §5.1 or
  `cargo bench --bench sha3_bench sha3_512_hash_many` to reproduce.
- **No external security audit has been performed.** No cryptography
  firm, no peer review, no CVE history. Treat it as unaudited until
  stated otherwise.
- **No formal compliance certification.** Not FIPS 140 validated, not
  Common Criteria evaluated, no CNSA 2.0 endorsement. "Passes NIST's
  published test vectors" and "government-certified" are different
  claims; only the first is true here.
- **Not production-proven.** This is a new crate. It hasn't shipped
  inside a deployed product yet.

## Links

- Crates.io: https://crates.io/crates/sha3-kernel-hasher
- Documentation: https://docs.rs/sha3-kernel-hasher
- GitHub: https://github.com/Basty-devel/sha3-kernel-hasher
- Contact: sebastian.nestler@tutanota.de

## Verification

```bash
curl -O https://crates.io/api/v1/crates/sha3-kernel-hasher/0.3.0/download
curl -O https://github.com/Basty-devel/sha3-kernel-hasher/releases/download/v0.3.0/sha3-kernel-hasher-0.3.0.tar.gz.asc
gpg --import PUBLIC_KEY.asc
gpg --verify sha3-kernel-hasher-0.3.0.tar.gz.asc download
```

## What's next

Closing the single-message AVX-512 gap (currently a small net loss, not
a win) and a faster AVX2 rotate for batch hashing (currently only ~1.2x,
versus AVX-512×8's ~5.5x) are the open performance items — see
CONTRIBUTING.md. A NEON (AArch64) path doesn't exist at all yet. None of
that changes the core correctness story: this is a correct, tested,
dual-mode SHA3-512 with a genuinely accelerated batch-hashing path where
it matters most.
