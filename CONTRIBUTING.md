# Contributing to sha3-kernel-hasher

Contributions welcome. This is a pure-Rust SHA3-512 (FIPS 202) implementation
with a `no_std` + `alloc` build mode for kernel drivers, and real (not
stubbed) AVX-512/AVX2 SIMD paths on `std` + `x86_64` — see README §5.1 for
exactly which SIMD flag does what before proposing a change, since not
all of them are throughput wins (single-message AVX-512 isn't; batch
AVX-512×8 is a ~5.5x one).

## Getting started

### Prerequisites
- Rust 1.89+, stable toolchain (raised from 1.73 for the AVX-512 SIMD
  intrinsics in `src/simd_avx512.rs` — see README §4.2)
- Familiarity with the FIPS 202 / Keccak sponge construction helps for
  anything touching `src/keccak.rs` or `src/lib.rs`'s absorb/squeeze code
- Kernel-mode programming background if you're working on the `kernel`
  feature or `kernel_safe` module

### Development setup
```bash
git clone https://github.com/Basty-devel/sha3-kernel-hasher
cd sha3-kernel-hasher
cargo test
cargo bench
```

## What's useful to contribute

- **Closing the single-message AVX-512 gap.** `hash()`'s AVX-512 path
  (`src/simd_avx512.rs`) is correct but currently ~3-4% *slower* than
  scalar at 1 MiB (README §5.1) — π's cross-row mixing pays for real
  shuffle/blend instructions a single Keccak-f[1600] instance has no
  independent lanes to amortize against. An implementation that closes
  or reverses that gap (without breaking any of the module's existing
  cross-validation tests) would be genuinely valuable. The repo's own
  criterion benchmark is the test: run
  `cargo bench --bench sha3_bench sha3_512_throughput` before and after
  and show the real numbers in the PR.
- **A faster AVX2 rotate for `hash_many`'s batching.** AVX2×4 batching
  (`src/simd_parallel.rs`) measures only ~1.2x over a scalar loop versus
  AVX-512×8's ~5.5x (README §5.1), because AVX2 has no 64-bit rotate
  instruction — ρ is emulated as shift-left + shift-right + or. If
  there's a cheaper way to get a per-lane rotate out of AVX2 (or a
  restructuring that reduces the instruction count elsewhere to
  compensate), that's a real, benchmarkable win.
- **A NEON (AArch64) path**, for either `hash()` or `hash_many` — there
  is no non-x86 SIMD path at all today (README §8.2).
- Additional test vectors, edge cases, or fuzzing coverage.
- `no_std`/kernel-mode correctness fixes and cross-platform testing.
- Documentation and example fixes — including catching claims that
  don't match what the code does, which is exactly the kind of issue
  this crate's docs had before its first release.

## Code review

There's no formal, funded third-party security audit in place — this is a
new, unaudited crate, and this document won't claim otherwise or name a
specific audit firm as though an engagement exists. What actually happens
today:

- Code review on every PR, focused on correctness and constant-time
  properties for anything touching digest comparison or the sponge state.
- `cargo test`, `cargo clippy -- -D warnings`, and `cargo fmt --check` are
  expected to pass.
- Any performance claim in a PR description must be backed by a
  reproducible `cargo bench` or `cargo run --release --example benchmark`
  result in the PR itself — not a number without a way to check it.

If a funded third-party audit happens in the future, it'll be announced
with the real firm's name and a link to the actual report — not claimed in
advance.

## Development workflow

```bash
git clone https://github.com/Basty-devel/sha3-kernel-hasher
cd sha3-kernel-hasher
git checkout -b feature/your-feature-name

# before opening a PR
cargo test
cargo clippy --all-features -- -D warnings
cargo fmt --check
cargo bench   # if your change touches a hot path — include the before/after numbers in the PR

git push origin feature/your-feature-name
# open a PR on GitHub
```

## Reporting security issues

Email sebastian.nestler@tutanota.de. PGP key: [`PUBLIC_KEY.asc`](PUBLIC_KEY.asc).
Please allow a reasonable window for a fix before public disclosure.

## License

By contributing, you agree your contributions are licensed under the same
terms as the project (MIT OR Apache-2.0).

## Resources

- API docs: [docs.rs/sha3-kernel-hasher](https://docs.rs/sha3-kernel-hasher)
- Examples in this repo: [`examples/basic.rs`](examples/basic.rs),
  [`examples/kernel.rs`](examples/kernel.rs),
  [`examples/benchmark.rs`](examples/benchmark.rs),
  [`examples/large_file_integrity.rs`](examples/large_file_integrity.rs)
- Benchmarks: [`benches/sha3_bench.rs`](benches/sha3_bench.rs),
  [`scripts/benchmark_suite.ps1`](scripts/benchmark_suite.ps1) and
  [`scripts/benchmark_vs_windows.ps1`](scripts/benchmark_vs_windows.ps1)
  (Windows-only, compares against `Get-FileHash -Algorithm SHA512`)
