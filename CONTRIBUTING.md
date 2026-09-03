# Contributing to sha3-kernel-hasher

Contributions welcome. This is a pure-Rust SHA3-512 (FIPS 202) implementation
with a `no_std` + `alloc` build mode for kernel drivers — see the README for
what's actually implemented today (scalar only — no SIMD acceleration yet)
before proposing a change.

## Getting started

### Prerequisites
- Rust 1.73+, stable toolchain
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

- **A real SIMD-accelerated Keccak-f[1600] permutation.** This is the
  single most valuable contribution right now — `PerformanceConfig`'s
  `use_avx2`/`use_avx512` fields exist specifically so a real
  implementation can land without an API break. See README §5.1 for
  exactly what's missing and how to verify a change actually accelerates
  anything (the repo's own criterion benchmark is the test: a real
  SIMD path should show a measurable throughput difference between the
  `scalar` and `avx2`/`avx512` benchmark groups — today they're
  statistically identical, which is the bug this would fix).
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
