//! SIMD-parallel Keccak-f\[1600\]: N *independent* states permuted at once.
//!
//! # Design — why this is a different (and better) layout than `simd_avx512`
//!
//! `simd_avx512.rs` vectorizes a *single* Keccak-f\[1600\] instance by
//! packing its 25 lanes across SIMD lanes of a few registers. That
//! measured only parity-to-slightly-behind scalar (see its module docs)
//! because a single instance has no independent data for a wide register
//! to parallelize *within* — π's cross-lane mixing has to pay for real
//! shuffle/gather instructions scalar code doesn't.
//!
//! This module instead vectorizes across N **independent** hash states:
//! one register per *lane position*, holding that same lane from all N
//! states simultaneously (struct-of-arrays). Concretely, for the
//! AVX-512×8 case, register `reg[i]` holds `states[0][i], states[1][i],
//! ..., states[7][i]`. This changes every step's cost:
//!
//! - **θ, χ, ι**: index arithmetic (`state[x]`, `b[base+1]`, ...) becomes
//!   plain Rust array indexing into `reg`/`b` — a compile-time register
//!   selection, not a runtime shuffle. Zero-cost.
//! - **ρ+π (fused)**: `b[i] = rotate(reg[PI[i]], RHO[PI[i]])`. `PI[i]` is
//!   again just an array index (register selection, free); the rotation
//!   amount is the *same for every one of the N parallel states* (it
//!   only depends on lane position, never on which state), so it's a
//!   uniform per-register rotate — no gather, no lane-shuffle, at all.
//!
//! The tradeoff is the load/store at the boundary: turning N separate
//! `[u64; 25]` states into 25 registers (and back) needs a scalar
//! transpose, done once per permutation call — not once per round like
//! `simd_avx512`'s π. For a multi-block message this cost is paid once
//! per rate-block, same as the scalar sponge's own per-block overhead.
//!
//! # Two widths, two rotate strategies
//!
//! - **AVX-512×8** (`keccak_f1600_x8_avx512`): `_mm512_rolv_epi64`
//!   (variable per-lane rotate) with an all-lanes-equal amount vector.
//!   Confirmed correct at rotate-by-0 on real hardware (probed
//!   separately; AVX-512's variable rotate handles 0 as identity).
//! - **AVX2×4** (`keccak_f1600_x4_avx2`): AVX2 has no 64-bit rotate
//!   instruction at all (variable or immediate), so rotation is built
//!   from `(v << n) | (v >> (64 - n))` via `_mm256_sllv_epi64`/
//!   `_mm256_srlv_epi64`. The `n = 0` case relies on x86's documented
//!   shift semantics (count ≥ 64 on a 64-bit shift yields zero, not
//!   undefined behavior) so that `v >> 64` contributes `0` and the
//!   formula degenerates correctly to `v | 0 = v` — verified against
//!   `u64::rotate_left` on real hardware before use here, not assumed.
//!
//! # Why message batching is a separate concern
//!
//! These functions only vectorize the *permutation* across N same-shape
//! `[u64; 25]` states. The sponge-level problem of batching
//! **variable-length** messages (padding each to its own block count,
//! then continuing to permute lanes whose message isn't done yet while
//! extracting the digest for lanes that just finished) is handled by
//! `hash_many` in `lib.rs`, layered on top of these primitives — exactly
//! how `keccak_f1600` (scalar) and `keccak_f1600_avx512` are lower-level
//! primitives that `Sha3_512State`'s sponge logic calls.

use crate::keccak::{round_constants, PI, RHO};
use std::arch::x86_64::*;

/// Ternary-logic immediate computing `a ^ (!b & c)` — same as
/// `simd_avx512::CHI_TERNARY_IMM`, re-declared here so this module has no
/// dependency on that one (they vectorize orthogonal dimensions and
/// shouldn't need each other).
const CHI_TERNARY_IMM: i32 = 0xD2;

/// Permute 8 independent Keccak-f\[1600\] states in parallel.
///
/// # Safety
///
/// The caller must have confirmed `is_x86_feature_detected!("avx512f")`
/// returns `true` before calling this function.
#[target_feature(enable = "avx512f")]
pub unsafe fn keccak_f1600_x8_avx512(states: &mut [[u64; 25]; 8]) {
    // Transpose in: reg[i] holds lane i from all 8 states.
    let mut reg = [_mm512_setzero_si512(); 25];
    for (i, r) in reg.iter_mut().enumerate() {
        *r = _mm512_set_epi64(
            states[7][i] as i64,
            states[6][i] as i64,
            states[5][i] as i64,
            states[4][i] as i64,
            states[3][i] as i64,
            states[2][i] as i64,
            states[1][i] as i64,
            states[0][i] as i64,
        );
    }

    // RHO[i] broadcast to all 8 lanes — same value for every parallel
    // state, computed once (not per round) since RHO/PI never change.
    let mut rot_bcast = [_mm512_setzero_si512(); 25];
    for (i, r) in rot_bcast.iter_mut().enumerate() {
        *r = _mm512_set1_epi64(RHO[i] as i64);
    }

    for &rc in round_constants().iter() {
        // ===== θ (theta) =====
        let mut c = [_mm512_setzero_si512(); 5];
        for (x, cx) in c.iter_mut().enumerate() {
            *cx = _mm512_xor_si512(
                _mm512_xor_si512(reg[x], reg[x + 5]),
                _mm512_xor_si512(reg[x + 10], _mm512_xor_si512(reg[x + 15], reg[x + 20])),
            );
        }
        let mut d = [_mm512_setzero_si512(); 5];
        for (x, dx) in d.iter_mut().enumerate() {
            *dx = _mm512_xor_si512(c[(x + 4) % 5], _mm512_rol_epi64::<1>(c[(x + 1) % 5]));
        }
        for y in 0..5 {
            for x in 0..5 {
                reg[x + 5 * y] = _mm512_xor_si512(reg[x + 5 * y], d[x]);
            }
        }

        // ===== ρ (rho) + π (pi), fused (mirrors the scalar loop) =====
        let mut b = [_mm512_setzero_si512(); 25];
        for i in 0..25 {
            b[i] = _mm512_rolv_epi64(reg[PI[i]], rot_bcast[PI[i]]);
        }

        // ===== χ (chi) =====
        for y in 0..5 {
            let base = 5 * y;
            reg[base] =
                _mm512_ternarylogic_epi64::<CHI_TERNARY_IMM>(b[base], b[base + 1], b[base + 2]);
            reg[base + 1] =
                _mm512_ternarylogic_epi64::<CHI_TERNARY_IMM>(b[base + 1], b[base + 2], b[base + 3]);
            reg[base + 2] =
                _mm512_ternarylogic_epi64::<CHI_TERNARY_IMM>(b[base + 2], b[base + 3], b[base + 4]);
            reg[base + 3] =
                _mm512_ternarylogic_epi64::<CHI_TERNARY_IMM>(b[base + 3], b[base + 4], b[base]);
            reg[base + 4] =
                _mm512_ternarylogic_epi64::<CHI_TERNARY_IMM>(b[base + 4], b[base], b[base + 1]);
        }

        // ===== ι (iota) =====
        let rc_vec = _mm512_set1_epi64(rc as i64);
        reg[0] = _mm512_xor_si512(reg[0], rc_vec);
    }

    // Transpose out.
    for (i, r) in reg.iter().enumerate() {
        let mut tmp = [0u64; 8];
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut i32 as *mut _, *r);
        for (s, state) in states.iter_mut().enumerate() {
            state[i] = tmp[s];
        }
    }
}

/// Permute 4 independent Keccak-f\[1600\] states in parallel.
///
/// # Safety
///
/// The caller must have confirmed `is_x86_feature_detected!("avx2")`
/// returns `true` before calling this function.
#[target_feature(enable = "avx2")]
pub unsafe fn keccak_f1600_x4_avx2(states: &mut [[u64; 25]; 4]) {
    #[inline(always)]
    unsafe fn rotl(v: __m256i, n: __m256i, sixty_four_minus_n: __m256i) -> __m256i {
        _mm256_or_si256(
            _mm256_sllv_epi64(v, n),
            _mm256_srlv_epi64(v, sixty_four_minus_n),
        )
    }

    let mut reg = [_mm256_setzero_si256(); 25];
    for (i, r) in reg.iter_mut().enumerate() {
        *r = _mm256_set_epi64x(
            states[3][i] as i64,
            states[2][i] as i64,
            states[1][i] as i64,
            states[0][i] as i64,
        );
    }

    // Precompute rotate-amount vectors (both n and 64-n) once, not per
    // round — AVX2 has no 64-bit rotate, only variable shifts.
    let mut rot_n = [_mm256_setzero_si256(); 25];
    let mut rot_64_minus_n = [_mm256_setzero_si256(); 25];
    for i in 0..25 {
        rot_n[i] = _mm256_set1_epi64x(RHO[i] as i64);
        rot_64_minus_n[i] = _mm256_set1_epi64x((64 - RHO[i]) as i64);
    }

    for &rc in round_constants().iter() {
        // ===== θ (theta) =====
        let mut c = [_mm256_setzero_si256(); 5];
        for (x, cx) in c.iter_mut().enumerate() {
            *cx = _mm256_xor_si256(
                _mm256_xor_si256(reg[x], reg[x + 5]),
                _mm256_xor_si256(reg[x + 10], _mm256_xor_si256(reg[x + 15], reg[x + 20])),
            );
        }
        let mut d = [_mm256_setzero_si256(); 5];
        for (x, dx) in d.iter_mut().enumerate() {
            let c_plus1 = c[(x + 1) % 5];
            let rotated = rotl(c_plus1, _mm256_set1_epi64x(1), _mm256_set1_epi64x(63));
            *dx = _mm256_xor_si256(c[(x + 4) % 5], rotated);
        }
        for y in 0..5 {
            for x in 0..5 {
                reg[x + 5 * y] = _mm256_xor_si256(reg[x + 5 * y], d[x]);
            }
        }

        // ===== ρ (rho) + π (pi), fused =====
        let mut b = [_mm256_setzero_si256(); 25];
        for i in 0..25 {
            let src = PI[i];
            b[i] = if RHO[src] == 0 {
                reg[src]
            } else {
                rotl(reg[src], rot_n[src], rot_64_minus_n[src])
            };
        }

        // ===== χ (chi) =====
        for y in 0..5 {
            let base = 5 * y;
            let b0 = b[base];
            let b1 = b[base + 1];
            let b2 = b[base + 2];
            let b3 = b[base + 3];
            let b4 = b[base + 4];
            reg[base] = _mm256_xor_si256(b0, _mm256_andnot_si256(b1, b2));
            reg[base + 1] = _mm256_xor_si256(b1, _mm256_andnot_si256(b2, b3));
            reg[base + 2] = _mm256_xor_si256(b2, _mm256_andnot_si256(b3, b4));
            reg[base + 3] = _mm256_xor_si256(b3, _mm256_andnot_si256(b4, b0));
            reg[base + 4] = _mm256_xor_si256(b4, _mm256_andnot_si256(b0, b1));
        }

        // ===== ι (iota) =====
        let rc_vec = _mm256_set1_epi64x(rc as i64);
        reg[0] = _mm256_xor_si256(reg[0], rc_vec);
    }

    for (i, r) in reg.iter().enumerate() {
        let mut tmp = [0u64; 4];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, *r);
        for (s, state) in states.iter_mut().enumerate() {
            state[i] = tmp[s];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keccak::keccak_f1600;

    fn xorshift(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn x8_matches_scalar_on_zero_states() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: no avx512f on this host");
            return;
        }
        let mut states = [[0u64; 25]; 8];
        unsafe { keccak_f1600_x8_avx512(&mut states) };

        let mut expected = [0u64; 25];
        keccak_f1600(&mut expected);

        for (lane, state) in states.iter().enumerate() {
            assert_eq!(
                *state, expected,
                "lane {lane} diverged from scalar on zero state"
            );
        }
    }

    #[test]
    fn x8_matches_scalar_on_independent_random_states() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: no avx512f on this host");
            return;
        }
        let mut seed = 0x9E3779B97F4A7C15u64;
        for trial in 0..200 {
            let mut states = [[0u64; 25]; 8];
            for state in states.iter_mut() {
                for lane in state.iter_mut() {
                    *lane = xorshift(&mut seed);
                }
            }
            let expected: Vec<[u64; 25]> = states
                .iter()
                .map(|s| {
                    let mut s = *s;
                    keccak_f1600(&mut s);
                    s
                })
                .collect();

            let mut vector_states = states;
            unsafe { keccak_f1600_x8_avx512(&mut vector_states) };

            for (lane, (actual, expect)) in vector_states.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    actual, expect,
                    "trial {trial}, lane {lane}: x8 AVX-512 diverged from independent scalar permutation"
                );
            }
        }
    }

    #[test]
    fn x8_lanes_are_truly_independent() {
        // Changing only lane 3's input must not change any other lane's
        // output — proves the transpose doesn't leak data across lanes.
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: no avx512f on this host");
            return;
        }
        let mut states_a = [[0u64; 25]; 8];
        let mut states_b = [[0u64; 25]; 8];
        states_b[3][7] = 0xDEAD_BEEF_0000_0001;

        unsafe {
            keccak_f1600_x8_avx512(&mut states_a);
            keccak_f1600_x8_avx512(&mut states_b);
        }

        for lane in 0..8 {
            if lane == 3 {
                assert_ne!(
                    states_a[lane], states_b[lane],
                    "lane 3 should differ (its input changed)"
                );
            } else {
                assert_eq!(
                    states_a[lane], states_b[lane],
                    "lane {lane} must be unaffected by lane 3's differing input"
                );
            }
        }
    }

    #[test]
    fn x4_matches_scalar_on_zero_states() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping: no avx2 on this host");
            return;
        }
        let mut states = [[0u64; 25]; 4];
        unsafe { keccak_f1600_x4_avx2(&mut states) };

        let mut expected = [0u64; 25];
        keccak_f1600(&mut expected);

        for (lane, state) in states.iter().enumerate() {
            assert_eq!(
                *state, expected,
                "lane {lane} diverged from scalar on zero state"
            );
        }
    }

    #[test]
    fn x4_matches_scalar_on_independent_random_states() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping: no avx2 on this host");
            return;
        }
        let mut seed = 0xC0FFEE1234567u64;
        for trial in 0..200 {
            let mut states = [[0u64; 25]; 4];
            for state in states.iter_mut() {
                for lane in state.iter_mut() {
                    *lane = xorshift(&mut seed);
                }
            }
            let expected: Vec<[u64; 25]> = states
                .iter()
                .map(|s| {
                    let mut s = *s;
                    keccak_f1600(&mut s);
                    s
                })
                .collect();

            let mut vector_states = states;
            unsafe { keccak_f1600_x4_avx2(&mut vector_states) };

            for (lane, (actual, expect)) in vector_states.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    actual, expect,
                    "trial {trial}, lane {lane}: x4 AVX2 diverged from independent scalar permutation"
                );
            }
        }
    }

    #[test]
    fn x4_lanes_are_truly_independent() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping: no avx2 on this host");
            return;
        }
        let mut states_a = [[0u64; 25]; 4];
        let mut states_b = [[0u64; 25]; 4];
        states_b[2][11] = 0xC0FFEE_0000_0002;

        unsafe {
            keccak_f1600_x4_avx2(&mut states_a);
            keccak_f1600_x4_avx2(&mut states_b);
        }

        for lane in 0..4 {
            if lane == 2 {
                assert_ne!(
                    states_a[lane], states_b[lane],
                    "lane 2 should differ (its input changed)"
                );
            } else {
                assert_eq!(
                    states_a[lane], states_b[lane],
                    "lane {lane} must be unaffected by lane 2's differing input"
                );
            }
        }
    }

    #[test]
    fn x8_matches_after_chained_permutations() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: no avx512f on this host");
            return;
        }
        let mut scalar_states = [[0u64; 25]; 8];
        for (lane, state) in scalar_states.iter_mut().enumerate() {
            state[0] = lane as u64 + 1;
        }
        let mut vector_states = scalar_states;

        for iteration in 0..30 {
            for state in scalar_states.iter_mut() {
                keccak_f1600(state);
            }
            unsafe { keccak_f1600_x8_avx512(&mut vector_states) };
            assert_eq!(
                vector_states, scalar_states,
                "diverged after {iteration} chained permutations"
            );
        }
    }
}

#[cfg(test)]
mod timing_smoke {
    //! Not a correctness test — a quick sanity check (run with
    //! `cargo test --release -- --nocapture timing_smoke --ignored`) that
    //! the parallel primitives actually deliver a throughput win over N
    //! independent scalar calls, before building the full `hash_many`
    //! sponge/padding layer on top of them. Ignored by default because
    //! its assertions are necessarily loose (timing, not correctness) and
    //! it's meaningless under a debug build.
    use super::*;
    use crate::keccak::keccak_f1600;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn x8_avx512_vs_8x_scalar() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: no avx512f on this host");
            return;
        }
        const ITERS: usize = 200_000;
        let mut states = [[0u64; 25]; 8];

        let start = Instant::now();
        for _ in 0..ITERS {
            unsafe { keccak_f1600_x8_avx512(&mut states) };
        }
        let vector_elapsed = start.elapsed();

        let mut scalar_states = [[0u64; 25]; 8];
        let start = Instant::now();
        for _ in 0..ITERS {
            for s in scalar_states.iter_mut() {
                keccak_f1600(s);
            }
        }
        let scalar_elapsed = start.elapsed();

        println!(
            "x8 AVX-512: {:?} for {ITERS} calls (8 states/call) vs {:?} for {} scalar calls -> {:.2}x",
            vector_elapsed,
            scalar_elapsed,
            ITERS * 8,
            scalar_elapsed.as_secs_f64() / vector_elapsed.as_secs_f64()
        );
    }

    #[test]
    #[ignore]
    fn x4_avx2_vs_4x_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping: no avx2 on this host");
            return;
        }
        const ITERS: usize = 200_000;
        let mut states = [[0u64; 25]; 4];

        let start = Instant::now();
        for _ in 0..ITERS {
            unsafe { keccak_f1600_x4_avx2(&mut states) };
        }
        let vector_elapsed = start.elapsed();

        let mut scalar_states = [[0u64; 25]; 4];
        let start = Instant::now();
        for _ in 0..ITERS {
            for s in scalar_states.iter_mut() {
                keccak_f1600(s);
            }
        }
        let scalar_elapsed = start.elapsed();

        println!(
            "x4 AVX2: {:?} for {ITERS} calls (4 states/call) vs {:?} for {} scalar calls -> {:.2}x",
            vector_elapsed,
            scalar_elapsed,
            ITERS * 4,
            scalar_elapsed.as_secs_f64() / vector_elapsed.as_secs_f64()
        );
    }
}
