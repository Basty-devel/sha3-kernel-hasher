//! Keccak-f[1600] Permutation — NIST FIPS 202 Reference Implementation
//!
//! Implements the full 24-round Keccak-f[1600] permutation with the five
//! step mappings: θ (theta), ρ (rho), π (pi), χ (chi), ι (iota).
//!
//! State layout: 5×5 matrix of 64-bit lanes indexed as `state[x + 5*y]`
//! where x is the column and y is the row (FIPS 202, §3.1.2).
//!
//! References:
//!   - NIST FIPS 202: SHA-3 Standard (August 2015)
//!   - The Keccak Reference, v3.0 (January 2011)
//!   - XKCP (eXtended Keccak Code Package) for optimisation patterns

/// Number of rounds in Keccak-f[1600].
pub const KECCAK_ROUNDS: usize = 24;

/// Round constants for the ι (iota) step of Keccak-f[1600].
///
/// Derived from the LFSR-based generation rule in FIPS 202, §3.2.5.
/// Each constant is applied to lane (0,0) after the χ step.
pub(crate) const RC: [u64; KECCAK_ROUNDS] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808A,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808B,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008A,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000A,
    0x0000_0000_8000_808B,
    0x8000_0000_0000_008B,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800A,
    0x8000_0000_8000_000A,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// Rotation offsets for the ρ (rho) step, indexed as `[x + 5*y]`.
///
/// These are the lane-specific rotation amounts defined in FIPS 202, §3.2.2.
/// Lane (0,0) has offset 0; all others are computed from the recurrence
/// relation (t+1)(t+2)/2 mod 64.
pub(crate) const RHO: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

/// The π (pi) step source indices: `new[i] = old[PI[i]]` after ρ rotation.
///
/// For each destination index `d = x + 5*y`, `PI[d]` gives the source
/// index `x' + 5*y'` where `x' = y, y' = (2*x + 3*y) mod 5` (inverted).
///
/// Pre-computed lookup avoids modular arithmetic in the hot loop.
pub(crate) const PI: [usize; 25] = [
    0, 6, 12, 18, 24, 3, 9, 10, 16, 22, 1, 7, 13, 19, 20, 4, 5, 11, 17, 23, 2, 8, 14, 15, 21,
];

/// Returns the ι-step round constants, in round order.
///
/// Exposed for the AVX-512 kernel (`simd_avx512.rs`), which re-derives its
/// own copy from this value in `tests::table_consistency` rather than
/// trusting a hand-transcribed duplicate. Gated identically to that
/// module (see `lib.rs`) so it isn't flagged as dead code in builds where
/// `simd_avx512` itself is excluded (`kernel`/non-x86_64).
#[cfg(all(not(feature = "kernel"), target_arch = "x86_64"))]
#[inline]
pub(crate) const fn round_constants() -> [u64; KECCAK_ROUNDS] {
    RC
}

/// Apply the full Keccak-f[1600] permutation in-place.
///
/// This is the core cryptographic primitive. It operates on 25 × 64-bit
/// lanes (1600 bits total) and applies 24 rounds of the five step mappings.
///
/// # Performance
///
/// The implementation is optimised for modern out-of-order CPUs:
/// - θ uses five column-parity accumulators (register-friendly)
/// - ρ+π are fused into a single pass with a pre-computed lookup table
/// - χ operates on 5-element row slices to maximise ILP
/// - The compiler can auto-vectorise the inner loops when beneficial
///
/// # Safety
///
/// Pure safe Rust — no `unsafe` blocks.
#[inline]
pub fn keccak_f1600(state: &mut [u64; 25]) {
    for round in RC.iter().take(KECCAK_ROUNDS) {
        // ===== θ (theta) step =====
        // Compute column parities and diffusion terms.
        let c0 = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
        let c1 = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
        let c2 = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
        let c3 = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
        let c4 = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

        let d0 = c4 ^ c1.rotate_left(1);
        let d1 = c0 ^ c2.rotate_left(1);
        let d2 = c1 ^ c3.rotate_left(1);
        let d3 = c2 ^ c4.rotate_left(1);
        let d4 = c3 ^ c0.rotate_left(1);

        state[0] ^= d0;
        state[5] ^= d0;
        state[10] ^= d0;
        state[15] ^= d0;
        state[20] ^= d0;
        state[1] ^= d1;
        state[6] ^= d1;
        state[11] ^= d1;
        state[16] ^= d1;
        state[21] ^= d1;
        state[2] ^= d2;
        state[7] ^= d2;
        state[12] ^= d2;
        state[17] ^= d2;
        state[22] ^= d2;
        state[3] ^= d3;
        state[8] ^= d3;
        state[13] ^= d3;
        state[18] ^= d3;
        state[23] ^= d3;
        state[4] ^= d4;
        state[9] ^= d4;
        state[14] ^= d4;
        state[19] ^= d4;
        state[24] ^= d4;

        // ===== ρ (rho) + π (pi) steps (fused) =====
        // Rotate each lane by its ρ offset, then permute via π lookup.
        let mut b = [0u64; 25];
        for i in 0..25 {
            b[i] = state[PI[i]].rotate_left(RHO[PI[i]]);
        }

        // ===== χ (chi) step =====
        // Non-linear mixing: each bit depends on two neighbours in the row.
        for y in 0..5 {
            let base = 5 * y;
            state[base] = b[base] ^ ((!b[base + 1]) & b[base + 2]);
            state[base + 1] = b[base + 1] ^ ((!b[base + 2]) & b[base + 3]);
            state[base + 2] = b[base + 2] ^ ((!b[base + 3]) & b[base + 4]);
            state[base + 3] = b[base + 3] ^ ((!b[base + 4]) & b[base]);
            state[base + 4] = b[base + 4] ^ ((!b[base]) & b[base + 1]);
        }

        // ===== ι (iota) step =====
        // Break symmetry by XORing a round constant into lane (0,0).
        state[0] ^= *round;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the all-zero state permutation produces the expected output.
    ///
    /// Reference: The Keccak team's KAT (Known Answer Test) for f[1600]
    /// on the zero state. First 4 lanes of the result after one full
    /// permutation of the zero state.
    #[test]
    fn test_keccak_f1600_zero_state() {
        let mut state = [0u64; 25];
        keccak_f1600(&mut state);

        // After permuting the all-zero state, lane 0 must NOT be zero
        // (the round constants break symmetry).
        assert_ne!(state[0], 0, "ι step must inject round constants");
        assert_ne!(state, [0u64; 25], "Permutation must modify state");
    }

    /// Verify determinism: applying f[1600] twice to the same input
    /// must produce identical results.
    #[test]
    fn test_keccak_f1600_deterministic() {
        let mut s1 = [0u64; 25];
        let mut s2 = [0u64; 25];
        // Seed with non-trivial data
        for i in 0..25 {
            s1[i] = i as u64 * 0x0123_4567_89AB_CDEF;
            s2[i] = s1[i];
        }
        keccak_f1600(&mut s1);
        keccak_f1600(&mut s2);
        assert_eq!(s1, s2);
    }

    /// Verify that the permutation is a bijection (invertible) by checking
    /// that different inputs produce different outputs.
    #[test]
    fn test_keccak_f1600_different_inputs() {
        let mut s1 = [0u64; 25];
        let mut s2 = [0u64; 25];
        s2[0] = 1; // single-bit difference

        keccak_f1600(&mut s1);
        keccak_f1600(&mut s2);

        assert_ne!(s1, s2, "Single-bit input difference must avalanche");
    }

    /// Verify the known output for Keccak-f[1600] on the zero state.
    ///
    /// Reference values from the XKCP (eXtended Keccak Code Package)
    /// test vectors and independently verified against the Python
    /// reference implementation.
    #[test]
    fn test_keccak_f1600_known_answer() {
        let mut state = [0u64; 25];
        keccak_f1600(&mut state);

        // Lane 0 after one permutation of zero state
        // Reference: XKCP KeccakF-1600-IntermediateValues.txt
        assert_eq!(
            state[0], 0xF1258F7940E1DDE7,
            "Lane 0 must match XKCP reference"
        );
        assert_eq!(
            state[1], 0x84D5CCF933C0478A,
            "Lane 1 must match XKCP reference"
        );
        assert_eq!(
            state[2], 0xD598261EA65AA9EE,
            "Lane 2 must match XKCP reference"
        );
        assert_eq!(
            state[3], 0xBD1547306F80494D,
            "Lane 3 must match XKCP reference"
        );
        assert_eq!(
            state[4], 0x8B284E056253D057,
            "Lane 4 must match XKCP reference"
        );
    }
}
