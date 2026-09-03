//! Constant-Time Comparison for SHA3-512 Digests
//!
//! Implements timing-safe equality comparison to prevent side-channel
//! attacks when verifying hash digests. A naive byte-by-byte comparison
//! with early exit leaks information about which byte position differs
//! first, enabling an adversary to iteratively reconstruct a target
//! digest (Kocher, 1996).
//!
//! # Algorithm
//!
//! The comparison accumulates the XOR of every byte pair into a single
//! accumulator. The result is zero if and only if all bytes are equal.
//! Every byte is always visited regardless of where a difference occurs,
//! ensuring constant execution time relative to the digest length.
//!
//! # References
//!
//! - Kocher, P.C. (1996) 'Timing Attacks on Implementations of
//!   Diffie-Hellman, RSA, DSS, and Other Systems', in *Advances in
//!   Cryptology — CRYPTO '96*. Lecture Notes in Computer Science,
//!   vol. 1109. Berlin: Springer, pp. 104–113.
//!   doi:10.1007/3-540-68697-5_9.
//!
//! - Bernstein, D.J., Lange, T. and Schwabe, P. (2012) 'The Security
//!   Impact of a New Cryptographic Library', in *Progress in Cryptology
//!   — LATINCRYPT 2012*. Lecture Notes in Computer Science, vol. 7533.
//!   Berlin: Springer, pp. 159–176.

/// Compare two byte slices in constant time.
///
/// Returns `true` if and only if `a` and `b` have equal length and
/// identical contents.  The execution time depends only on the length
/// of the slices, never on the position of the first differing byte.
///
/// # Examples
///
/// ```
/// use sha3_kernel_hasher::ct::ct_eq;
///
/// let a = [0xABu8; 64];
/// let b = [0xABu8; 64];
/// assert!(ct_eq(&a, &b));
/// ```
#[inline]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc: u8 = 0;
    for i in 0..a.len() {
        acc |= a[i] ^ b[i];
    }
    acc == 0
}

/// Compare two 64-byte SHA3-512 digests in constant time.
///
/// This is a specialised version of [`ct_eq`] that operates on
/// fixed-size arrays, eliminating the length check and enabling
/// the compiler to unroll the loop.
///
/// # Examples
///
/// ```
/// use sha3_kernel_hasher::ct::ct_hash_eq;
///
/// let h1 = [0u8; 64];
/// let h2 = [0u8; 64];
/// assert!(ct_hash_eq(&h1, &h2));
/// ```
#[inline]
pub fn ct_hash_eq(a: &[u8; 64], b: &[u8; 64]) -> bool {
    let mut acc: u8 = 0;
    for i in 0..64 {
        acc |= a[i] ^ b[i];
    }
    acc == 0
}

/// Select between two values in constant time based on a boolean
/// condition.
///
/// Returns `a` if `condition` is `true`, otherwise `b`.  Both branches
/// are always evaluated; the selection is performed with bitwise
/// operations to avoid branch prediction leakage.
#[inline]
pub fn ct_select(condition: bool, a: u8, b: u8) -> u8 {
    let mask = (condition as u8).wrapping_neg(); // 0xFF if true, 0x00 if false
    (mask & a) | (!mask & b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ct_eq_equal() {
        let a = [0x42u8; 64];
        let b = [0x42u8; 64];
        assert!(ct_eq(&a, &b));
        assert!(ct_hash_eq(&a, &b));
    }

    #[test]
    fn test_ct_eq_different() {
        let a = [0x42u8; 64];
        let mut b = [0x42u8; 64];
        b[63] = 0x43;
        assert!(!ct_eq(&a, &b));
        assert!(!ct_hash_eq(&a, &b));
    }

    #[test]
    fn test_ct_eq_different_lengths() {
        let a = [0u8; 64];
        let b = [0u8; 32];
        assert!(!ct_eq(&a, &b));
    }

    #[test]
    fn test_ct_select() {
        assert_eq!(ct_select(true, 0xAA, 0xBB), 0xAA);
        assert_eq!(ct_select(false, 0xAA, 0xBB), 0xBB);
    }

    #[test]
    fn test_ct_eq_all_zero() {
        let a = [0u8; 64];
        let b = [0u8; 64];
        assert!(ct_hash_eq(&a, &b));
    }

    #[test]
    fn test_ct_eq_single_bit_diff() {
        let a = [0u8; 64];
        let mut b = [0u8; 64];
        b[0] = 1;
        assert!(!ct_hash_eq(&a, &b));
    }
}
