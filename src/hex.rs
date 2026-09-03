//! Hexadecimal Encoding Utilities for SHA3-512 Digests
//!
//! Provides efficient hex encoding and a newtype wrapper (`Sha3Digest`)
//! with `Display`, `LowerHex`, and `UpperHex` implementations for
//! ergonomic output formatting.
//!
//! # Design Rationale
//!
//! Raw `[u8; 64]` arrays lack formatting traits, forcing callers to
//! write ad-hoc hex loops.  `Sha3Digest` wraps the array and exposes
//! standard Rust formatting so that `println!("{}", digest)` and
//! `format!("{:X}", digest)` work out of the box.
//!
//! The encoder is branch-free and operates on a pre-computed nibble
//! lookup table, avoiding data-dependent branches that could leak
//! information through timing side-channels (cf. Bernstein, 2005).
//!
//! # References
//!
//! - Bernstein, D.J. (2005) 'Cache-timing attacks on AES', Available at:
//!   <https://cr.yp.to/antiforgery/cachetiming-20050414.pdf>.

#[cfg(feature = "kernel")]
use alloc::string::String;
#[cfg(not(feature = "kernel"))]
use std::string::String;

use core::fmt;

/// Lowercase hex nibble lookup table (ASCII `0-9a-f`).
const HEX_LOWER: [u8; 16] = *b"0123456789abcdef";

/// Uppercase hex nibble lookup table (ASCII `0-9A-F`).
const HEX_UPPER: [u8; 16] = *b"0123456789ABCDEF";

/// A SHA3-512 digest wrapper providing formatting and comparison traits.
///
/// When the `serde` feature is enabled, this type implements
/// `Serialize` and `Deserialize` for seamless integration with
/// serialisation frameworks.
///
/// # Examples
///
/// ```
/// use sha3_kernel_hasher::{Sha3_512Kernel, Sha3Digest};
///
/// let mut hasher = Sha3_512Kernel::new();
/// let digest = Sha3Digest::new(hasher.hash(b"hello"));
/// println!("lowercase : {}", digest);
/// println!("uppercase : {:X}", digest);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha3Digest(pub [u8; 64]);

#[cfg(feature = "serde")]
mod digest_serde {
    use super::Sha3Digest;
    use core::fmt;
    use serde::de::{self, SeqAccess, Visitor};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for Sha3Digest {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_bytes(&self.0)
        }
    }

    impl<'de> Deserialize<'de> for Sha3Digest {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct DigestVisitor;

            impl<'de> Visitor<'de> for DigestVisitor {
                type Value = Sha3Digest;

                fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("64 bytes (SHA3-512 digest)")
                }

                fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Sha3Digest, E> {
                    if v.len() != 64 {
                        return Err(de::Error::invalid_length(v.len(), &"64"));
                    }
                    let mut buf = [0u8; 64];
                    buf.copy_from_slice(v);
                    Ok(Sha3Digest(buf))
                }

                fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Sha3Digest, A::Error> {
                    let mut buf = [0u8; 64];
                    for (i, slot) in buf.iter_mut().enumerate() {
                        *slot = seq
                            .next_element::<u8>()?
                            .ok_or_else(|| de::Error::invalid_length(i, &"64"))?;
                    }
                    Ok(Sha3Digest(buf))
                }
            }

            deserializer.deserialize_bytes(DigestVisitor)
        }
    }
}

impl Sha3Digest {
    /// Wrap a raw 64-byte hash in a `Sha3Digest`.
    #[inline]
    pub fn new(bytes: [u8; 64]) -> Self {
        Self(bytes)
    }

    /// Return a reference to the underlying byte array.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 64] {
        &self.0
    }

    /// Consume `self` and return the inner byte array.
    #[inline]
    pub fn into_bytes(self) -> [u8; 64] {
        self.0
    }

    /// Encode the digest as a lowercase hex string.
    pub fn to_hex_lower(&self) -> String {
        encode_hex(&self.0, &HEX_LOWER)
    }

    /// Encode the digest as an uppercase hex string.
    pub fn to_hex_upper(&self) -> String {
        encode_hex(&self.0, &HEX_UPPER)
    }

    /// Attempt to decode a 128-character hex string into a `Sha3Digest`.
    ///
    /// Returns `None` if the string length is not 128 or contains
    /// non-hex characters.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 128 {
            return None;
        }
        let mut bytes = [0u8; 64];
        for (i, byte) in bytes.iter_mut().enumerate() {
            let hi = nibble_from_ascii(hex.as_bytes()[i * 2])?;
            let lo = nibble_from_ascii(hex.as_bytes()[i * 2 + 1])?;
            *byte = (hi << 4) | lo;
        }
        Some(Self(bytes))
    }
}

impl fmt::Display for Sha3Digest {
    /// Formats the digest as a 128-character lowercase hex string.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &b in &self.0 {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

impl fmt::LowerHex for Sha3Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &b in &self.0 {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

impl fmt::UpperHex for Sha3Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &b in &self.0 {
            write!(f, "{:02X}", b)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Sha3Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sha3Digest({})", self)
    }
}

impl From<[u8; 64]> for Sha3Digest {
    #[inline]
    fn from(bytes: [u8; 64]) -> Self {
        Self(bytes)
    }
}

impl From<Sha3Digest> for [u8; 64] {
    #[inline]
    fn from(digest: Sha3Digest) -> Self {
        digest.0
    }
}

impl AsRef<[u8]> for Sha3Digest {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Encode a byte slice to hex using the given nibble table.
fn encode_hex(bytes: &[u8], table: &[u8; 16]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(table[(b >> 4) as usize] as char);
        out.push(table[(b & 0x0F) as usize] as char);
    }
    out
}

/// Convert an ASCII hex character to its 4-bit value.
#[inline]
fn nibble_from_ascii(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_hex() {
        let bytes = [0xABu8; 64];
        let digest = Sha3Digest::new(bytes);
        let hex = digest.to_hex_lower();
        let parsed = Sha3Digest::from_hex(&hex).expect("valid hex");
        assert_eq!(parsed, digest);
    }

    #[test]
    fn test_display_format() {
        let bytes = [0x00u8; 64];
        let digest = Sha3Digest::new(bytes);
        let s = format!("{}", digest);
        assert_eq!(s.len(), 128);
        assert!(s.chars().all(|c| c == '0'));
    }

    #[test]
    fn test_upper_hex() {
        let mut bytes = [0u8; 64];
        bytes[0] = 0xDE;
        bytes[1] = 0xAD;
        let digest = Sha3Digest::new(bytes);
        let s = format!("{:X}", digest);
        assert!(s.starts_with("DEAD"));
    }

    #[test]
    fn test_from_hex_invalid_length() {
        assert!(Sha3Digest::from_hex("abcd").is_none());
    }

    #[test]
    fn test_from_hex_invalid_char() {
        let bad = "g".repeat(128);
        assert!(Sha3Digest::from_hex(&bad).is_none());
    }
}
