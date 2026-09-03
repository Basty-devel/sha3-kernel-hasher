//! Streaming I/O Hashing via `std::io::Read`
//!
//! Provides [`hash_reader`] and [`Sha3Reader`] for computing SHA3-512
//! digests over arbitrary byte streams — files, network sockets,
//! memory-mapped regions, or any type implementing [`std::io::Read`].
//!
//! This module is only available when the `std` feature is enabled
//! (the default).  Kernel-mode callers should use the incremental
//! `update` / `finalize` API on [`Sha3_512Kernel`] directly.
//!
//! # Design
//!
//! The reader adapter uses an internal 8 KiB staging buffer to
//! amortise system-call overhead while keeping stack usage bounded.
//! Each buffer-full is fed to the sponge via `absorb`, and the final
//! partial buffer triggers the SHA3 padding and squeeze phases.
//!
//! # Examples
//!
//! ```no_run
//! use sha3_kernel_hasher::io::hash_reader;
//! use std::fs::File;
//!
//! let file = File::open("firmware.bin").unwrap();
//! let digest = hash_reader(file).unwrap();
//! println!("SHA3-512: {}", digest);
//! ```
//!
//! # References
//!
//! - National Institute of Standards and Technology (2015) *SHA-3
//!   Standard: Permutation-Based Hash and Extendable-Output Functions*.
//!   FIPS PUB 202. Gaithersburg, MD: NIST.
//!   doi:10.6028/NIST.FIPS.202.

use std::io::{self, Read};

use crate::hex::Sha3Digest;
use crate::Sha3_512Kernel;

/// Default internal buffer size (8 KiB).
const IO_BUF_SIZE: usize = 8 * 1024;

/// Compute the SHA3-512 digest of an entire byte stream.
///
/// Reads `reader` to completion using an 8 KiB staging buffer and
/// returns the resulting [`Sha3Digest`].  I/O errors are propagated
/// transparently.
///
/// # Arguments
///
/// * `reader` — Any type implementing [`std::io::Read`].
///
/// # Errors
///
/// Returns `io::Error` if the underlying reader fails.
///
/// # Examples
///
/// ```no_run
/// use sha3_kernel_hasher::io::hash_reader;
/// use std::io::Cursor;
///
/// let data = b"The quick brown fox";
/// let digest = hash_reader(std::io::Cursor::new(data)).unwrap();
/// println!("{}", digest);
/// ```
pub fn hash_reader<R: Read>(reader: R) -> io::Result<Sha3Digest> {
    let mut wrapper = Sha3Reader::new(reader);
    wrapper.consume_all()?;
    Ok(wrapper.finalize())
}

/// Compute the SHA3-512 digest of a file at the given path.
///
/// Convenience function that opens the file and delegates to
/// [`hash_reader`].
///
/// # Examples
///
/// ```no_run
/// use sha3_kernel_hasher::io::hash_file;
///
/// let digest = hash_file("firmware.bin").unwrap();
/// println!("{}", digest);
/// ```
pub fn hash_file<P: AsRef<std::path::Path>>(path: P) -> io::Result<Sha3Digest> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    hash_reader(reader)
}

/// A streaming SHA3-512 hasher wrapping an [`std::io::Read`] source.
///
/// Useful when the caller needs fine-grained control over how much
/// data is consumed per step (e.g., progress reporting).
///
/// # Examples
///
/// ```no_run
/// use sha3_kernel_hasher::io::Sha3Reader;
/// use std::io::Cursor;
///
/// let data = b"incremental stream";
/// let mut reader = Sha3Reader::new(Cursor::new(data));
/// reader.consume_all().unwrap();
/// let digest = reader.finalize();
/// println!("{}", digest);
/// ```
pub struct Sha3Reader<R: Read> {
    inner: R,
    hasher: Sha3_512Kernel,
    buf: [u8; IO_BUF_SIZE],
    /// Total bytes consumed so far.
    bytes_consumed: u64,
}

impl<R: Read> Sha3Reader<R> {
    /// Wrap a reader in a streaming SHA3-512 hasher.
    pub fn new(reader: R) -> Self {
        Self {
            inner: reader,
            hasher: Sha3_512Kernel::new(),
            buf: [0u8; IO_BUF_SIZE],
            bytes_consumed: 0,
        }
    }

    /// Read and hash the next chunk from the underlying reader.
    ///
    /// Returns the number of bytes consumed (0 at EOF).
    pub fn consume_chunk(&mut self) -> io::Result<usize> {
        let n = self.inner.read(&mut self.buf)?;
        if n > 0 {
            self.hasher.update(&self.buf[..n]);
            self.bytes_consumed += n as u64;
        }
        Ok(n)
    }

    /// Read the entire stream to completion, hashing all data.
    pub fn consume_all(&mut self) -> io::Result<u64> {
        loop {
            let n = self.consume_chunk()?;
            if n == 0 {
                break;
            }
        }
        Ok(self.bytes_consumed)
    }

    /// Finalize the hash and return the digest.
    ///
    /// After calling this, the `Sha3Reader` should not be reused.
    pub fn finalize(&mut self) -> Sha3Digest {
        Sha3Digest::new(self.hasher.finalize())
    }

    /// Return the total number of bytes consumed so far.
    pub fn bytes_consumed(&self) -> u64 {
        self.bytes_consumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_hash_reader_empty() {
        let digest = hash_reader(Cursor::new(b"")).unwrap();
        // Must match SHA3-512 of empty string
        let mut hasher = Sha3_512Kernel::new();
        let expected = Sha3Digest::new(hasher.hash(b""));
        assert_eq!(digest, expected);
    }

    #[test]
    fn test_hash_reader_matches_direct() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let digest = hash_reader(Cursor::new(data)).unwrap();

        let mut hasher = Sha3_512Kernel::new();
        let expected = Sha3Digest::new(hasher.hash(data));
        assert_eq!(digest, expected);
    }

    #[test]
    fn test_hash_reader_large() {
        let data = vec![0xCDu8; 100_000];
        let digest = hash_reader(Cursor::new(&data)).unwrap();

        let mut hasher = Sha3_512Kernel::new();
        let expected = Sha3Digest::new(hasher.hash(&data));
        assert_eq!(digest, expected);
    }

    #[test]
    fn test_sha3_reader_bytes_consumed() {
        let data = vec![0x42u8; 12345];
        let mut reader = Sha3Reader::new(Cursor::new(&data));
        reader.consume_all().unwrap();
        assert_eq!(reader.bytes_consumed(), 12345);
    }
}
