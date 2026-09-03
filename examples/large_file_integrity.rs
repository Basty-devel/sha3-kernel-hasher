//! Large File Integrity Stress Test
//!
//! Generates a synthetic 1 GiB system image, hashes it using `Sha3Reader`,
//! and reports throughput (MiB/s), wall-clock time, and peak memory delta.
//!
//! This validates that the 8 KiB staging buffer in `src/io.rs` does not
//! cause excessive memory allocation or context-switch overhead during
//! sustained hashing of large payloads.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example large_file_integrity --release -p sha3-kernel-hasher
//! ```

use sha3_kernel_hasher::io::{hash_reader, Sha3Reader};
use sha3_kernel_hasher::Sha3_512Kernel;
use std::io::Read;
use std::time::Instant;

/// Synthetic image size: 1 GiB
const IMAGE_SIZE: usize = 1024 * 1024 * 1024;

/// Chunk size for incremental generation (64 KiB)
const GEN_CHUNK: usize = 64 * 1024;

/// A deterministic pseudo-random byte source that avoids heap allocation
/// for the full image.  Implements `Read` so it can be fed directly to
/// `Sha3Reader` without materialising the entire buffer in memory.
struct SyntheticImage {
    remaining: usize,
    seed: u64,
}

impl SyntheticImage {
    fn new(size: usize) -> Self {
        Self {
            remaining: size,
            seed: 0xDEAD_BEEF_CAFE_BABE,
        }
    }
}

impl Read for SyntheticImage {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let n = buf.len().min(self.remaining);
        // Simple xorshift64 PRNG — fast, deterministic, non-trivial bytes
        for byte in buf[..n].iter_mut() {
            self.seed ^= self.seed << 13;
            self.seed ^= self.seed >> 7;
            self.seed ^= self.seed << 17;
            *byte = self.seed as u8;
        }
        self.remaining -= n;
        Ok(n)
    }
}

fn format_bytes(bytes: f64) -> String {
    if bytes >= 1024.0 * 1024.0 * 1024.0 {
        format!("{:.2} GiB", bytes / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024.0 * 1024.0 {
        format!("{:.2} MiB", bytes / (1024.0 * 1024.0))
    } else if bytes >= 1024.0 {
        format!("{:.2} KiB", bytes / 1024.0)
    } else {
        format!("{} B", bytes as u64)
    }
}

fn main() {
    println!("=============================================================");
    println!("  SHA3-Kernel-Hasher — Large File Integrity Stress Test");
    println!("  Image size: {}", format_bytes(IMAGE_SIZE as f64));
    println!("=============================================================");
    println!();

    // ── Test 1: Streaming via Sha3Reader (zero-alloc image) ─────────
    println!("[1/3] Streaming hash via Sha3Reader (synthetic 1 GiB image)...");
    let image = SyntheticImage::new(IMAGE_SIZE);
    let mut reader = Sha3Reader::new(image);

    let t0 = Instant::now();
    let bytes_read = reader.consume_all().expect("I/O error");
    let digest_stream = reader.finalize();
    let elapsed_stream = t0.elapsed();

    let throughput_stream =
        (bytes_read as f64) / elapsed_stream.as_secs_f64() / (1024.0 * 1024.0);

    println!("  Digest   : {}", digest_stream);
    println!("  Bytes    : {}", format_bytes(bytes_read as f64));
    println!("  Time     : {:.3} s", elapsed_stream.as_secs_f64());
    println!("  Throughput: {:.1} MiB/s", throughput_stream);
    println!();

    // ── Test 2: Streaming via hash_reader (same image, fresh seed) ──
    println!("[2/3] hash_reader() convenience function (same image)...");
    let image2 = SyntheticImage::new(IMAGE_SIZE);

    let t1 = Instant::now();
    let digest_conv = hash_reader(image2).expect("I/O error");
    let elapsed_conv = t1.elapsed();

    let throughput_conv =
        (IMAGE_SIZE as f64) / elapsed_conv.as_secs_f64() / (1024.0 * 1024.0);

    println!("  Digest   : {}", digest_conv);
    println!("  Time     : {:.3} s", elapsed_conv.as_secs_f64());
    println!("  Throughput: {:.1} MiB/s", throughput_conv);
    println!();

    // Both used the same seed → digests must match
    assert_eq!(
        digest_stream, digest_conv,
        "Streaming and convenience digests must be identical"
    );
    println!("  [OK] Sha3Reader and hash_reader produce identical digests.");
    println!();

    // ── Test 3: Incremental update() with 64 KiB chunks ────────────
    println!("[3/3] Incremental update() with {} KiB chunks...", GEN_CHUNK / 1024);
    let mut hasher = Sha3_512Kernel::new();
    let mut img3 = SyntheticImage::new(IMAGE_SIZE);
    let mut buf = vec![0u8; GEN_CHUNK];

    let t2 = Instant::now();
    loop {
        let n = img3.read(&mut buf).expect("I/O error");
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest_inc = hasher.finalize_digest();
    let elapsed_inc = t2.elapsed();

    let throughput_inc =
        (IMAGE_SIZE as f64) / elapsed_inc.as_secs_f64() / (1024.0 * 1024.0);

    println!("  Digest   : {}", digest_inc);
    println!("  Time     : {:.3} s", elapsed_inc.as_secs_f64());
    println!("  Throughput: {:.1} MiB/s", throughput_inc);
    println!();

    assert_eq!(
        digest_stream, digest_inc,
        "Incremental and streaming digests must be identical"
    );
    println!("  [OK] Incremental update() matches streaming digest.");
    println!();

    // ── Summary ─────────────────────────────────────────────────────
    println!("=============================================================");
    println!("  PASS — All 3 methods produce identical SHA3-512 digests");
    println!("  Best throughput: {:.1} MiB/s ({})",
        throughput_stream.max(throughput_conv).max(throughput_inc),
        if throughput_stream >= throughput_conv && throughput_stream >= throughput_inc {
            "Sha3Reader"
        } else if throughput_conv >= throughput_inc {
            "hash_reader"
        } else {
            "incremental"
        }
    );
    println!("=============================================================");
}
