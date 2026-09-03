//! SHA3-Kernel-Hasher — Performance Measurement Example
//!
//! Standalone benchmark that measures SHA3-512 throughput across multiple
//! payload sizes and hashing modes (one-shot vs. incremental).
//!
//! Run: `cargo run --example benchmark --release`

use sha3_kernel_hasher::Sha3_512Kernel;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Number of iterations per measurement to amortise timer overhead.
const WARMUP_ITERS: usize = 100;
const BENCH_ITERS: usize = 1_000;

fn main() {
    println!("SHA3-Kernel-Hasher  Performance Benchmark");
    println!("==========================================\n");

    let features = Sha3_512Kernel::new().simd_features();
    println!("SIMD features detected:");
    println!("  AVX-512 : {}", features.avx512_available);
    println!("  AVX2    : {}", features.avx2_available);
    println!("  SSE4.2  : {}", features.sse42_available);
    println!("  BMI2    : {}", features.bmi2_available);
    println!();

    // ---- One-shot hashing across payload sizes ----
    println!("{:<12} {:>12} {:>14}", "Size", "Time/hash", "Throughput");
    println!("{:-<12} {:-<12} {:-<14}", "", "", "");

    let sizes: &[usize] = &[64, 256, 1024, 4096, 16384, 65536, 1_048_576];

    for &size in sizes {
        let data = vec![0xA3u8; size];
        let mut hasher = Sha3_512Kernel::new();

        // Warmup
        for _ in 0..WARMUP_ITERS {
            black_box(hasher.hash(&data));
        }

        // Measure
        let start = Instant::now();
        for _ in 0..BENCH_ITERS {
            black_box(hasher.hash(&data));
        }
        let elapsed = start.elapsed();

        let per_hash = elapsed / BENCH_ITERS as u32;
        let throughput_mib =
            (size as f64 * BENCH_ITERS as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);

        println!(
            "{:<12} {:>12} {:>10.1} MiB/s",
            format_size(size),
            format_duration(per_hash),
            throughput_mib,
        );
    }

    println!();

    // ---- Incremental vs one-shot comparison ----
    println!("Incremental vs One-Shot (4 KiB payload, 64-byte chunks)");
    println!("{:-<56}", "");

    let data = vec![0x42u8; 4096];

    // One-shot
    let mut hasher = Sha3_512Kernel::new();
    for _ in 0..WARMUP_ITERS {
        black_box(hasher.hash(&data));
    }
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        black_box(hasher.hash(&data));
    }
    let oneshot_elapsed = start.elapsed();

    // Incremental (64-byte chunks)
    for _ in 0..WARMUP_ITERS {
        hasher.reset();
        for chunk in data.chunks(64) {
            hasher.update(chunk);
        }
        black_box(hasher.finalize());
    }
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        hasher.reset();
        for chunk in data.chunks(64) {
            hasher.update(chunk);
        }
        black_box(hasher.finalize());
    }
    let incr_elapsed = start.elapsed();

    println!(
        "  One-shot    : {} / hash",
        format_duration(oneshot_elapsed / BENCH_ITERS as u32)
    );
    println!(
        "  Incremental : {} / hash",
        format_duration(incr_elapsed / BENCH_ITERS as u32)
    );

    println!("\nDone.");
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{} MiB", bytes / 1_048_576)
    } else if bytes >= 1024 {
        format!("{} KiB", bytes / 1024)
    } else {
        format!("{} B", bytes)
    }
}

fn format_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos >= 1_000_000 {
        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
    } else if nanos >= 1_000 {
        format!("{:.2} us", nanos as f64 / 1_000.0)
    } else {
        format!("{} ns", nanos)
    }
}
