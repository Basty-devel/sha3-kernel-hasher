use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sha3_kernel_hasher::{PerformanceConfig, Sha3_512Kernel};

fn bench_hash_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sha3_512_hash");

    for size in [64, 256, 1024, 4096, 16384, 65536] {
        let data = vec![0xABu8; size];
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &data, |b, data| {
            let mut hasher = Sha3_512Kernel::with_config(PerformanceConfig {
                use_avx512: false,
                use_avx2: false,
                ..Default::default()
            });
            b.iter(|| black_box(hasher.hash(data)));
        });

        group.bench_with_input(BenchmarkId::new("avx2", size), &data, |b, data| {
            let mut hasher = Sha3_512Kernel::with_config(PerformanceConfig {
                use_avx512: false,
                use_avx2: true,
                ..Default::default()
            });
            b.iter(|| black_box(hasher.hash(data)));
        });

        group.bench_with_input(BenchmarkId::new("avx512", size), &data, |b, data| {
            let mut hasher = Sha3_512Kernel::with_config(PerformanceConfig {
                use_avx512: true,
                use_avx2: true,
                ..Default::default()
            });
            b.iter(|| black_box(hasher.hash(data)));
        });

        group.bench_with_input(BenchmarkId::new("auto", size), &data, |b, data| {
            let mut hasher = Sha3_512Kernel::new();
            b.iter(|| black_box(hasher.hash(data)));
        });
    }

    group.finish();
}

fn bench_incremental(c: &mut Criterion) {
    let mut group = c.benchmark_group("sha3_512_incremental");
    let data = b"The quick brown fox jumps over the lazy dog. Bunker DNS SHA3-512 benchmark.";

    group.bench_function("8-byte_chunks", |b| {
        b.iter(|| {
            let mut hasher = Sha3_512Kernel::new();
            for chunk in data.chunks(8) {
                hasher.update(chunk);
            }
            black_box(hasher.finalize())
        })
    });

    group.bench_function("single_call", |b| {
        let mut hasher = Sha3_512Kernel::new();
        b.iter(|| black_box(hasher.hash(data)))
    });

    group.finish();
}

fn bench_throughput_1mb(c: &mut Criterion) {
    let mut group = c.benchmark_group("sha3_512_throughput");
    let data = vec![0xCDu8; 1024 * 1024]; // 1 MB
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("1MB_auto", |b| {
        let mut hasher = Sha3_512Kernel::new();
        b.iter(|| black_box(hasher.hash(&data)))
    });

    group.bench_function("1MB_scalar", |b| {
        let mut hasher = Sha3_512Kernel::with_config(PerformanceConfig {
            use_avx512: false,
            use_avx2: false,
            ..Default::default()
        });
        b.iter(|| black_box(hasher.hash(&data)))
    });

    group.finish();
}

fn bench_hash_many(c: &mut Criterion) {
    let mut group = c.benchmark_group("sha3_512_hash_many");
    // 64 independent 4 KiB messages: representative of batch workloads
    // (hashing many files/records at once) where hash_many's SIMD
    // batching has real independent work to parallelize, unlike a
    // single large message.
    let messages: Vec<Vec<u8>> = (0..64).map(|i| vec![(i & 0xFF) as u8; 4096]).collect();
    let refs: Vec<&[u8]> = messages.iter().map(|m| m.as_slice()).collect();
    group.throughput(Throughput::Bytes((messages.len() * 4096) as u64));

    group.bench_function("64x4KiB_avx512", |b| {
        let hasher = Sha3_512Kernel::with_config(PerformanceConfig {
            use_avx512: true,
            use_avx2: false,
            ..Default::default()
        });
        b.iter(|| black_box(hasher.hash_many(&refs)));
    });

    group.bench_function("64x4KiB_avx2", |b| {
        let hasher = Sha3_512Kernel::with_config(PerformanceConfig {
            use_avx512: false,
            use_avx2: true,
            ..Default::default()
        });
        b.iter(|| black_box(hasher.hash_many(&refs)));
    });

    group.bench_function("64x4KiB_scalar_loop", |b| {
        let hasher = Sha3_512Kernel::with_config(PerformanceConfig {
            use_avx512: false,
            use_avx2: false,
            ..Default::default()
        });
        b.iter(|| black_box(hasher.hash_many(&refs)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hash_sizes,
    bench_incremental,
    bench_throughput_1mb,
    bench_hash_many
);
criterion_main!(benches);
