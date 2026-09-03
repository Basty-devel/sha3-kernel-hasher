//! Basic usage example for SHA3-Kernel-Hasher
//!
//! Demonstrates simple hashing operations

use sha3_kernel_hasher::Sha3_512Kernel;

fn hash_hex(h: &[u8; 64]) -> String {
    h.iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    println!("🚀 SHA3-Kernel-Hasher - Basic Usage Example");

    // Create hasher
    let mut hasher = Sha3_512Kernel::new();

    // Hash simple string
    let data = b"Hello, SHA3-Kernel-Hasher!";
    let hash = hasher.hash(data);

    println!("📝 Input: {}", String::from_utf8_lossy(data));
    println!("SHA3-512: {}", hash_hex(&hash));

    // Hash larger data
    let large_data = vec![0u8; 1024 * 1024]; // 1MB
    let large_hash = hasher.hash(&large_data);

    println!("Large data (1MB) hash: {}", hash_hex(&large_hash));

    // Demonstrate SIMD features
    let features = hasher.simd_features();
    println!("⚡ SIMD Features:");
    println!("   AVX-512: {}", features.avx512_available);
    println!("   AVX2: {}", features.avx2_available);
    println!("   SSE4.2: {}", features.sse42_available);
    println!("   BMI2: {}", features.bmi2_available);

    // Performance comparison
    println!("\n🏁 Performance Comparison:");
    let start = std::time::Instant::now();

    // Multiple hashes to show performance
    for i in 0..1000 {
        let test_data = format!("Test data {}", i);
        hasher.hash(test_data.as_bytes());
    }

    let duration = start.elapsed();
    println!("   1000 hashes in {:?}", duration);
    println!("   Average: {:?} per hash", duration / 1000);

    println!("✅ Basic example completed!");
}
