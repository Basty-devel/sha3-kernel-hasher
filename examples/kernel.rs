//! Kernel-mode hashing example for SHA3-Kernel-Hasher
//! 
//! Demonstrates safe memory hashing in kernel environments

use sha3_kernel_hasher::{Sha3_512Kernel, MemoryRegion, kernel_safe};

fn hash_hex(h: &[u8; 64]) -> String {
    h.iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    println!("🖥️ SHA3-Kernel-Hasher - Kernel Usage Example");
    
    // Create hasher with kernel-optimized configuration
    let mut hasher = Sha3_512Kernel::new();
    
    // Example memory regions (simulated kernel addresses)
    let regions = vec![
        MemoryRegion {
            base_address: 0xFFFF8800000000,
            size: 4096,
            is_readable: true,
            is_writable: false,
            is_executable: true,
        },
        MemoryRegion {
            base_address: 0xFFFF8800010000,
            size: 8192,
            is_readable: true,
            is_writable: true,
            is_executable: false,
        },
        MemoryRegion {
            base_address: 0xFFFF8800020000,
            size: 16384,
            is_readable: true,
            is_writable: false,
            is_executable: false,
        },
    ];
    
    println!("🔍 Hashing {} memory regions...", regions.len());
    
    // Safe kernel hashing with processor state management
    let hashes = unsafe {
        // Save processor state for SIMD operations
        let _saved_state = kernel_safe::save_processor_state();
        
        // Hash each memory region
        let results = hasher.hash_memory_regions(&regions);
        
        // Restore processor state
        kernel_safe::restore_processor_state(_saved_state);
        
        results
    };
    
    // Display results
    for (i, (region, hash)) in regions.iter().zip(hashes.iter()).enumerate() {
        println!("Region {}: Address 0x{:x}, Size {}KB", i + 1, region.base_address, region.size / 1024);
        println!("  Readable: {}, Writable: {}, Executable: {}", 
                 region.is_readable, region.is_writable, region.is_executable);
        println!("  SHA3-512: {}", hash_hex(hash));
        
        // Verify hash integrity
        if verify_hash_integrity(hash) {
            println!("  ✅ Hash integrity verified");
        } else {
            println!("  ❌ Hash integrity check failed");
        }
        println!();
    }
    
    // Demonstrate incremental hashing
    println!("🔄 Incremental Hashing Example:");
    let mut incremental_hasher = Sha3_512Kernel::new();
    
    let data_chunks: Vec<&[u8]> = vec![
        b"First part of ",
        b"data to be hashed ",
        b"incrementally with ",
        b"SHA3-Kernel-Hasher",
    ];
    
    for chunk in data_chunks {
        incremental_hasher.update(chunk);
        println!("  Updated with: {}", String::from_utf8_lossy(chunk));
    }
    
    let final_hash = incremental_hasher.finalize();
    println!("  Final SHA3-512: {}", hash_hex(&final_hash));
    
    // Timing demonstration — NOT a real SIMD comparison.
    //
    // `PerformanceConfig::use_avx2`/`use_avx512` are currently no-ops:
    // `Sha3_512Kernel::hash` always runs the same scalar Keccak-f[1600]
    // path regardless of this config (see the crate-level "SIMD Status"
    // docs). Both hashers below execute identical code; any timing delta
    // you see is measurement noise, not acceleration. This is here to
    // demonstrate the config-struct API shape, not a performance claim.
    println!("\n⏱️  Timing (scalar-only — see note above):");
    let performance_data = vec![0u8; 1024 * 100]; // 100KB

    let scalar_start = std::time::Instant::now();
    let scalar_hash = hasher.hash(&performance_data);
    let scalar_time = scalar_start.elapsed();

    let mut avx2_hasher = Sha3_512Kernel::with_config(sha3_kernel_hasher::PerformanceConfig {
        use_avx512: false,
        use_avx2: true,
        ..Default::default()
    });
    let avx2_start = std::time::Instant::now();
    let avx2_hash = avx2_hasher.hash(&performance_data);
    let avx2_time = avx2_start.elapsed();

    println!("  Scalar (use_avx2: false): {:?} - {}", scalar_time, hash_hex(&scalar_hash));
    println!("  use_avx2: true (same code path today): {:?} - {}", avx2_time, hash_hex(&avx2_hash));
    println!("  (Digests are identical: {})", scalar_hash == avx2_hash);

    println!("✅ Kernel example completed!");
}

/// Verify hash integrity (simplified)
fn verify_hash_integrity(hash: &[u8; 64]) -> bool {
    // Simple integrity check - in real implementation, this would
    // verify against known good hashes or use cryptographic signatures
    !hash.iter().all(|&b| b == 0)
}
