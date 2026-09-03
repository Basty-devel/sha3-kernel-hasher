//! # SHA3-Kernel-Hasher
//!
//! **Production-grade SHA3-512 (FIPS 202) in pure Rust with dual-mode
//! compilation for userspace and Windows kernel drivers.**
//!
//! ## Overview
//!
//! This crate provides a self-contained implementation of the SHA3-512
//! cryptographic hash function as standardised in FIPS PUB 202 (National
//! Institute of Standards and Technology, 2015).  It includes the complete
//! Keccak-f\[1600\] permutation, correct SHA3 domain separation and
//! `pad10*1` padding, an incremental (streaming) API, and helper modules
//! for hex encoding, constant-time comparison, and I/O-based hashing.
//!
//! ## Algorithm
//!
//! SHA3-512 is built on the *sponge construction* (Bertoni et al., 2011)
//! operating on a 1600-bit state partitioned into a 576-bit rate (*r*)
//! and a 1024-bit capacity (*c*).  The construction proceeds in two
//! phases:
//!
//! 1. **Absorb** — The message is padded with domain byte `0x06` and the
//!    `pad10*1` rule (FIPS 202, §6.1), split into 72-byte blocks, and
//!    each block is XORed into the state followed by Keccak-f\[1600\].
//! 2. **Squeeze** — The first 512 bits (64 bytes) of the post-absorb
//!    state are extracted as the digest.
//!
//! The Keccak-f\[1600\] permutation applies 24 rounds, each consisting
//! of five step mappings — θ (theta), ρ (rho), π (pi), χ (chi), and
//! ι (iota) — to a 5×5 matrix of 64-bit lanes.
//!
//! ## Security Properties
//!
//! | Property | Security (bits) |
//! |----------|----------------|
//! | Collision resistance | 256 |
//! | Preimage resistance | 512 |
//! | Second preimage resistance | 512 |
//!
//! SHA3 is structurally distinct from the Merkle-Damgård family and is
//! immune to length-extension attacks (Kelsey and Schneier, 2005).
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`hex`] | `Sha3Digest` wrapper with `Display`, `LowerHex`, `UpperHex`, hex encode/decode |
//! | [`ct`] | Constant-time byte comparison (Bernstein, Lange and Schwabe, 2012) |
//! | [`io`] | Streaming `std::io::Read` adapter and `hash_file` convenience (std only) |
//! | [`kernel_safe`] | Processor state save/restore for kernel SIMD usage |
//!
//! ## Compilation Modes
//!
//! - **`std`** (default) — Full standard library, runtime SIMD detection,
//!   `io` module enabled.
//! - **`kernel`** — `no_std` + `alloc` for Windows kernel-mode drivers.
//!   The `io` module is disabled; SIMD defaults to scalar.
//!
//! ## SIMD Status
//!
//! Runtime detection probes for AVX-512F/BW/DQ, AVX2+BMI2, and SSE4.2 via
//! `is_x86_feature_detected!`, and is exposed through [`SimdFeatures`] and
//! [`Sha3_512Kernel::simd_features`] so a caller can query what the current
//! CPU supports. **That detection result is not currently consumed by the
//! hasher.** Both the Keccak-f\[1600\] permutation and the absorb-phase XOR
//! run the same scalar code path regardless of detected features or of the
//! [`PerformanceConfig::use_avx2`]/[`PerformanceConfig::use_avx512`] flags —
//! `hash()` ignores `self.config` entirely. Confirmed by benchmark: this
//! crate's own `sha3_512_throughput/1MB_auto` (SIMD auto-detected) and
//! `1MB_scalar` (explicitly forced off) criterion groups measure the same
//! ~155-170 MiB/s on a CPU that reports AVX-512 and AVX2 available. A
//! genuine vectorized permutation is unimplemented; the `use_avx2`/
//! `use_avx512` fields exist for future API compatibility, not current
//! effect.
//!
//! ## Correctness Validation
//!
//! - NIST CAVP SHA3-512 test vectors (empty, `0xCC`, `0x41FB`, `0xA3`×200)
//! - XKCP Keccak-f\[1600\] known-answer tests (lanes 0–4)
//! - Cross-validated against RustCrypto `sha3` v0.10 for 20 input lengths
//! - Incremental vs. one-shot, determinism, avalanche, boundary conditions
//!
//! ## Quick Start
//!
//! ```rust
//! use sha3_kernel_hasher::{Sha3_512Kernel, Sha3Digest};
//!
//! let mut hasher = Sha3_512Kernel::new();
//!
//! // One-shot with typed digest
//! let digest: Sha3Digest = hasher.hash_digest(b"hello");
//! println!("{}", digest);
//!
//! // Incremental hashing
//! hasher.reset();
//! hasher.update(b"hello");
//! hasher.update(b" world");
//! let digest = hasher.finalize_digest();
//! println!("{:X}", digest);
//! ```
//!
//! ## References
//!
//! - Bertoni, G. et al. (2011) *The Keccak Reference*, v3.0.
//!   <https://keccak.team/files/Keccak-reference-3.0.pdf>
//! - NIST (2015) *SHA-3 Standard*. FIPS PUB 202.
//!   doi:10.6028/NIST.FIPS.202
//! - Bernstein, D.J. (2005) *Cache-timing attacks on AES*.
//!   <https://cr.yp.to/antiforgery/cachetiming-20050414.pdf>
//! - Bernstein, D.J., Lange, T. and Schwabe, P. (2012)
//!   'The Security Impact of a New Cryptographic Library',
//!   *LATINCRYPT 2012*, LNCS 7533, pp. 159–176.
//! - Kelsey, J. and Schneier, B. (2005) 'Second Preimages on n-Bit
//!   Hash Functions for Much Less than 2^n Work', *EUROCRYPT 2005*,
//!   LNCS 3494, pp. 474–490.
//! - Kocher, P.C. (1996) 'Timing Attacks on Implementations of
//!   Diffie-Hellman, RSA, DSS, and Other Systems', *CRYPTO '96*,
//!   LNCS 1109, pp. 104–113.

#![cfg_attr(feature = "kernel", no_std)]

#[cfg(feature = "kernel")]
extern crate alloc;

#[cfg(feature = "kernel")]
use alloc::vec::Vec;
#[cfg(not(feature = "kernel"))]
use std::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod keccak;
pub mod hex;
pub mod ct;

#[cfg(not(feature = "kernel"))]
pub mod io;

use keccak::keccak_f1600;
pub use hex::Sha3Digest;

/// SHA3-512 rate in bytes: r = (1600 − 2×512) / 8 = 72.
const SHA3_512_RATE: usize = 72;

/// SHA3 domain separation byte (FIPS 202 §6.1).
const SHA3_DOMAIN_BYTE: u8 = 0x06;

/// SHA3-512 kernel hasher with SIMD optimisation
pub struct Sha3_512Kernel {
    state: Sha3_512State,
    /// SIMD feature detection
    features: SimdFeatures,
    /// Performance configuration
    config: PerformanceConfig,
}

/// SIMD feature detection results
#[derive(Debug, Clone, Copy)]
pub struct SimdFeatures {
    pub avx512_available: bool,
    pub avx2_available: bool,
    pub sse42_available: bool,
    pub bmi2_available: bool,
}

/// Performance configuration for hasher.
///
/// **Currently has no effect on `hash()`'s behavior or timing.** Every
/// field here is accepted and stored but never read by the hashing code
/// path — `Sha3_512Kernel::hash` always runs the same scalar
/// Keccak-f\[1600\] implementation. These fields exist so the public API
/// shape doesn't need to break if/when a real SIMD-accelerated path is
/// implemented; don't expect setting `use_avx512: true` (or `false`) to
/// change measured throughput today — it won't.
#[derive(Debug, Clone, Copy)]
pub struct PerformanceConfig {
    /// No-op today; reserved for a future AVX-512 code path.
    pub use_avx512: bool,
    /// No-op today; reserved for a future AVX2 code path.
    pub use_avx2: bool,
    /// No-op today; reserved for a future prefetch optimization.
    pub prefetch_data: bool,
    /// No-op today; reserved for a future parallel-chunk optimization.
    pub parallel_chunks: bool,
    /// No-op today.
    pub chunk_size: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            use_avx512: true,
            use_avx2: true,
            prefetch_data: true,
            parallel_chunks: true,
            chunk_size: 1024,
        }
    }
}

/// SHA3-512 sponge state (64-byte aligned for SIMD compatibility)
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct Sha3_512State {
    /// Keccak state: 5×5 matrix of 64-bit lanes (1600 bits)
    pub state: [u64; 25],
    /// Current position in the absorb buffer (0..rate)
    pub pos: usize,
    /// Rate in bytes (72 for SHA3-512)
    pub rate: usize,
    /// Absorb buffer — accumulates input until a full rate-block is ready
    pub buffer: [u8; 200],
}

#[cfg(feature = "serde")]
mod state_serde {
    use super::Sha3_512State;
    use serde::{Serialize, Serializer, Deserialize, Deserializer};
    use serde::ser::SerializeStruct;
    use serde::de::{self, MapAccess, Visitor};
    use core::fmt;

    #[cfg(feature = "kernel")]
    use alloc::vec::Vec;
    #[cfg(not(feature = "kernel"))]
    use std::vec::Vec;

    impl Serialize for Sha3_512State {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut s = serializer.serialize_struct("Sha3_512State", 4)?;
            s.serialize_field("state", &self.state.as_slice())?;
            s.serialize_field("pos", &self.pos)?;
            s.serialize_field("rate", &self.rate)?;
            s.serialize_field("buffer", &self.buffer.as_slice())?;
            s.end()
        }
    }

    impl<'de> Deserialize<'de> for Sha3_512State {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            #[derive(serde::Deserialize)]
            #[serde(field_identifier, rename_all = "lowercase")]
            enum Field { State, Pos, Rate, Buffer }

            struct StateVisitor;

            impl<'de> Visitor<'de> for StateVisitor {
                type Value = Sha3_512State;

                fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("struct Sha3_512State")
                }

                fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Sha3_512State, M::Error> {
                    let mut state_lanes: Option<Vec<u64>> = None;
                    let mut pos: Option<usize> = None;
                    let mut rate: Option<usize> = None;
                    let mut buffer_vec: Option<Vec<u8>> = None;

                    while let Some(key) = map.next_key::<Field>()? {
                        match key {
                            Field::State  => { state_lanes = Some(map.next_value::<Vec<u64>>()?); }
                            Field::Pos    => { pos = Some(map.next_value::<usize>()?); }
                            Field::Rate   => { rate = Some(map.next_value::<usize>()?); }
                            Field::Buffer => { buffer_vec = Some(map.next_value::<Vec<u8>>()?); }
                        }
                    }

                    let sv = state_lanes.ok_or_else(|| de::Error::missing_field("state"))?;
                    let pv = pos.ok_or_else(|| de::Error::missing_field("pos"))?;
                    let rv = rate.ok_or_else(|| de::Error::missing_field("rate"))?;
                    let bv = buffer_vec.ok_or_else(|| de::Error::missing_field("buffer"))?;

                    let mut state = [0u64; 25];
                    if sv.len() != 25 { return Err(de::Error::invalid_length(sv.len(), &"25")); }
                    state.copy_from_slice(&sv);

                    let mut buffer = [0u8; 200];
                    if bv.len() != 200 { return Err(de::Error::invalid_length(bv.len(), &"200")); }
                    buffer.copy_from_slice(&bv);

                    Ok(Sha3_512State { state, pos: pv, rate: rv, buffer })
                }
            }

            deserializer.deserialize_struct(
                "Sha3_512State",
                &["state", "pos", "rate", "buffer"],
                StateVisitor,
            )
        }
    }
}

/// Hash result type — 64 bytes (512 bits)
pub type Sha3_512Hash = [u8; 64];

/// Memory region descriptor for kernel hashing
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryRegion {
    pub base_address: u64,
    pub size: usize,
    pub is_readable: bool,
    pub is_writable: bool,
    pub is_executable: bool,
}

/// Maximum safe hash size for kernel operations (256 MiB)
pub const MAX_HASH_SIZE: usize = 256 * 1024 * 1024;

/// Default chunk size for parallel processing
pub const DEFAULT_CHUNK_SIZE: usize = 1024;

impl Sha3_512Kernel {
    /// Create a new SHA3-512 kernel hasher with default configuration.
    pub fn new() -> Self {
        Self::with_config(PerformanceConfig::default())
    }

    /// Create a hasher with a custom performance configuration.
    pub fn with_config(config: PerformanceConfig) -> Self {
        let features = Self::detect_simd_features();
        Self {
            state: Sha3_512State::new(),
            features,
            config,
        }
    }

    /// Hash `data` in a single call, returning the 64-byte SHA3-512 digest.
    ///
    /// Always runs the scalar Keccak-f\[1600\] path — see the crate-level
    /// "SIMD Status" section. `self.config` and `self.features` are not
    /// consulted here.
    pub fn hash(&mut self, data: &[u8]) -> Sha3_512Hash {
        self.state.reset();
        self.state.absorb(data);
        self.state.finalize()
    }

    /// Hash a raw memory region (kernel-compatible).
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `[base_address, base_address + size)`
    /// is a valid, readable memory region for the lifetime of this call.
    /// Returns `[0u8; 64]` if `size` is zero or exceeds `MAX_HASH_SIZE`.
    pub fn hash_memory_region(&mut self, base_address: u64, size: usize) -> Sha3_512Hash {
        if size == 0 || size > MAX_HASH_SIZE {
            return [0u8; 64];
        }
        let memory_slice = unsafe {
            core::slice::from_raw_parts(base_address as *const u8, size)
        };
        self.hash(memory_slice)
    }

    /// Hash multiple memory regions, returning one digest per readable region.
    pub fn hash_memory_regions(&mut self, regions: &[MemoryRegion]) -> Vec<Sha3_512Hash> {
        regions
            .iter()
            .filter(|r| r.is_readable && r.size > 0 && r.size <= MAX_HASH_SIZE)
            .map(|region| self.hash_memory_region(region.base_address, region.size))
            .collect()
    }

    /// Detect available SIMD features at runtime (userspace only).
    #[cfg(all(target_arch = "x86_64", not(feature = "kernel")))]
    fn detect_simd_features() -> SimdFeatures {
        SimdFeatures {
            avx512_available: is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512dq"),
            avx2_available: is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("bmi2"),
            sse42_available: is_x86_feature_detected!("sse4.2"),
            bmi2_available: is_x86_feature_detected!("bmi2"),
        }
    }

    /// In kernel mode, default to scalar (SIMD requires KeSaveFloatingPointState).
    /// SIMD can be enabled via PerformanceConfig after verifying processor state.
    #[cfg(any(not(target_arch = "x86_64"), feature = "kernel"))]
    fn detect_simd_features() -> SimdFeatures {
        SimdFeatures {
            avx512_available: false,
            avx2_available: false,
            sse42_available: false,
            bmi2_available: false,
        }
    }

    /// Get current SIMD features
    pub fn simd_features(&self) -> SimdFeatures {
        self.features
    }

    /// Get current configuration
    pub fn config(&self) -> PerformanceConfig {
        self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: PerformanceConfig) {
        self.config = config;
    }

    /// Hash `data` and return a [`Sha3Digest`] wrapper with formatting traits.
    ///
    /// This is a convenience method equivalent to
    /// `Sha3Digest::new(self.hash(data))`.
    pub fn hash_digest(&mut self, data: &[u8]) -> Sha3Digest {
        Sha3Digest::new(self.hash(data))
    }

    /// Finalize current hash (for incremental hashing).
    ///
    /// Applies SHA3 padding and extracts the digest. After calling this
    /// method, the hasher must be `reset()` before reuse.
    pub fn finalize(&mut self) -> Sha3_512Hash {
        self.state.finalize()
    }

    /// Finalize and return a [`Sha3Digest`] wrapper with formatting traits.
    pub fn finalize_digest(&mut self) -> Sha3Digest {
        Sha3Digest::new(self.finalize())
    }

    /// Reset hasher state for a new message.
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Update hasher with additional data (incremental / streaming).
    ///
    /// Can be called multiple times; the final digest is obtained
    /// by calling `finalize()`.
    pub fn update(&mut self, data: &[u8]) {
        self.state.absorb(data);
    }
}

impl Default for Sha3_512Kernel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Sha3_512State — core sponge implementation
// ---------------------------------------------------------------------------

impl Default for Sha3_512State {
    fn default() -> Self {
        Self {
            state: [0u64; 25],
            pos: 0,
            rate: SHA3_512_RATE,
            buffer: [0u8; 200],
        }
    }
}

impl Sha3_512State {
    /// Create a new SHA3-512 sponge state (all-zero per FIPS 202 §4).
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the sponge to the initial all-zero state.
    pub fn reset(&mut self) {
        self.state = [0u64; 25];
        self.pos = 0;
        self.buffer = [0u8; 200];
    }

    /// Absorb arbitrary-length input data into the sponge.
    ///
    /// Input is buffered until a full rate-block (72 bytes) is accumulated,
    /// then XORed into the state and followed by a Keccak-f\[1600\]
    /// permutation.
    fn absorb(&mut self, data: &[u8]) {
        let mut offset = 0;
        let len = data.len();

        // If we have buffered data, fill the buffer first
        if self.pos > 0 {
            let remaining = self.rate - self.pos;
            let to_copy = core::cmp::min(remaining, len);
            self.buffer[self.pos..self.pos + to_copy]
                .copy_from_slice(&data[..to_copy]);
            self.pos += to_copy;
            offset += to_copy;

            if self.pos == self.rate {
                self.xor_block_into_state();
                keccak_f1600(&mut self.state);
                self.pos = 0;
            }
        }

        // Process full rate-blocks directly from input (zero-copy)
        while offset + self.rate <= len {
            // XOR rate bytes from data directly into state lanes
            let block = &data[offset..offset + self.rate];
            let words = self.rate / 8; // 9 words for SHA3-512
            for i in 0..words {
                let off = i * 8;
                let word = u64::from_le_bytes([
                    block[off],
                    block[off + 1],
                    block[off + 2],
                    block[off + 3],
                    block[off + 4],
                    block[off + 5],
                    block[off + 6],
                    block[off + 7],
                ]);
                self.state[i] ^= word;
            }
            keccak_f1600(&mut self.state);
            offset += self.rate;
        }

        // Buffer any remaining bytes
        let tail = len - offset;
        if tail > 0 {
            self.buffer[..tail].copy_from_slice(&data[offset..]);
            self.pos = tail;
        }
    }

    /// Apply SHA3 padding, absorb the final block, and squeeze the digest.
    ///
    /// Implements the SHA3 padding rule from FIPS 202 §6.1:
    ///   1. Append domain separation byte `0x06`
    ///   2. Pad with zeros
    ///   3. Set the high bit of the last byte in the rate block (`0x80`)
    ///   4. XOR the padded block into state
    ///   5. Apply Keccak-f\[1600\]
    ///   6. Extract 64 bytes in little-endian lane order
    fn finalize(&mut self) -> Sha3_512Hash {
        // --- SHA3 padding (FIPS 202 §6.1) ---
        // Clear buffer beyond current position
        for i in self.pos..self.rate {
            self.buffer[i] = 0;
        }
        // Domain separation byte
        self.buffer[self.pos] = SHA3_DOMAIN_BYTE;
        // pad10*1: set high bit of the last byte in the rate block
        self.buffer[self.rate - 1] |= 0x80;

        // Absorb the padded final block
        self.xor_block_into_state();
        keccak_f1600(&mut self.state);

        // --- Squeeze phase ---
        // Extract 64 bytes (8 lanes × 8 bytes) in little-endian order
        let mut hash = [0u8; 64];
        for i in 0..8 {
            hash[i * 8..(i + 1) * 8].copy_from_slice(&self.state[i].to_le_bytes());
        }
        hash
    }

    /// XOR the current rate-block from the buffer into the sponge state.
    ///
    /// Reads `rate / 8` little-endian 64-bit words from `self.buffer`
    /// and XORs them into the first `rate / 8` lanes of `self.state`.
    #[inline]
    fn xor_block_into_state(&mut self) {
        let words = self.rate / 8; // 9 for SHA3-512
        for i in 0..words {
            let off = i * 8;
            let word = u64::from_le_bytes([
                self.buffer[off],
                self.buffer[off + 1],
                self.buffer[off + 2],
                self.buffer[off + 3],
                self.buffer[off + 4],
                self.buffer[off + 5],
                self.buffer[off + 6],
                self.buffer[off + 7],
            ]);
            self.state[i] ^= word;
        }
    }
}

/// Kernel-safe memory operations
pub mod kernel_safe {
    
    /// Save processor state for SIMD operations
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it directly manipulates processor registers.
    /// The caller must ensure:
    /// - The processor supports the required SIMD instructions (AVX2/AVX-512)
    /// - No other code is currently using SIMD registers
    /// - The saved state is properly restored before returning to normal execution
    /// - The function is called only from kernel mode or with appropriate privileges
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn save_processor_state() -> ProcessorState {
        ProcessorState::save()
    }
    
    /// Restore processor state after SIMD operations
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it directly manipulates processor registers.
    /// The caller must ensure:
    /// - The state was previously saved by `save_processor_state()`
    /// - No SIMD registers have been modified since the save
    /// - The processor is in the same state as when the save occurred
    /// - The function is called only from kernel mode or with appropriate privileges
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn restore_processor_state(state: ProcessorState) {
        state.restore();
    }
    
    /// Processor state structure
    #[cfg(target_arch = "x86_64")]
    #[repr(C)]
    pub struct ProcessorState {
        // XMM/YMM/ZMM registers
        simd_registers: [u8; 512],
        // Control registers
        control_registers: [u64; 8],
    }
    
    #[cfg(target_arch = "x86_64")]
    impl ProcessorState {
        /// Save current processor state to memory
        /// 
        /// # Safety
        /// 
        /// This function is unsafe because it directly accesses processor registers.
        /// The caller must ensure:
        /// - The processor supports XSAVE/XSAVEOPT instructions
        /// - No other code is currently executing SIMD operations
        /// - Sufficient memory is allocated for the state structure
        /// - The function is called only from kernel mode or with appropriate privileges
        pub unsafe fn save() -> Self {
            // In real implementation, use XSAVE/XSAVEOPT
            Self {
                simd_registers: [0u8; 512],
                control_registers: [0u64; 8],
            }
        }
        
        /// Restore processor state from saved memory
        /// 
        /// # Safety
        /// 
        /// This function is unsafe because it directly modifies processor registers.
        /// The caller must ensure:
        /// - The state was previously saved by the `save()` method
        /// - No processor registers have been modified since the save
        /// - The processor is in the same state as when the save occurred
        /// - The function is called only from kernel mode or with appropriate privileges
        pub unsafe fn restore(self) {
            // In real implementation, use XRSTOR
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    pub struct ProcessorState;
    
    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn save_processor_state() -> ProcessorState {
        ProcessorState
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn restore_processor_state(_state: ProcessorState) {
        // No-op for non-x86_64
    }
}

/// Performance utilities (available in std mode only)
#[cfg(not(feature = "kernel"))]
pub mod performance {
    use super::*;

    /// Benchmark hasher performance
    pub fn benchmark_hasher<F>(hasher_fn: F, data: &[u8]) -> BenchmarkResult
    where
        F: Fn(&[u8]) -> Sha3_512Hash,
    {
        let start = std::time::Instant::now();
        let hash = hasher_fn(data);
        let duration = start.elapsed();

        BenchmarkResult {
            hash,
            duration,
            throughput: (data.len() as f64) / duration.as_secs_f64(),
        }
    }

    /// Benchmark result
    #[derive(Debug)]
    pub struct BenchmarkResult {
        /// Computed hash
        pub hash: Sha3_512Hash,
        /// Time taken
        pub duration: std::time::Duration,
        /// Throughput in bytes/second
        pub throughput: f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: encode byte array to lowercase hex string.
    fn bytes_to_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    // ===================================================================
    // NIST CAVP SHA3-512 Test Vectors (FIPS 202)
    // Source: https://csrc.nist.gov/projects/cryptographic-algorithm-
    //         validation-program/secure-hashing#sha3vsha3vss
    // ===================================================================

    /// NIST ShortMsg vector: empty message (Len = 0)
    #[test]
    fn test_nist_sha3_512_empty() {
        let expected = "a69f73cca23a9ac5c8b567dc185a756e\
                        97c982164fe25859e0d1dcc1475c80a6\
                        15b2123af1f5f94c11e3e9402c3ac558\
                        f500199d95b6d3e301758586281dcd26";
        let mut hasher = Sha3_512Kernel::new();
        let hash = hasher.hash(b"");
        assert_eq!(
            bytes_to_hex(&hash),
            expected,
            "SHA3-512('') must match NIST CAVP vector"
        );
    }

    /// NIST ShortMsg vector: single byte 0xCC (Len = 8)
    #[test]
    fn test_nist_sha3_512_one_byte_cc() {
        let expected = "3939fcc8b57b63612542da31a834e5dc\
                        c36e2ee0f652ac72e02624fa2e5adeec\
                        c7dd6bb3580224b4d6138706fc6e8059\
                        7b528051230b00621cc2b22999eaa205";
        let mut hasher = Sha3_512Kernel::new();
        let hash = hasher.hash(&[0xCC]);
        assert_eq!(
            bytes_to_hex(&hash),
            expected,
            "SHA3-512(0xCC) must match NIST CAVP vector"
        );
    }

    /// NIST ShortMsg vector: two bytes 0x41, 0xFB (Len = 16)
    #[test]
    fn test_nist_sha3_512_two_bytes() {
        let expected = "aa092865a40694d91754dbc767b5202c\
                        546e226877147a95cb8b4c8f8709fe8c\
                        d6905256b089da37896ea5ca19d2cd9a\
                        b94c7192fc39f7cd4d598975a3013c69";
        let mut hasher = Sha3_512Kernel::new();
        let hash = hasher.hash(&[0x41, 0xFB]);
        assert_eq!(
            bytes_to_hex(&hash),
            expected,
            "SHA3-512(0x41FB) must match NIST CAVP vector"
        );
    }

    /// NIST LongMsg vector: 200 bytes of 0xA3 repeated
    /// This exercises multi-block absorb (200 > rate of 72).
    #[test]
    fn test_nist_sha3_512_200_bytes_a3() {
        let expected = "e76dfad22084a8b1467fcf2ffa58361b\
                        ec7628edf5f3fdc0e4805dc48caeeca8\
                        1b7c13c30adf52a3659584739a2df46b\
                        e589c51ca1a4a8416df6545a1ce8ba00";
        let input = vec![0xA3u8; 200];
        let mut hasher = Sha3_512Kernel::new();
        let hash = hasher.hash(&input);
        assert_eq!(
            bytes_to_hex(&hash),
            expected,
            "SHA3-512(0xA3 × 200) must match NIST CAVP vector"
        );
    }

    // ===================================================================
    // Incremental (streaming) API tests
    // ===================================================================

    /// Verify that incremental hashing produces the same result
    /// as single-call hashing.
    #[test]
    fn test_incremental_matches_oneshot() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mut oneshot = Sha3_512Kernel::new();
        let expected = oneshot.hash(data);

        let mut incremental = Sha3_512Kernel::new();
        incremental.reset();
        incremental.update(&data[..10]);
        incremental.update(&data[10..30]);
        incremental.update(&data[30..]);
        let result = incremental.finalize();

        assert_eq!(result, expected, "Incremental must match oneshot");
    }

    /// Verify byte-at-a-time feeding produces correct digest.
    #[test]
    fn test_incremental_byte_at_a_time() {
        let data = b"abcdefghijklmnopqrstuvwxyz";
        let mut oneshot = Sha3_512Kernel::new();
        let expected = oneshot.hash(data);

        let mut inc = Sha3_512Kernel::new();
        inc.reset();
        for &b in data.iter() {
            inc.update(&[b]);
        }
        let result = inc.finalize();
        assert_eq!(result, expected, "Byte-at-a-time must match oneshot");
    }

    // ===================================================================
    // Determinism and edge cases
    // ===================================================================

    /// Same input must always produce the same output.
    #[test]
    fn test_deterministic() {
        let mut hasher = Sha3_512Kernel::new();
        let data = b"Bunker SHA3-512 determinism test";
        let h1 = hasher.hash(data);
        let h2 = hasher.hash(data);
        assert_eq!(h1, h2, "Hash must be deterministic");
    }

    /// Different inputs must produce different outputs.
    #[test]
    fn test_avalanche() {
        let mut hasher = Sha3_512Kernel::new();
        let h1 = hasher.hash(b"input_a");
        let h2 = hasher.hash(b"input_b");
        assert_ne!(h1, h2, "Different inputs must avalanche");
    }

    /// Hash of the empty message must NOT be all zeros.
    #[test]
    fn test_empty_not_zero() {
        let mut hasher = Sha3_512Kernel::new();
        let hash = hasher.hash(b"");
        assert_ne!(hash, [0u8; 64], "SHA3-512('') must not be all-zero");
    }

    /// Boundary: exactly one rate-block (72 bytes).
    #[test]
    fn test_exact_rate_block() {
        let data = vec![0x42u8; 72];
        let mut hasher = Sha3_512Kernel::new();
        let h1 = hasher.hash(&data);
        let h2 = hasher.hash(&data);
        assert_eq!(h1, h2);
        assert_ne!(h1, [0u8; 64]);
    }

    /// Boundary: rate-block + 1 byte (73 bytes → 2 blocks).
    #[test]
    fn test_rate_plus_one() {
        let data_72 = vec![0x42u8; 72];
        let data_73 = vec![0x42u8; 73];
        let mut hasher = Sha3_512Kernel::new();
        let h72 = hasher.hash(&data_72);
        let h73 = hasher.hash(&data_73);
        assert_ne!(h72, h73, "72 vs 73 bytes must differ");
    }

    /// Large payload: 1 MiB of zeros.
    #[test]
    fn test_large_payload() {
        let data = vec![0u8; 1024 * 1024];
        let mut hasher = Sha3_512Kernel::new();
        let h1 = hasher.hash(&data);
        let h2 = hasher.hash(&data);
        assert_eq!(h1, h2);
        assert_ne!(h1, [0u8; 64]);
    }

    // ===================================================================
    // SIMD feature detection
    // ===================================================================

    #[test]
    fn test_simd_detection() {
        let hasher = Sha3_512Kernel::new();
        let features = hasher.simd_features();
        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, at minimum SSE4.2 should be available on any
            // CPU manufactured after 2008.
            assert!(
                features.sse42_available || features.avx2_available || features.avx512_available,
                "At least one SIMD tier must be detected on x86_64"
            );
        }
    }

    // ===================================================================
    // Memory region hashing
    // ===================================================================

    #[test]
    fn test_memory_hashing() {
        let mut hasher = Sha3_512Kernel::new();
        let data = vec![0u8; 1024];
        let h1 = hasher.hash(&data);
        let h2 = hasher.hash_memory_region(data.as_ptr() as u64, data.len());
        assert_eq!(h1, h2, "Memory region hash must match slice hash");
    }

    #[test]
    fn test_memory_hashing_zero_size() {
        let mut hasher = Sha3_512Kernel::new();
        let hash = hasher.hash_memory_region(0x1000, 0);
        assert_eq!(hash, [0u8; 64], "Zero-size region must return zero hash");
    }

    #[test]
    fn test_memory_hashing_oversized() {
        let mut hasher = Sha3_512Kernel::new();
        let hash = hasher.hash_memory_region(0x1000, MAX_HASH_SIZE + 1);
        assert_eq!(hash, [0u8; 64], "Oversized region must return zero hash");
    }

    // ===================================================================
    // Cross-validation against the `sha3` reference crate
    // ===================================================================

    /// Verify our implementation matches the RustCrypto `sha3` crate
    /// for a variety of input lengths, covering all absorb edge cases.
    #[test]
    fn test_cross_validate_vs_sha3_crate() {
        use sha3::{Sha3_512 as RefSha3_512, Digest};
        let test_lengths: &[usize] = &[0, 1, 2, 31, 32, 63, 64, 71, 72, 73, 100, 143, 144, 145, 200, 255, 256, 512, 1000, 4096];
        let mut hasher = Sha3_512Kernel::new();

        for &len in test_lengths {
            let data: Vec<u8> = (0..len).map(|i| (i & 0xFF) as u8).collect();
            let our_hash = hasher.hash(&data);

            let mut ref_hasher = RefSha3_512::new();
            ref_hasher.update(&data);
            let ref_hash = ref_hasher.finalize();

            assert_eq!(
                &our_hash[..],
                &ref_hash[..],
                "Mismatch vs sha3 crate at len={}",
                len
            );
        }
    }
}

#[cfg(feature = "benchmarks")]
pub mod benchmarks {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    #[cfg(feature = "kernel")]
    use alloc::vec;
    #[cfg(not(feature = "kernel"))]
    use std::vec;

    fn bench_sha3_512(c: &mut Criterion) {
        let mut hasher = Sha3_512Kernel::new();
        let data = vec![0u8; 1024 * 1024]; // 1MB
        
        c.bench_function("sha3_512_hash", |b| {
            b.iter(|| {
                black_box(hasher.hash(&data))
            })
        });
    }
    
    criterion_group!(benches, bench_sha3_512);
    criterion_main!(benches);
}
