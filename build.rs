//! Build script for SHA3 Kernel Hasher
//!
//! Detects WDK for kernel-mode and emits SIMD target feature hints.

use std::env;
use std::path::PathBuf;

const WDK_ROOTS: &[&str] = &[
    r"C:\Program Files (x86)\Windows Kits\10",
    r"C:\Program Files\Windows Kits\10",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target = env::var("TARGET").unwrap_or_default();
    let kernel_mode = env::var("CARGO_FEATURE_KERNEL").is_ok();

    // SIMD feature detection hints
    if target.contains("x86_64") {
        println!("cargo:rustc-cfg=target_has_avx2");
        // AVX-512 requires runtime detection; we only hint availability
        println!("cargo:rustc-cfg=target_may_have_avx512");
    }

    if !target.contains("windows") {
        return;
    }

    if let Some((root, version)) = detect_wdk() {
        println!("cargo:rustc-cfg=wdk_available");
        println!("cargo:WDK_VERSION={}", version);

        if kernel_mode {
            let arch = if target.contains("x86_64") { "x64" } else { "x86" };
            let km_lib = root.join("Lib").join(&version).join("km").join(arch);
            if km_lib.exists() {
                println!("cargo:rustc-link-search=native={}", km_lib.display());
            }
        }
    } else if kernel_mode {
        println!("cargo:warning=WDK not found — kernel-mode sha3-kernel-hasher will use stubs");
    }
}

fn detect_wdk() -> Option<(PathBuf, String)> {
    for root_str in WDK_ROOTS {
        let root = PathBuf::from(root_str);
        let lib_base = root.join("Lib");
        if !lib_base.exists() {
            continue;
        }
        let mut versions: Vec<String> = std::fs::read_dir(&lib_base)
            .ok()?
            .filter_map(|e| {
                let name = e.ok()?.file_name().to_string_lossy().to_string();
                if name.starts_with("10.") { Some(name) } else { None }
            })
            .collect();
        versions.sort();
        if let Some(version) = versions.pop() {
            let lib_dir = lib_base.join(&version);
            if lib_dir.exists() {
                return Some((root, version));
            }
        }
    }
    None
}
