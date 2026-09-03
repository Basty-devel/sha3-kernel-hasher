<# 
.SYNOPSIS
    Benchmark: sha3-kernel-hasher hash_file() vs Windows Get-FileHash (SHA-512)

.DESCRIPTION
    Generates a temporary file of configurable size, hashes it with both
    the Rust sha3-kernel-hasher (via a compiled helper binary) and the
    built-in PowerShell Get-FileHash cmdlet, then reports throughput in
    MiB/s for each.

    This script validates that the Sha3Reader's 8 KiB staging buffer
    achieves competitive throughput against the Windows CNG SHA-512
    implementation.

.PARAMETER SizeMB
    Size of the test file in MiB (default: 256).

.EXAMPLE
    .\benchmark_vs_windows.ps1
    .\benchmark_vs_windows.ps1 -SizeMB 1024

.NOTES
    Author : Sebastian Friedrich Nestler
    Requires: Rust toolchain (cargo), PowerShell 5.1+
#>

param(
    [int]$SizeMB = 256
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$CrateRoot   = Join-Path $ProjectRoot "sha3-kernel-hasher"
$TempFile    = Join-Path $env:TEMP "sha3_bench_testfile.bin"

Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  SHA3-Kernel-Hasher vs Windows Get-FileHash Benchmark"        -ForegroundColor Cyan
Write-Host "  Test file size: $SizeMB MiB"                                  -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Generate test file ───────────────────────────────────────
Write-Host "[1/4] Generating $SizeMB MiB test file..." -ForegroundColor Yellow
$SizeBytes = $SizeMB * 1024 * 1024
$ChunkSize = 1024 * 1024  # 1 MiB chunks
$Rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$Chunk = New-Object byte[] $ChunkSize

try {
    $Stream = [System.IO.File]::Create($TempFile)
    for ($i = 0; $i -lt $SizeMB; $i++) {
        $Rng.GetBytes($Chunk)
        $Stream.Write($Chunk, 0, $ChunkSize)
    }
    $Stream.Close()
    Write-Host "  Created: $TempFile ($SizeMB MiB)" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Failed to create test file: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ── Step 2: Build the Rust benchmark binary ──────────────────────────
Write-Host "[2/4] Building sha3-kernel-hasher (release)..." -ForegroundColor Yellow
Push-Location $CrateRoot
try {
    & cargo build --example large_file_integrity --release 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Cargo build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Build successful" -ForegroundColor Green
} finally {
    Pop-Location
}
Write-Host ""

# ── Step 3: Benchmark Windows Get-FileHash (SHA512) ─────────────────
Write-Host "[3/4] Benchmarking Windows Get-FileHash -Algorithm SHA512..." -ForegroundColor Yellow
$SwWin = [System.Diagnostics.Stopwatch]::StartNew()
$WinResult = Get-FileHash -Path $TempFile -Algorithm SHA512
$SwWin.Stop()

$WinSeconds    = $SwWin.Elapsed.TotalSeconds
$WinThroughput = $SizeMB / $WinSeconds

Write-Host "  Algorithm : SHA-512 (Windows CNG)"               -ForegroundColor White
Write-Host "  Hash      : $($WinResult.Hash.Substring(0,32))..." -ForegroundColor White
Write-Host "  Time      : $([math]::Round($WinSeconds, 3)) s"   -ForegroundColor White
Write-Host ("  Throughput: {0:N1} MiB/s" -f $WinThroughput)     -ForegroundColor White
Write-Host ""

# ── Step 4: Benchmark sha3-kernel-hasher hash_file() ─────────────────
Write-Host "[4/4] Benchmarking sha3-kernel-hasher hash_file() (SHA3-512)..." -ForegroundColor Yellow

# We use a small inline Rust program via cargo run --example
# But since the example hashes a synthetic image, we'll time a direct
# file hash by writing a quick one-liner Rust script approach.
# Instead, we'll measure by calling the example with a file path.

# For a fair comparison, we'll hash the temp file using a small Rust runner.
$RunnerSrc = @"
use sha3_kernel_hasher::io::hash_file;
use std::env;
use std::time::Instant;

fn main() {
    let path = env::args().nth(1).expect("Usage: hash_bench <file_path>");
    let t0 = Instant::now();
    let digest = hash_file(&path).expect("Failed to hash file");
    let elapsed = t0.elapsed();
    let size_bytes = std::fs::metadata(&path).unwrap().len();
    let throughput = (size_bytes as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);
    println!("DIGEST:{}", digest);
    println!("TIME:{:.6}", elapsed.as_secs_f64());
    println!("THROUGHPUT:{:.1}", throughput);
}
"@

$RunnerPath = Join-Path $CrateRoot "examples\hash_bench_runner.rs"
$RunnerSrc | Out-File -Encoding utf8 -FilePath $RunnerPath

Push-Location $CrateRoot
try {
    & cargo build --example hash_bench_runner --release 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Runner build failed" -ForegroundColor Red
        exit 1
    }

    $TargetDir = Join-Path $env:TEMP "kernel-driver-target"
    $RunnerExe = Join-Path $TargetDir "release\examples\hash_bench_runner.exe"
    if (-not (Test-Path $RunnerExe)) {
        # Fallback: check default target dir
        $RunnerExe = Join-Path $CrateRoot "target\release\examples\hash_bench_runner.exe"
    }

    $RustOutput = & $RunnerExe $TempFile 2>&1
    $DigestLine     = ($RustOutput | Where-Object { $_ -match "^DIGEST:" }) -replace "^DIGEST:", ""
    $TimeLine       = ($RustOutput | Where-Object { $_ -match "^TIME:" }) -replace "^TIME:", ""
    $ThroughputLine = ($RustOutput | Where-Object { $_ -match "^THROUGHPUT:" }) -replace "^THROUGHPUT:", ""

    Write-Host "  Algorithm : SHA3-512 (sha3-kernel-hasher)"            -ForegroundColor White
    Write-Host "  Hash      : $($DigestLine.Substring(0,32))..."        -ForegroundColor White
    Write-Host "  Time      : $TimeLine s"                              -ForegroundColor White
    Write-Host ("  Throughput: {0} MiB/s" -f $ThroughputLine)           -ForegroundColor White
} finally {
    Pop-Location
}
Write-Host ""

# ── Summary ──────────────────────────────────────────────────────────
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  COMPARISON SUMMARY ($SizeMB MiB file)"                       -ForegroundColor Cyan
Write-Host "-------------------------------------------------------------" -ForegroundColor Cyan
Write-Host ("  Windows SHA-512 (CNG)   : {0:N1} MiB/s" -f $WinThroughput)  -ForegroundColor White
Write-Host ("  SHA3-512 (Rust Keccak)  : {0} MiB/s" -f $ThroughputLine)    -ForegroundColor White

$Ratio = [double]$ThroughputLine / $WinThroughput
if ($Ratio -ge 1.0) {
    Write-Host ("  Rust is {0:N2}x FASTER" -f $Ratio) -ForegroundColor Green
} else {
    Write-Host ("  Windows CNG is {0:N2}x faster (expected: SHA-2 has HW accel)" -f (1.0 / $Ratio)) -ForegroundColor Yellow
    Write-Host "  Note: SHA3-512 != SHA-512. SHA-2 benefits from Intel SHA-NI." -ForegroundColor Yellow
    Write-Host "  SHA3 on pure scalar is expected to be slower than HW-accel SHA-2." -ForegroundColor Yellow
}
Write-Host "=============================================================" -ForegroundColor Cyan

# ── Cleanup ──────────────────────────────────────────────────────────
Remove-Item -Path $TempFile -Force -ErrorAction SilentlyContinue
Remove-Item -Path $RunnerPath -Force -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "Temporary files cleaned up." -ForegroundColor Gray
