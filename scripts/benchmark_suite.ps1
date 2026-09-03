<#
.SYNOPSIS
    SHA3-Kernel-Hasher Comprehensive Benchmark Suite

.DESCRIPTION
    Runs a multi-tier benchmark comparing the Rust SHA3-512 implementation
    against Windows CNG SHA-512 (Get-FileHash) across a range of file sizes.
    Outputs a formatted table suitable for GitHub README embedding.

    Tiers:
      1. Micro   (1 KiB - 64 KiB)   -- L1/L2 cache resident payloads
      2. Medium  (1 MiB - 64 MiB)    -- L3 cache and memory-bound payloads
      3. Large   (256 MiB - 1 GiB)   -- Sustained throughput, I/O bound

    Results are printed as a Markdown table for direct copy-paste into README.

.PARAMETER MaxSizeMB
    Upper bound for the large tier in MiB (default: 256).
    Set to 1024 for the full 1 GiB test if disk space permits.

.PARAMETER Iterations
    Number of iterations per size for micro/medium tiers (default: 3).

.EXAMPLE
    .\benchmark_suite.ps1
    .\benchmark_suite.ps1 -MaxSizeMB 1024 -Iterations 5

.NOTES
    Author  : Sebastian Friedrich Nestler
    Requires: Rust toolchain (cargo), PowerShell 5.1+
#>

param(
    [int]$MaxSizeMB  = 256,
    [int]$Iterations = 3
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$CrateRoot   = Join-Path $ProjectRoot "sha3-kernel-hasher"
$TempDir     = Join-Path $env:TEMP "sha3_benchmark"

# ── Ensure temp directory ────────────────────────────────────────────
if (-not (Test-Path $TempDir)) { New-Item -ItemType Directory -Path $TempDir | Out-Null }

# ── Build the runner binary ──────────────────────────────────────────
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  SHA3-Kernel-Hasher  --  Comprehensive Benchmark Suite"          -ForegroundColor Cyan
Write-Host "  Date     : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"           -ForegroundColor Cyan
Write-Host "  Host     : $env:COMPUTERNAME"                                    -ForegroundColor Cyan
Write-Host "  MaxSize  : $MaxSizeMB MiB"                                       -ForegroundColor Cyan
Write-Host "  Iterations: $Iterations per size (micro/medium)"                 -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Create a minimal Rust runner that hashes a file and reports timing
$RunnerSrc = @"
use sha3_kernel_hasher::io::hash_file;
use std::env;
use std::time::Instant;

fn main() {
    let path = env::args().nth(1).expect("Usage: bench_runner <file>");
    let t0 = Instant::now();
    let digest = hash_file(&path).expect("hash_file failed");
    let elapsed = t0.elapsed();
    let size = std::fs::metadata(&path).unwrap().len();
    let mbps = (size as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);
    println!("HASH:{}", digest);
    println!("BYTES:{}", size);
    println!("SECS:{:.9}", elapsed.as_secs_f64());
    println!("MBPS:{:.2}", mbps);
}
"@

$RunnerPath = Join-Path $CrateRoot "examples\bench_runner.rs"
$RunnerSrc | Out-File -Encoding utf8 -FilePath $RunnerPath -Force

Write-Host "[BUILD] Compiling bench_runner (release)..." -ForegroundColor Yellow
Push-Location $CrateRoot
try {
    & cargo build --example bench_runner --release 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: cargo build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "  OK" -ForegroundColor Green
} finally {
    Pop-Location
}
Write-Host ""

# Locate the built binary
$RunnerExe = Join-Path $env:TEMP "kernel-driver-target\release\examples\bench_runner.exe"
if (-not (Test-Path $RunnerExe)) {
    $RunnerExe = Join-Path $CrateRoot "target\release\examples\bench_runner.exe"
}
if (-not (Test-Path $RunnerExe)) {
    Write-Host "ERROR: Cannot locate bench_runner.exe" -ForegroundColor Red
    exit 1
}

# ── Helper: generate random file ─────────────────────────────────────
function New-RandomFile {
    param([string]$Path, [long]$SizeBytes)
    $Rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $ChunkSize = [math]::Min($SizeBytes, 1048576)  # 1 MiB chunks
    $Chunk = New-Object byte[] $ChunkSize
    $Stream = [System.IO.File]::Create($Path)
    $Remaining = $SizeBytes
    while ($Remaining -gt 0) {
        $n = [math]::Min($Remaining, $ChunkSize)
        if ($n -lt $ChunkSize) { $Chunk = New-Object byte[] $n }
        $Rng.GetBytes($Chunk)
        $Stream.Write($Chunk, 0, $Chunk.Length)
        $Remaining -= $n
    }
    $Stream.Close()
}

# ── Helper: format size ──────────────────────────────────────────────
function Format-Size {
    param([long]$Bytes)
    if ($Bytes -ge 1073741824) { return "{0:N0} GiB" -f ($Bytes / 1073741824) }
    if ($Bytes -ge 1048576)    { return "{0:N0} MiB" -f ($Bytes / 1048576) }
    if ($Bytes -ge 1024)       { return "{0:N0} KiB" -f ($Bytes / 1024) }
    return "$Bytes B"
}

# ── Define benchmark sizes ───────────────────────────────────────────
$Sizes = @(
    # Micro tier (cache-resident)
    @{ Label = "1 KiB";    Bytes = 1024 },
    @{ Label = "4 KiB";    Bytes = 4096 },
    @{ Label = "16 KiB";   Bytes = 16384 },
    @{ Label = "64 KiB";   Bytes = 65536 },
    # Medium tier (memory-bound)
    @{ Label = "1 MiB";    Bytes = 1048576 },
    @{ Label = "16 MiB";   Bytes = 16777216 },
    @{ Label = "64 MiB";   Bytes = 67108864 }
)

# Large tier (sustained throughput)
if ($MaxSizeMB -ge 256) {
    $Sizes += @{ Label = "256 MiB"; Bytes = 268435456 }
}
if ($MaxSizeMB -ge 512) {
    $Sizes += @{ Label = "512 MiB"; Bytes = 536870912 }
}
if ($MaxSizeMB -ge 1024) {
    $Sizes += @{ Label = "1 GiB";   Bytes = 1073741824 }
}

# ── Run benchmarks ───────────────────────────────────────────────────
$Results = @()

foreach ($size in $Sizes) {
    $Label     = $size.Label
    $SizeBytes = $size.Bytes
    $FilePath  = Join-Path $TempDir "bench_$($SizeBytes).bin"
    $Iters     = if ($SizeBytes -le 67108864) { $Iterations } else { 1 }

    Write-Host "[BENCH] $Label ($Iters iteration(s))..." -ForegroundColor Yellow -NoNewline

    # Generate file
    New-RandomFile -Path $FilePath -SizeBytes $SizeBytes

    # ── SHA3-512 (Rust) ──────────────────────────────────────────
    $RustTimes = @()
    for ($i = 0; $i -lt $Iters; $i++) {
        $Output = & $RunnerExe $FilePath 2>&1
        $SecsLine = ($Output | Where-Object { $_ -match "^SECS:" }) -replace "^SECS:", ""
        $RustTimes += [double]$SecsLine
    }
    $RustAvg = ($RustTimes | Measure-Object -Average).Average
    $RustMBps = ($SizeBytes / $RustAvg) / 1048576

    # ── SHA-512 (Windows CNG) ───────────────────────────────────
    $WinTimes = @()
    for ($i = 0; $i -lt $Iters; $i++) {
        $Sw = [System.Diagnostics.Stopwatch]::StartNew()
        Get-FileHash -Path $FilePath -Algorithm SHA512 | Out-Null
        $Sw.Stop()
        $WinTimes += $Sw.Elapsed.TotalSeconds
    }
    $WinAvg = ($WinTimes | Measure-Object -Average).Average
    $WinMBps = ($SizeBytes / $WinAvg) / 1048576

    # ── Ratio ────────────────────────────────────────────────────
    $Ratio = $RustMBps / $WinMBps

    $Results += [PSCustomObject]@{
        Size       = $Label
        SHA3_MBps  = [math]::Round($RustMBps, 1)
        SHA512_MBps = [math]::Round($WinMBps, 1)
        Ratio      = [math]::Round($Ratio, 2)
    }

    # Clean up file immediately to conserve disk space
    Remove-Item -Path $FilePath -Force -ErrorAction SilentlyContinue

    Write-Host " SHA3: $([math]::Round($RustMBps,1)) MiB/s | SHA-512: $([math]::Round($WinMBps,1)) MiB/s | Ratio: $([math]::Round($Ratio,2))x" -ForegroundColor Green
}

# ── Output Markdown table ────────────────────────────────────────────
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  RESULTS  --  Markdown Table (copy to README)"                    -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "| Input Size | SHA3-512 (MiB/s) | SHA-512 CNG (MiB/s) | Ratio |"
Write-Host "|------------|------------------|----------------------|-------|"
foreach ($r in $Results) {
    $RatioStr = if ($r.Ratio -ge 1.0) { "{0:N2}x faster" -f $r.Ratio } else { "{0:N2}x slower" -f $r.Ratio }
    Write-Host ("| {0,-10} | {1,16:N1} | {2,20:N1} | {3} |" -f $r.Size, $r.SHA3_MBps, $r.SHA512_MBps, $RatioStr)
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  NOTES"                                                           -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  - SHA3-512 uses Keccak-f[1600] (sponge), SHA-512 uses SHA-2 (Merkle-Damgard)"
Write-Host "  - Windows CNG SHA-512 benefits from Intel SHA-NI hardware acceleration"
Write-Host "  - SHA3-512 scalar implementation is expected to be slower than HW-accel SHA-2"
Write-Host "  - Future AVX2/AVX-512 Keccak intrinsics will close the throughput gap"
Write-Host "  - The security advantage of SHA3 is post-quantum sponge construction"
Write-Host ""

# ── Cleanup ──────────────────────────────────────────────────────────
Remove-Item -Path $RunnerPath -Force -ErrorAction SilentlyContinue
Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "Temporary files cleaned up." -ForegroundColor Gray
Write-Host ""
