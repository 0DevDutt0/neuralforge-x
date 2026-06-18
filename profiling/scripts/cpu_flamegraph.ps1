<#
.SYNOPSIS
    Capture a CPU flamegraph of a NeuralForge-X kernel under sustained load.

.DESCRIPTION
    Builds the `neuralforge_profile` target with debug line tables (the
    `profiling` cargo profile) and runs `cargo flamegraph` against it, sampling
    the real shipped kernels.

    Windows note: `cargo flamegraph` samples via ETW (the `blondie` backend),
    which requires an **elevated** shell ("Run as administrator"). From a normal
    shell it fails with `NotAnAdmin`. Linux/macOS use perf/dtrace and need no
    elevation. The criterion-based report (profiling/CPU_PROFILE.md) needs no
    special privileges and is the portable analysis path.

.PARAMETER Workload
    Which kernel to profile: dot | batch | topk | all. Default: topk.

.PARAMETER Seconds
    Sustained-load duration. Default: 15.

.PARAMETER Corpus
    Corpus size for batch/topk. Default: 100000.

.EXAMPLE
    pwsh profiling/scripts/cpu_flamegraph.ps1 -Workload topk -Seconds 20
#>
[CmdletBinding()]
param(
    [ValidateSet("dot", "batch", "topk", "all")]
    [string]$Workload = "topk",
    [double]$Seconds = 15,
    [int]$Corpus = 100000
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$cargo = "$env:USERPROFILE\.cargo\bin\cargo.exe"
if (-not (Test-Path $cargo)) { $cargo = 'cargo' }

$outDir = Join-Path $root 'profiling\out'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$out = Join-Path $outDir "flamegraph_$Workload.svg"

# Warn early if not elevated (ETW capture will otherwise fail).
$isAdmin = ([Security.Principal.WindowsPrincipal] `
        [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltinRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "Not elevated. On Windows, cargo flamegraph (ETW/blondie) needs an admin shell."
    Write-Warning "Re-run this script from a terminal started with 'Run as administrator'."
}

if (-not (Get-Command cargo-flamegraph -ErrorAction SilentlyContinue)) {
    Write-Host "Installing cargo-flamegraph (one-time, compiled locally)..." -ForegroundColor Cyan
    & $cargo install flamegraph
}

Write-Host "==> Profiling '$Workload' for ${Seconds}s -> $out" -ForegroundColor Cyan
& $cargo flamegraph --profile profiling --bin neuralforge_profile --output $out `
    -- $Workload --seconds $Seconds --corpus $Corpus

if (Test-Path $out) {
    Write-Host "==> Wrote $out" -ForegroundColor Green
} else {
    Write-Warning "No flamegraph produced (see errors above; most often the admin/ETW requirement)."
}
