<#
.SYNOPSIS
    Capture a GPU profile of the NeuralForge-X CUDA engine with NVIDIA Nsight.

.DESCRIPTION
    Runs Nsight Systems (`nsys`) to record a CUDA timeline of a GPU workload
    (default: the four-way GPU benchmark) and emits a stats summary; optionally
    runs Nsight Compute (`ncu`) for per-kernel metrics. Executable paths are
    auto-detected from the standard install location.

    Blackwell (sm_120) note: kernel-level capture needs a recent Nsight. Older
    `nsys`/`ncu` may record the CUDA API + memcpy timeline but report no GPU
    kernel rows on sm_120 — the same toolchain-freshness gap documented for the
    CUDA engine itself. The CUDA-API/transfer timeline is still informative
    (it shows the H2D/D2H copies that make single-query top-k transfer-bound).

.PARAMETER Target
    Python script to profile. Default: profiling/gpu_workload.py (a steady loop
    over the GPU kernels). Point it at cuda_engine/benchmarks/bench_gpu.py for the
    full four-way benchmark instead.

.PARAMETER Ncu
    Also run Nsight Compute for per-kernel metrics (slow; may need elevated perms).

.EXAMPLE
    pwsh profiling/scripts/gpu_nsight.ps1
    pwsh profiling/scripts/gpu_nsight.ps1 -Ncu
#>
[CmdletBinding()]
param(
    [string]$Target,
    [switch]$Ncu
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$venvPy = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) { throw "venv python not found at $venvPy" }
if (-not $Target) { $Target = Join-Path $root 'profiling\gpu_workload.py' }

$outDir = Join-Path $root 'profiling\out'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Pick the newest installed nsys.exe / ncu.exe.
function Find-Newest([string]$leaf) {
    Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Recurse -Filter $leaf -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}

$nsys = Find-Newest 'nsys.exe'
if (-not $nsys) { throw "nsys.exe not found. Install NVIDIA Nsight Systems." }
Write-Host "==> Using $nsys" -ForegroundColor Cyan
& $nsys --version | Select-Object -First 1

$rep = Join-Path $outDir 'gpu_timeline'
Write-Host "==> nsys profile -> $rep.nsys-rep" -ForegroundColor Cyan
& $nsys profile --force-overwrite true -o $rep --trace=cuda,nvtx --sample=none $venvPy $Target

$stats = Join-Path $outDir 'gpu_stats.txt'
Write-Host "==> nsys stats -> $stats" -ForegroundColor Cyan
& $nsys stats --report cuda_api_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum `
    --format column "$rep.nsys-rep" | Tee-Object -FilePath $stats

if ($Ncu) {
    $ncu = Find-Newest 'ncu.exe'
    if (-not $ncu) { Write-Warning "ncu.exe not found; skipping Nsight Compute." }
    else {
        Write-Host "==> ncu (per-kernel metrics) -> $outDir\gpu_kernels.ncu-rep" -ForegroundColor Cyan
        & $ncu --set basic --launch-count 20 --force-overwrite `
            -o (Join-Path $outDir 'gpu_kernels') $venvPy $Target
    }
}

Write-Host "==> Done. Artifacts in $outDir (git-ignored)." -ForegroundColor Green
