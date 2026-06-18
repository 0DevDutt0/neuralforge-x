<#
.SYNOPSIS
    Build the NeuralForge-X extension locally without the maturin launcher.

.DESCRIPTION
    Some Windows 11 machines block the prebuilt maturin.exe under an Application
    Control / Smart App Control policy. Locally compiled cargo output is not
    affected, so this script:
      1. builds the PyO3 cdylib with cargo (release),
      2. copies target/release/neuralforge_native.dll to the package as
         neuralforge/_native.pyd,
      3. writes an editable .pth into the active venv so `import neuralforge` works.

    Run from the repo root with the venv active (or pass -VenvPath).
#>
[CmdletBinding()]
param(
    [string]$VenvPath = "$PSScriptRoot\..\venv",
    [switch]$Debug
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..").Path
$profile = if ($Debug) { 'debug' } else { 'release' }

$cargo = "$env:USERPROFILE\.cargo\bin\cargo.exe"
if (-not (Test-Path $cargo)) { $cargo = 'cargo' }   # fall back to PATH

$venvPy = Join-Path $VenvPath 'Scripts\python.exe'
if (-not (Test-Path $venvPy)) { throw "venv python not found at $venvPy" }

Write-Host "==> Building neuralforge cdylib ($profile)..." -ForegroundColor Cyan
$env:PYO3_PYTHON = $venvPy
$buildArgs = @('build', '-p', 'neuralforge')
if (-not $Debug) { $buildArgs += '--release' }
& $cargo @buildArgs
if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" }

$src = Join-Path $root "target\$profile\neuralforge_native.dll"
$dst = Join-Path $root 'python_sdk\python\neuralforge\_native.pyd'
Copy-Item $src $dst -Force
Write-Host "==> Installed $dst" -ForegroundColor Green

$site = Join-Path $VenvPath 'Lib\site-packages'
$pth  = Join-Path $site 'neuralforge_dev.pth'
Set-Content -Path $pth -Value (Join-Path $root 'python_sdk\python') -Encoding ascii
Write-Host "==> Wrote editable path: $pth" -ForegroundColor Green

& $venvPy -c "import neuralforge as nf; print('neuralforge', nf.__version__, 'OK')"
