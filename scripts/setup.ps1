<#
.SYNOPSIS
    One-shot local setup for NeuralForge-X on Windows.
.DESCRIPTION
    Verifies the Rust toolchain, creates/uses a venv, installs Python dev deps,
    and builds the extension (via dev_build.ps1 to sidestep blocked launchers).
#>
[CmdletBinding()]
param([string]$VenvPath = "$PSScriptRoot\..\venv")

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host "==> Checking Rust toolchain..." -ForegroundColor Cyan
$cargo = "$env:USERPROFILE\.cargo\bin\cargo.exe"
if (-not (Test-Path $cargo)) {
    throw "cargo not found. Install Rust from https://rustup.rs (use stable-msvc)."
}
& $cargo --version

if (-not (Test-Path (Join-Path $VenvPath 'Scripts\python.exe'))) {
    Write-Host "==> Creating venv at $VenvPath..." -ForegroundColor Cyan
    python -m venv $VenvPath
}
$py = Join-Path $VenvPath 'Scripts\python.exe'

Write-Host "==> Installing Python dev dependencies..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip
& $py -m pip install "numpy>=1.26" pandas pydantic pytest hypothesis ruff mypy "maturin>=1.7,<2.0"

Write-Host "==> Building extension..." -ForegroundColor Cyan
& "$PSScriptRoot\dev_build.ps1" -VenvPath $VenvPath

Write-Host "==> Setup complete. Activate with: . $VenvPath\Scripts\Activate.ps1" -ForegroundColor Green
