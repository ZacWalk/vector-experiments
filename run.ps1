#!/usr/bin/env pwsh
#Requires -Version 5.1
<#
.SYNOPSIS
    Configure (Release), build and run the vector-experiments benchmarks.

.DESCRIPTION
    Uses CMake with the platform's default generator, so it works out of the box:
      * Windows : the latest Visual Studio generator (finds MSVC automatically,
                  no developer prompt required).
      * Linux   : Unix Makefiles with GCC or Clang.
      * macOS   : Unix Makefiles / Xcode with Apple Clang (arm64).

.PARAMETER Iterations
    Optional iteration count passed straight to the benchmark (handy for a quick
    smoke run, e.g. -Iterations 5000000). Omit for the full default run.

.PARAMETER BuildDir
    Build directory (default: build).

.PARAMETER Clean
    Delete the build directory before configuring.

.EXAMPLE
    ./run.ps1
.EXAMPLE
    ./run.ps1 -Iterations 5000000 -Clean
#>
[CmdletBinding()]
param(
    [long]$Iterations = 0,
    [string]$BuildDir = "build",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$onWindows = ($env:OS -eq 'Windows_NT')

function Find-CMake {
    # 1) Already on PATH?
    $cmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    if (-not $onWindows) { return $null }

    # 2) Bundled with any Visual Studio install (located via vswhere).
    $candidates = @()
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        $installs = & $vswhere -all -prerelease -products * -property installationPath 2>$null
        foreach ($vs in $installs) {
            $candidates += Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
        }
    }
    # 3) Standalone CMake install.
    $candidates += (Join-Path $env:ProgramFiles 'CMake\bin\cmake.exe')
    if (${env:ProgramFiles(x86)}) {
        $candidates += (Join-Path ${env:ProgramFiles(x86)} 'CMake\bin\cmake.exe')
    }

    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return (Resolve-Path $c).Path }
    }
    return $null
}

function Get-VSGenerator {
    param([string]$CMakeExe)
    # Returns the default "Visual Studio NN YYYY" generator name, or $null.
    try { $help = & $CMakeExe --help 2>$null } catch { return $null }
    $vs = $help | Where-Object { $_ -match '^\*?\s*Visual Studio \d+ \d{4}' }
    if (-not $vs) { return $null }
    $line = $vs | Where-Object { $_ -match '^\*' } | Select-Object -First 1
    if (-not $line) { $line = $vs | Select-Object -First 1 }
    return ((($line -replace '^\*', '').Trim()) -replace '\s*=.*$', '').Trim()
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    $cmake = Find-CMake
    if (-not $cmake) {
        throw "Could not find 'cmake'. Install CMake, or Visual Studio with the " +
              "'C++ CMake tools for Windows' component, and try again."
    }
    Write-Host "Using cmake: $cmake" -ForegroundColor DarkGray

    # On Windows, prefer the self-contained Visual Studio generator unless we are
    # already inside a Developer Command Prompt. Ninja/NMake builds with MSVC need
    # the vcvars environment (INCLUDE/LIB), which a plain shell does not have; the
    # Visual Studio generator drives MSBuild, which sets that up by itself.
    $genArgs = @()
    if ($onWindows -and -not $env:VCINSTALLDIR) {
        $vsGen = Get-VSGenerator $cmake
        if ($vsGen) {
            $genArgs = @('-G', $vsGen)
            Write-Host "Generator: $vsGen" -ForegroundColor DarkGray
        }
    }

    if ($Clean -and (Test-Path $BuildDir)) {
        Write-Host "Cleaning $BuildDir" -ForegroundColor Cyan
        Remove-Item -Recurse -Force $BuildDir
    }

    Write-Host "Configuring (Release)..." -ForegroundColor Cyan
    & $cmake -S . -B $BuildDir @genArgs -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) {
        # Most commonly a stale build dir created with a different generator.
        Write-Host "Configure failed; wiping '$BuildDir' and retrying clean..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue
        & $cmake -S . -B $BuildDir @genArgs -DCMAKE_BUILD_TYPE=Release
        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }
    }

    Write-Host "Building (Release)..." -ForegroundColor Cyan
    & $cmake --build $BuildDir --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed ($LASTEXITCODE)" }

    $exe = Get-ChildItem -Path $BuildDir -Recurse -File |
        Where-Object { $_.Name -in @("vector-experiments", "vector-experiments.exe") } |
        Select-Object -First 1
    if (-not $exe) { throw "Could not find the built executable under $BuildDir" }

    Write-Host "Running $($exe.FullName)" -ForegroundColor Cyan
    if ($Iterations -gt 0) {
        & $exe.FullName $Iterations
    }
    else {
        & $exe.FullName
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
