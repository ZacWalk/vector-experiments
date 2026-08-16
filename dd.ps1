#!/usr/bin/env pwsh
#Requires -Version 5.1
<#
.SYNOPSIS
    Developer driver for vector-experiments: build, run and clean.

.DESCRIPTION
    Uses CMake with the platform's default generator, so it works out of the box:
      * Windows : the latest Visual Studio generator (finds MSVC automatically,
                  no developer prompt required).
      * Linux   : Unix Makefiles with GCC or Clang.
      * macOS   : Unix Makefiles / Xcode with Apple Clang (arm64).

    Commands:
      run      Configure + build (Release) and run the benchmark. (default)
      build    Configure + build only.
      clean    Delete the build directory.
      rebuild  Clean, then build.
      help     Show this help.

.PARAMETER Command
    One of run / build / clean / rebuild / help.

.PARAMETER Rest
    Extra arguments forwarded to the benchmark executable. The first one is the
    measurement budget in milliseconds per row, e.g. `./dd.ps1 run 50` for a
    quick smoke run.

.PARAMETER BuildDir
    Build directory (default: build).

.PARAMETER Config
    CMake build configuration (default: Release).

.EXAMPLE
    ./dd.ps1 run
.EXAMPLE
    ./dd.ps1 run 50
.EXAMPLE
    ./dd.ps1 rebuild
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('run', 'build', 'clean', 'rebuild', 'help')]
    [string]$Command = 'run',

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @(),

    [string]$BuildDir = 'build',

    [ValidateSet('Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Config = 'Release'
)

$ErrorActionPreference = 'Stop'
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

function Get-CMakeExe {
    $cmake = Find-CMake
    if (-not $cmake) {
        throw "Could not find 'cmake'. Install CMake, or Visual Studio with the " +
              "'C++ CMake tools for Windows' component, and try again."
    }
    Write-Host "Using cmake: $cmake" -ForegroundColor DarkGray
    return $cmake
}

function Invoke-Clean {
    if (Test-Path $BuildDir) {
        Write-Host "Cleaning $BuildDir" -ForegroundColor Cyan
        Remove-Item -Recurse -Force $BuildDir
    }
    else {
        Write-Host "Nothing to clean ($BuildDir does not exist)" -ForegroundColor DarkGray
    }
}

function Invoke-Build {
    $cmake = Get-CMakeExe

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

    Write-Host "Configuring ($Config)..." -ForegroundColor Cyan
    & $cmake -S . -B $BuildDir @genArgs "-DCMAKE_BUILD_TYPE=$Config"
    if ($LASTEXITCODE -ne 0) {
        # Most commonly a stale build dir created with a different generator.
        Write-Host "Configure failed; wiping '$BuildDir' and retrying clean..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue
        & $cmake -S . -B $BuildDir @genArgs "-DCMAKE_BUILD_TYPE=$Config"
        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }
    }

    Write-Host "Building ($Config)..." -ForegroundColor Cyan
    & $cmake --build $BuildDir --config $Config --parallel
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed ($LASTEXITCODE)" }
}

function Invoke-Run {
    Invoke-Build

    $exe = Get-ChildItem -Path $BuildDir -Recurse -File |
        Where-Object { $_.Name -in @('vector-experiments', 'vector-experiments.exe') } |
        Select-Object -First 1
    if (-not $exe) { throw "Could not find the built executable under $BuildDir" }

    Write-Host "Running $($exe.FullName)" -ForegroundColor Cyan
    if ($Rest.Count -gt 0) { & $exe.FullName @Rest } else { & $exe.FullName }
    return $LASTEXITCODE
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    switch ($Command) {
        'run' { exit (Invoke-Run) }
        'build' { Invoke-Build; exit 0 }
        'clean' { Invoke-Clean; exit 0 }
        'rebuild' { Invoke-Clean; Invoke-Build; exit 0 }
        'help' { Get-Help $MyInvocation.MyCommand.Path -Detailed; exit 0 }
    }
}
finally {
    Pop-Location
}
