[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [switch]$Offline
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$runtimeConfig = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $projectRoot 'configs/runtime.json') | ConvertFrom-Json
$localConfigPath = Join-Path $projectRoot 'configs/runtime.local.json'
if (-not $RuntimeRoot) {
    if (Test-Path -LiteralPath $localConfigPath) {
        $RuntimeRoot = (Get-Content -Raw -Encoding UTF8 -LiteralPath $localConfigPath | ConvertFrom-Json).runtime_root
    } else {
        $RuntimeRoot = $runtimeConfig.default_runtime_root
    }
}
if (-not [IO.Path]::IsPathRooted($RuntimeRoot)) { throw 'RuntimeRoot must be an absolute path.' }
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot).TrimEnd('\', '/')
if ($RuntimeRoot -eq $projectRoot -or $RuntimeRoot.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'RuntimeRoot must be outside the project directory.'
}
$installTarget = Join-Path $RuntimeRoot 'miniconda3'
$environmentPrefix = Join-Path $RuntimeRoot ('envs\' + $runtimeConfig.environment_name)
$pythonPath = Join-Path $environmentPrefix 'python.exe'
$downloadDir = Join-Path $RuntimeRoot 'cache\downloads'
$wheelDir = Join-Path $RuntimeRoot 'cache\wheels'

if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
    throw 'This bootstrap and wheel lock require Windows x64.'
}
if (-not (Test-Path -LiteralPath (Join-Path $installTarget 'python.exe'))) {
    if (Test-Path -LiteralPath $installTarget) {
        throw 'An incomplete Conda installation exists; preserving it for inspection.'
    }
    New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
    $installerPath = Join-Path $downloadDir $runtimeConfig.installer_filename
    if (-not (Test-Path -LiteralPath $installerPath)) {
        if ($Offline) { throw 'Offline installation requires the verified installer in cache/downloads.' }
        & curl.exe --fail --location --retry 2 --connect-timeout 15 --max-time 240 --output $installerPath $runtimeConfig.installer_url
        if ($LASTEXITCODE -ne 0) { throw 'Official installer download failed; partial file is preserved for inspection.' }
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installerPath).Hash
    if ($actualHash -ne $runtimeConfig.installer_sha256) { throw 'Installer SHA-256 mismatch; refusing to execute.' }
    # /D must be the final argument. Do not add to global PATH or register default Python.
    $installerProcess = Start-Process -FilePath $installerPath -ArgumentList @('/InstallationType=JustMe', '/AddToPath=0', '/RegisterPython=0', '/S', ('/D=' + $installTarget)) -WindowStyle Hidden -PassThru -Wait
    if ($installerProcess.ExitCode -ne 0) { throw ('Installer failed with exit code ' + $installerProcess.ExitCode) }
}

$condaPath = Join-Path $installTarget 'Scripts\conda.exe'
if (-not (Test-Path -LiteralPath $condaPath) -or -not (Test-Path -LiteralPath (Join-Path $installTarget 'conda-meta\history'))) {
    throw 'Conda installation is incomplete; refusing to modify it.'
}
# Keep environment discovery local to this manager; do not edit the user's global .condarc.
& $condaPath config --file (Join-Path $installTarget '.condarc') --add envs_dirs (Join-Path $RuntimeRoot 'envs')
if ($LASTEXITCODE -ne 0) { throw 'Could not configure this manager environment directory.' }
if (-not (Test-Path -LiteralPath $pythonPath)) {
    if (Test-Path -LiteralPath $environmentPrefix) {
        throw 'An incomplete project environment exists; preserving it for inspection.'
    }
    $createArgs = @('create', '--prefix', $environmentPrefix, ('python=' + $runtimeConfig.python_version), 'pip', '--yes', '--no-shortcuts')
    if ($Offline) { $createArgs += '--offline' }
    & $condaPath @createArgs
    if ($LASTEXITCODE -ne 0) { throw 'Project environment creation failed; existing output is preserved.' }
}
if (-not (Test-Path -LiteralPath (Join-Path $environmentPrefix 'conda-meta\history'))) {
    throw 'The project Python is not a complete Conda environment.'
}
$versionCheck = 'import platform, sys; actual=platform.python_version(); print(actual); sys.exit(0 if actual == sys.argv[1] else 1)'
& $pythonPath -c $versionCheck $runtimeConfig.python_version
if ($LASTEXITCODE -ne 0) { throw 'Unexpected Python version; no environment changes were made.' }
$pipArgs = @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', '--no-cache-dir', '--only-binary=:all:', '--require-hashes', '-r', (Join-Path $projectRoot $runtimeConfig.wheel_lock))
if ($Offline) {
    $pipArgs += @('--no-index', '--find-links', $wheelDir)
} else {
    $pipArgs += @('--index-url', 'https://pypi.org/simple')
}
& $pythonPath @pipArgs
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; no automatic cleanup or rollback is attempted.' }
& $pythonPath -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
& $pythonPath (Join-Path $PSScriptRoot 'check_environment.py')
if ($LASTEXITCODE -ne 0) { throw 'Synthetic environment smoke check failed.' }

# Publish machine-specific paths only after the new environment passes verification.
$localConfig = [ordered]@{
    runtime_root = $RuntimeRoot
    conda_root = $installTarget
    environment_prefix = $environmentPrefix
    python_path = $pythonPath
}
$localConfig | ConvertTo-Json | Set-Content -LiteralPath $localConfigPath -Encoding UTF8
Write-Output ('Project interpreter ready: ' + $pythonPath)
