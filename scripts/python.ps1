# Resolve the project interpreter without activating Conda or changing global PATH.
# Pass Python arguments directly, e.g. -m unittest discover -s tests -v.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$localConfigPath = Join-Path $projectRoot 'configs/runtime.local.json'
if (Test-Path -LiteralPath $localConfigPath) {
    $localConfig = Get-Content -Raw -Encoding UTF8 -LiteralPath $localConfigPath | ConvertFrom-Json
    $pythonPath = $localConfig.python_path
} else {
    $config = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $projectRoot 'configs/runtime.json') | ConvertFrom-Json
    $pythonPath = Join-Path $config.default_runtime_root ('envs\' + $config.environment_name + '\python.exe')
}
if (-not $pythonPath -or -not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'Project Python not found. Run scripts/setup/bootstrap.ps1 first.'
}
& $pythonPath @args
exit $LASTEXITCODE
