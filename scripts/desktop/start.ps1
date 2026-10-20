# Launch from any working directory, using the project's external environment.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
& (Join-Path $projectRoot 'scripts/python.ps1') (Join-Path $PSScriptRoot 'run_desktop.py') @args
exit $LASTEXITCODE
