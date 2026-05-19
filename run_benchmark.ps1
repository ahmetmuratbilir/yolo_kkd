$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$VenvSitePackages = Join-Path $ProjectDir ".venv\Lib\site-packages"
$VenvScripts = Join-Path $ProjectDir ".venv\Scripts"

$env:PYTHONPATH = $VenvSitePackages
$env:PATH = "$VenvScripts;$env:PATH"
$env:YOLO_CONFIG_DIR = $ProjectDir

Set-Location $ProjectDir
& $BundledPython benchmark_images.py
