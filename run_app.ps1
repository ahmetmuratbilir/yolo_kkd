$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$fso = New-Object -ComObject Scripting.FileSystemObject
$ProjectDir = $fso.GetFolder($ProjectDir).ShortPath
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$VenvSitePackages = Join-Path $ProjectDir ".venv\Lib\site-packages"
$VenvScripts = Join-Path $ProjectDir ".venv\Scripts"

Set-Location $ProjectDir

if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
} elseif (Test-Path $BundledPython) {
    $PythonExe = $BundledPython
    $env:PYTHONPATH = $VenvSitePackages
    $env:PATH = "$VenvScripts;$env:PATH"
} else {
    Write-Host "[HATA] Calistirilabilir Python bulunamadi." -ForegroundColor Red
    Write-Host "Once Python kurun veya sanal ortami yeniden olusturun: py -3.12 -m venv .venv" -ForegroundColor Yellow
    exit 1
}

$env:YOLO_CONFIG_DIR = $ProjectDir

Write-Host "[run_app] Python: $PythonExe"
Write-Host "[run_app] Proje:  $ProjectDir"
Write-Host "[run_app] Uygulama basliyor. Cikmak icin kamera penceresinde q tusuna basin."

& $PythonExe main.py
