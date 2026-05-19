$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelsDir = Join-Path $ProjectDir "models"
$OutputPath = Join-Path $ModelsDir "tanish_yolov8n_ppe_6class.pt"
$ModelUrl = "https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes/resolve/main/best.pt?download=true"

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

if (Test-Path $OutputPath) {
    Write-Host "[download] Model zaten var: $OutputPath"
    exit 0
}

Write-Host "[download] Model indiriliyor..."
Write-Host "[download] Kaynak: $ModelUrl"
Write-Host "[download] Hedef:  $OutputPath"

Invoke-WebRequest -Uri $ModelUrl -OutFile $OutputPath

Write-Host "[download] Tamamlandi: $OutputPath"
