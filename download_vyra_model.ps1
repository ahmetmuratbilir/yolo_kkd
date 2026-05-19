$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelsDir = Join-Path $ProjectDir "models"
$OutputPath = Join-Path $ModelsDir "vyra_yolo_ppe_best.pt"
$ModelUrl = "https://huggingface.co/Hexmon/vyra-yolo-ppe-detection/resolve/main/best.pt?download=true"

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
