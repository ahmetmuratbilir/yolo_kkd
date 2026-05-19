$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$VenvSitePackages = Join-Path $ProjectDir ".venv\Lib\site-packages"
$VenvScripts = Join-Path $ProjectDir ".venv\Scripts"
$DataYaml = Join-Path $ProjectDir "datasets\custom_ppe\data.yaml"
if (-not (Test-Path $DataYaml)) {
    $DataYaml = Join-Path $ProjectDir "datasets\auto_ppe\data.yaml"
}
$BaseModel = Join-Path $ProjectDir "models\vyra_yolo_ppe_best.pt"

if (-not (Test-Path $DataYaml)) {
    Write-Host "[train] data.yaml bulunamadi: $DataYaml" -ForegroundColor Red
    Write-Host "[train] Once dataset_review altindaki gorselleri etiketleyip YOLO formatinda datasets/custom_ppe klasorune aktar."
    exit 1
}

$env:PYTHONPATH = $VenvSitePackages
$env:PATH = "$VenvScripts;$env:PATH"
$env:YOLO_CONFIG_DIR = $ProjectDir

Set-Location $ProjectDir
$TrainCode = @"
from ultralytics import YOLO
model = YOLO(r"$BaseModel")
model.train(data=r"$DataYaml", epochs=50, imgsz=640, batch=8, project="runs_custom", name="ppe_finetune")
"@
& $BundledPython -c $TrainCode
