$ErrorActionPreference = "Stop"

[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
$datasetScript = Join-Path $projectRoot "train\prepare_dataset.py"
$trainScript = Join-Path $projectRoot "train\train_model.py"
$runsDirectory = Join-Path $projectRoot "runs"

Set-Location $projectRoot

if (-not (Test-Path $pythonExe)) {
    Write-Host ""
    Write-Host "[오류] venv 가상환경을 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "확인한 경로: $pythonExe"
    Write-Host ""
    Write-Host "다음 명령으로 가상환경을 생성해주세요."
    Write-Host "python -m venv venv"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1단계: 얼굴 데이터셋 검사" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

& $pythonExe $datasetScript

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[오류] 데이터셋 검사를 통과하지 못했습니다." -ForegroundColor Red
    Write-Host "dataset 폴더와 얼굴 이미지 파일을 확인해주세요."
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "2단계: TensorBoard 실행" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$tensorBoardArguments = @(
    "-m"
    "tensorboard.main"
    "--logdir"
    "`"$runsDirectory`""
    "--port"
    "6006"
)

Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $tensorBoardArguments `
    -WorkingDirectory $projectRoot

Start-Sleep -Seconds 3

Write-Host "TensorBoard 주소: http://localhost:6006"
Start-Process "http://localhost:6006"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "3단계: MobileNetV2 학습 시작" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

& $pythonExe $trainScript

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[오류] 모델 학습 중 문제가 발생했습니다." -ForegroundColor Red
    Write-Host "results 폴더의 최신 training.log를 확인해주세요."
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "학습이 정상적으로 완료되었습니다." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "results 폴더에서 학습 결과를 확인하세요."