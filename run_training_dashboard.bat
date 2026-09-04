@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_training_dashboard.ps1"

if errorlevel 1 (
    echo.
    echo The training process stopped because of an error.
)

echo.
pause