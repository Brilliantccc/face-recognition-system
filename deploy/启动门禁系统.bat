@echo off
cd /d "%~dp0"

echo ========================================
echo    C++ Gate System (ONNX Runtime)
echo ========================================
echo.
echo   Edit config/config.json to switch:
echo     - detector_model: detection ONNX
echo     - recognizer_model: recognition ONNX
echo     - database_path: embeddings file
echo.

if not exist bin\Release\face_engine.exe (
    echo Error: face_engine.exe not found!
    echo Run build.bat first to compile.
    pause
    exit /b 1
)

set PATH=%~dp0bin\Release;%PATH%
bin\Release\face_engine.exe
pause
