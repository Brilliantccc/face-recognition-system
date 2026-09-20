@echo off
title Gate System
cd /d "%~dp0"

:menu
cls
echo.
echo ========================================
echo    Face Recognition Gate System
echo ========================================
echo.
echo   [1] Start Gate System (PyQt5 GUI)
echo       - Model switching available in settings panel
echo.
echo   [2] Start Gate System (ONNX, high performance)
echo       - Uses ONNX Runtime, edit config/config.json to switch models
echo.
echo   [0] Exit
echo.
echo ========================================
echo.

set /p choice="Select [0-2]: "

if "%choice%"=="1" goto gate
if "%choice%"=="2" goto onnx
if "%choice%"=="0" goto exit

echo Invalid choice
timeout /t 2 >nul
goto menu

:gate
echo.
echo Starting Gate System (PyQt5)...
start "Gate" /D "%~dp0" cmd /c "python gate\main.py & pause"
goto menu

:onnx
echo.
echo Starting Gate System (ONNX)...
start "Gate-ONNX" /D "%~dp0" cmd /c "python deploy\gate_onnx.py & pause"
goto menu

:exit
exit
