@echo off
title Face Recognition System
cd /d "%~dp0"

:menu
cls
echo.
echo ========================================
echo    Face Recognition System v2.0
echo ========================================
echo.
echo   [1] Admin - User Management
echo   [2] Gate  - Access Control (PyQt5)
echo   [3] Gate  - Access Control (ONNX)
echo   [4] Gate  - Access Control (C++)
echo   [0] Exit
echo.
echo   Train: python trainer/train_optimal.py
echo ========================================
echo.

set /p choice="Select [0-4]: "

if "%choice%"=="1" goto admin
if "%choice%"=="2" goto gate
if "%choice%"=="3" goto onnx
if "%choice%"=="4" goto cpp
if "%choice%"=="0" goto exit

echo Invalid choice
timeout /t 2 >nul
goto menu

:admin
echo.
echo Starting admin system...
start "Admin" /D "%~dp0" cmd /c "python admin\main.py & pause"
goto menu

:gate
echo.
echo Starting gate system (PyQt5)...
start "Gate" /D "%~dp0" cmd /c "python gate\main.py & pause"
goto menu

:onnx
echo.
echo Starting gate system (ONNX)...
start "Gate-ONNX" /D "%~dp0" cmd /c "python deploy\gate_onnx.py & pause"
goto menu

:cpp
echo.
echo Starting gate system (C++)...
start "Gate-CPP" /D "%~dp0\deploy" cmd /c "bin\Release\face_engine.exe & pause"
goto menu

:exit
echo.
timeout /t 1 >nul
exit
