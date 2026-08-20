@echo off
title Face Recognition System
cd /d "%~dp0"

:menu
cls
echo.
echo ========================================
echo    Face Recognition System
echo ========================================
echo.
echo   [1] Admin  - User Management
echo   [2] Gate   - Access Control
echo   [0] Exit
echo.
echo   Train: python trainer/train_optimal.py
echo ========================================
echo.

set /p choice=Select [0-2]:

if "%choice%"=="1" goto admin
if "%choice%"=="2" goto gate
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
echo Starting gate system...
start "Gate" /D "%~dp0" cmd /c "python gate\main.py & pause"
goto menu

:exit
echo.
timeout /t 1 >nul
exit
