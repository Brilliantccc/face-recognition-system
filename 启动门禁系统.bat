@echo off
title Gate System
cd /d "%~dp0"
python gate/main.py
if errorlevel 1 (
    echo.
    echo Failed to start
    pause
)
