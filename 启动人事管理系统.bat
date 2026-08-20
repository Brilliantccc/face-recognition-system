@echo off
title Admin System
cd /d "%~dp0"
python admin/main.py
if errorlevel 1 (
    echo.
    echo Failed to start
    pause
)
