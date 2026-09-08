@echo off
REM Face Recognition Engine Build Script

echo ========================================
echo    Face Recognition Engine Builder
echo ========================================

REM Check vcpkg
if not exist "D:\vcpkg\vcpkg.exe" (
    echo Error: vcpkg not found at D:\vcpkg
    echo Please install vcpkg first
    pause
    exit /b 1
)

REM Create build directory
if not exist "build" mkdir build
cd build

REM Run CMake configuration
echo.
echo [1/2] Configuring CMake...
cmake .. -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake

if %errorlevel% neq 0 (
    echo Error: CMake configuration failed!
    pause
    exit /b 1
)

REM Build project
echo.
echo [2/2] Building project...
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo Error: Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Build successful!
echo ========================================
echo.
echo Executable: bin\Release\face_engine.exe
echo.

pause
