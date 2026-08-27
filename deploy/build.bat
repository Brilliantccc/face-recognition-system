@echo off
REM Face Recognition Engine Build Script

echo ========================================
echo    Face Recognition Engine Builder
echo ========================================

REM 检查 vcpkg 是否安装
if not exist "D:\vcpkg\vcpkg.exe" (
    echo Error: vcpkg not found at D:\vcpkg
    echo Please install vcpkg first
    exit /b 1
)

REM 创建构建目录
if not exist "build" mkdir build
cd build

REM 运行 CMake 配置
echo.
echo [1/2] Configuring CMake...
cmake .. -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake

if %errorlevel% neq 0 (
    echo Error: CMake configuration failed!
    exit /b 1
)

REM 编译项目
echo.
echo [2/2] Building project...
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo Error: Build failed!
    exit /b 1
)

echo.
echo ========================================
echo    Build successful!
echo ========================================
echo.
echo Executable: bin\Release\face_engine.exe
echo.
echo To run:
echo   cd ..
echo   bin\Release\face_engine.exe
echo.

pause
