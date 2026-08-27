cmake_minimum_required(VERSION 3.20)
project(face_engine VERSION 1.0.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC UTF-8 编码支持
if(MSVC)
    add_compile_options(/utf-8)
endif()

# OpenCV (手动指定路径)
set(OpenCV_INCLUDE_DIRS "D:/opencv/build/include")
set(OpenCV_LIB_DIR "D:/opencv/build/x64/vc16/lib")
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# ONNX Runtime (手动指定路径)
set(ONNXRUNTIME_ROOT "D:/onnxruntime/onnxruntime-win-x64-gpu-1.19.2")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(STATUS "Found onnxruntime at ${ONNXRUNTIME_ROOT}")
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
    link_directories(${ONNXRUNTIME_LIB_DIR})
else()
    message(FATAL_ERROR "onnxruntime not found at ${ONNXRUNTIME_ROOT}!")
endif()

# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
set(SOURCES
    src/main.cpp
    src/face_engine.cpp
    src/detector.cpp
    src/recognizer.cpp
    src/database.cpp
    src/preprocess.cpp
)

# 创建可执行文件
add_executable(face_engine ${SOURCES})

# 链接库 (直接指定lib文件名)
target_link_libraries(face_engine
    opencv_world4100
    onnxruntime
)

# 设置输出目录
set_target_properties(face_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# 复制 DLL 到 exe 同目录（避免加载 System32 下的旧版本）
add_custom_command(TARGET face_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
        "$<TARGET_FILE_DIR:face_engine>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_cuda.dll"
        "$<TARGET_FILE_DIR:face_engine>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_shared.dll"
        "$<TARGET_FILE_DIR:face_engine>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "D:/opencv/build/x64/vc16/bin/opencv_world4100.dll"
        "$<TARGET_FILE_DIR:face_engine>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "D:/opencv/build/x64/vc16/bin/opencv_videoio_ffmpeg4100_64.dll"
        "$<TARGET_FILE_DIR:face_engine>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "D:/opencv/build/x64/vc16/bin/opencv_videoio_msmf4100_64.dll"
        "$<TARGET_FILE_DIR:face_engine>/"
    COMMENT "Copying DLLs to exe directory"
)

# 安装规则
install(TARGETS face_engine DESTINATION bin)
