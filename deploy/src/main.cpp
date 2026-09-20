/**
 * main.cpp - 门禁系统主程序
 */
#include "face_engine.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

// Windows 控制台 UTF-8 编码
#ifdef _WIN32
#include <windows.h>
#endif

// 性能计时器
class Timer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        return duration.count() / 1000.0;  // 返回毫秒
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

int main() {
    // 设置控制台 UTF-8 编码
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "    人脸识别门禁系统 v3.3 (C++ ONNX)" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. 初始化引擎
    face_engine::EngineConfig config;
    config.detector_model_path = "models/yolov11l-face.onnx";
    config.recognizer_model_path = "models/mobilefacenet.onnx";
    config.database_path = "models/embeddings_mbn.bin";
    config.recognition_threshold = 0.55f;
    config.use_gpu = true;
    config.input_width = 640;
    config.input_height = 480;

    // 搜索模型文件（exe 可能在不同目录运行）
    auto find_file = [](const std::string& name) -> std::string {
        std::vector<std::string> prefixes = {"", "../", "../../"};
        for (const auto& p : prefixes) {
            std::ifstream f(p + name);
            if (f.good()) return p + name;
        }
        return name;  // fallback
    };
    config.detector_model_path = find_file(config.detector_model_path);
    config.recognizer_model_path = find_file(config.recognizer_model_path);
    config.database_path = find_file(config.database_path);

    std::cout << "Detector:   " << config.detector_model_path << std::endl;
    std::cout << "Recognizer: " << config.recognizer_model_path << std::endl;
    std::cout << "Database:   " << config.database_path << std::endl;

    std::cout << "\n[1/3] Initializing engine..." << std::endl;
    auto engine = face_engine::create_face_engine();
    if (!engine->initialize(config)) {
        std::cerr << "Failed to initialize engine!" << std::endl;
        return -1;
    }
    std::cout << "Engine initialized successfully!" << std::endl;

    // 2. 打开摄像头（依次尝试 index 0~5）
    std::cout << "\n[2/3] Opening camera..." << std::endl;
    cv::VideoCapture cap;
    for (int i = 0; i < 6; i++) {
        cap.open(i);
        if (cap.isOpened()) {
            std::cout << "Camera opened: index " << i << std::endl;
            break;
        }
    }
    if (!cap.isOpened()) {
        std::cerr << "Failed to open any camera!" << std::endl;
        return -1;
    }

    // 设置摄像头分辨率和帧率
    cap.set(cv::CAP_PROP_FRAME_WIDTH, config.input_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.input_height);
    cap.set(cv::CAP_PROP_FPS, 30);  // 设置目标帧率
    std::cout << "Camera opened successfully!" << std::endl;

    // 3. 实时识别循环
    std::cout << "\n[3/3] Starting recognition loop..." << std::endl;
    std::cout << "Press ESC to exit" << std::endl;
    std::cout << "========================================" << std::endl;

    cv::Mat frame;
    Timer timer;
    int frame_count = 0;
    double total_detection_ms = 0;
    double total_recognition_ms = 0;

    // 门禁冷却控制
    bool gate_opened = false;
    auto gate_open_time = std::chrono::steady_clock::now();
    const int COOLDOWN_SECONDS = 5;  // 冷却时间（秒）

    while (cap.read(frame)) {

        // 检查冷却期
        auto now = std::chrono::steady_clock::now();
        if (gate_opened) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - gate_open_time).count();
            if (elapsed < COOLDOWN_SECONDS) {
                // 冷却期内，跳过检测
                cv::putText(frame, "Gate opened, cooling down...",
                    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 2);
                cv::imshow("Face Recognition Gate System", frame);
                if (cv::waitKey(1) == 27) break;
                frame_count++;
                continue;
            } else {
                gate_opened = false;
                std::cout << "[GATE] Cooldown finished, resuming detection" << std::endl;
            }
        }

        timer.start();
        auto results = engine->recognize(frame);
        double inference_ms = timer.stop();

        // 绘制检测结果
        for (const auto& r : results) {
            const auto& det = r.detection;

            // 绘制边界框
            cv::Scalar color = r.is_known ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::rectangle(frame, det.bbox, color, 2);

            // 绘制标签
            std::string label;
            if (r.is_known) {
                label = r.name + " (" + std::to_string(static_cast<int>(r.similarity * 100)) + "%)";
            } else {
                label = "Stranger";
            }

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);

            cv::rectangle(frame,
                cv::Point(det.bbox.x, det.bbox.y - text_size.height - 10),
                cv::Point(det.bbox.x + text_size.width, det.bbox.y),
                color, -1);

            cv::putText(frame, label,
                cv::Point(det.bbox.x, det.bbox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 2);

            // 门禁控制：识别成功则开门
            if (r.is_known && r.similarity > config.recognition_threshold) {
                std::cout << "[GATE] Access granted: " << r.name
                          << " (similarity: " << r.similarity << ")" << std::endl;

                // 模拟开门信号（有实物后替换为实际代码）
                std::cout << "[SIGNAL] >>> OPEN GATE <<<" << std::endl;

                // 进入冷却期
                gate_opened = true;
                gate_open_time = std::chrono::steady_clock::now();
                std::cout << "[GATE] Entering cooldown for " << COOLDOWN_SECONDS << " seconds" << std::endl;
            }
        }

        // 显示 FPS
        auto stats = engine->get_performance_stats();
        float fps = 1000.0f / (inference_ms + 0.001f);
        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)),
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 255), 2);

        cv::imshow("Face Recognition Gate System", frame);
        if (cv::waitKey(1) == 27) break;  // ESC to exit

        frame_count++;
    }

    // 输出统计信息
    auto final_stats = engine->get_performance_stats();
    std::cout << "\n========================================" << std::endl;
    std::cout << "Session Statistics:" << std::endl;
    std::cout << "  Total frames: " << frame_count << std::endl;
    std::cout << "  Avg detection time: " << final_stats.avg_detection_ms << " ms" << std::endl;
    std::cout << "  Avg recognition time: " << final_stats.avg_recognition_ms << " ms" << std::endl;
    std::cout << "  Detected faces: " << final_stats.detected_faces << std::endl;
    std::cout << "  Recognized faces: " << final_stats.recognized_faces << std::endl;
    std::cout << "========================================" << std::endl;

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
