/**
 * face_engine.cpp - 人脸引擎实现
 */
#include "face_engine.h"
#include "detector.h"
#include "recognizer.h"
#include "database.h"
#include <iostream>
#include <chrono>

namespace face_engine {

// 简单计时器
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

class FaceEngineImpl : public FaceEngine {
public:
    FaceEngineImpl() = default;
    ~FaceEngineImpl() override = default;

    bool initialize(const EngineConfig& config) override {
        config_ = config;

        // 加载数据库
        std::cout << "Loading database: " << config.database_path << std::endl;
        database_ = std::make_unique<FaceDatabase>();
        if (!database_->load(config.database_path)) {
            std::cerr << "Failed to load database!" << std::endl;
            return false;
        }
        std::cout << "Database loaded: " << database_->get_person_count() << " persons" << std::endl;

        // 加载检测器
        std::cout << "Loading detector: " << config.detector_model_path << std::endl;
        detector_ = std::make_unique<Detector>(config.detector_model_path, config.use_gpu);

        // 加载识别器
        std::cout << "Loading recognizer: " << config.recognizer_model_path << std::endl;
        recognizer_ = std::make_unique<Recognizer>(config.recognizer_model_path, config.use_gpu);

        // 重置统计
        reset_stats();

        std::cout << "Engine initialized successfully!" << std::endl;
        return true;
    }

    std::vector<RecognitionResult> recognize(const cv::Mat& frame) override {
        std::vector<RecognitionResult> results;

        // 检查帧是否有效
        if (frame.empty()) {
            std::cerr << "Warning: empty frame" << std::endl;
            return results;
        }

        // 检测人脸
        Timer timer;
        timer.start();
        auto detections = detector_->detect(frame);
        double detection_ms = timer.stop();
        stats_.avg_detection_ms = (stats_.avg_detection_ms * stats_.total_frames + detection_ms) /
                                  (stats_.total_frames + 1);

        stats_.detected_faces += detections.size();

        for (const auto& det : detections) {
            RecognitionResult result;
            result.detection = det;

            // 提取人脸区域
            cv::Mat face = frame(det.bbox).clone();

            // 提取嵌入向量
            timer.start();
            result.embedding = recognizer_->extract_embedding(face);
            double recognition_ms = timer.stop();
            stats_.avg_recognition_ms = (stats_.avg_recognition_ms * stats_.total_frames + recognition_ms) /
                                        (stats_.total_frames + 1);

            // 匹配数据库
            auto [is_known, similarity, name] = database_->match(
                result.embedding, config_.recognition_threshold);

            result.is_known = is_known;
            result.similarity = similarity;
            result.name = name;

            if (is_known) {
                stats_.recognized_faces++;
            }

            results.push_back(result);
        }

        stats_.total_frames++;
        return results;
    }

    std::vector<std::vector<RecognitionResult>> recognize_batch(
        const std::vector<cv::Mat>& frames) override {

        std::vector<std::vector<RecognitionResult>> all_results;
        for (const auto& frame : frames) {
            all_results.push_back(recognize(frame));
        }
        return all_results;
    }

    PerformanceStats get_performance_stats() const override {
        return stats_;
    }

    void reset_stats() override {
        stats_ = PerformanceStats();
    }

private:
    EngineConfig config_;
    std::unique_ptr<Detector> detector_;
    std::unique_ptr<Recognizer> recognizer_;
    std::unique_ptr<FaceDatabase> database_;
    PerformanceStats stats_;
};

std::unique_ptr<FaceEngine> create_face_engine() {
    return std::make_unique<FaceEngineImpl>();
}

} // namespace face_engine
