/**
 * face_engine.h - 人脸识别引擎主接口
 *
 * 提供人脸检测、识别、数据库匹配等功能
 */
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace face_engine {

// 人脸检测结果
struct Detection {
    cv::Rect bbox;       // 边界框
    float confidence;    // 置信度
};

// 识别结果
struct RecognitionResult {
    Detection detection;     // 检测结果
    std::vector<float> embedding;  // 人脸嵌入向量
    bool is_known;           // 是否已知人员
    std::string name;        // 人员姓名
    float similarity;        // 相似度
};

// 性能统计
struct PerformanceStats {
    float avg_detection_ms;     // 平均检测时间
    float avg_recognition_ms;   // 平均识别时间
    float avg_preprocess_ms;    // 平均预处理时间
    int total_frames;           // 总帧数
    int detected_faces;         // 检测到的人脸数
    int recognized_faces;       // 识别到的人脸数
};

// 引擎配置
struct EngineConfig {
    std::string detector_model_path;    // 检测模型路径
    std::string recognizer_model_path;  // 识别模型路径
    std::string database_path;          // 数据库路径
    float recognition_threshold;        // 识别阈值
    bool use_gpu;                       // 是否使用 GPU
    int input_width;                    // 输入宽度
    int input_height;                   // 输入高度
};

// 人脸引擎接口
class FaceEngine {
public:
    virtual ~FaceEngine() = default;

    // 初始化引擎
    virtual bool initialize(const EngineConfig& config) = 0;

    // 单帧识别
    virtual std::vector<RecognitionResult> recognize(const cv::Mat& frame) = 0;

    // 批量识别
    virtual std::vector<std::vector<RecognitionResult>> recognize_batch(
        const std::vector<cv::Mat>& frames) = 0;

    // 获取性能统计
    virtual PerformanceStats get_performance_stats() const = 0;

    // 重置统计
    virtual void reset_stats() = 0;
};

// 创建引擎实例
std::unique_ptr<FaceEngine> create_face_engine();

} // namespace face_engine
