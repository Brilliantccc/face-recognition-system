/**
 * detector.h - YOLO 人脸检测器接口
 */
#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "face_engine.h"

namespace face_engine {

class Detector {
public:
    Detector(const std::string& model_path, bool use_gpu = true);
    ~Detector() = default;

    // 检测人脸
    std::vector<Detection> detect(const cv::Mat& frame);

    // 批量检测
    std::vector<std::vector<Detection>> detect_batch(const std::vector<cv::Mat>& frames);

    // 获取模型输入尺寸
    cv::Size get_input_size() const { return input_size_; }

private:
    // 预处理
    cv::Mat preprocess(const cv::Mat& frame);

    // 后处理
    std::vector<Detection> postprocess(
        const float* output_data,
        int output_size,
        int frame_width,
        int frame_height,
        float conf_threshold = 0.5f,
        float iou_threshold = 0.45f);

    // NMS
    std::vector<int> nms(
        const std::vector<cv::Rect>& boxes,
        const std::vector<float>& scores,
        float threshold);

    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
    cv::Size input_size_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

} // namespace face_engine
