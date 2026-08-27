/**
 * recognizer.h - MobileFaceNet 人脸识别器接口
 */
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace face_engine {

class Recognizer {
public:
    Recognizer(const std::string& model_path, bool use_gpu = true);
    ~Recognizer() = default;

    // 提取人脸嵌入向量
    std::vector<float> extract_embedding(const cv::Mat& face_image);

    // 批量提取嵌入向量
    std::vector<std::vector<float>> extract_embedding_batch(
        const std::vector<cv::Mat>& face_images);

    // 获取嵌入维度
    int get_embedding_size() const { return embedding_size_; }

private:
    // 预处理（与训练一致）
    cv::Mat preprocess(const cv::Mat& face_image);

    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
    int embedding_size_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

} // namespace face_engine
