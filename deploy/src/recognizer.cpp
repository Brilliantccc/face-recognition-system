/**
 * recognizer.cpp - MobileFaceNet 人脸识别器实现
 */
#include "recognizer.h"
#include "preprocess.h"
#include <iostream>
#include <cmath>

namespace face_engine {

Recognizer::Recognizer(const std::string& model_path, bool use_gpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "recognizer"),
      embedding_size_(128) {

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(4);

    // 启用 GPU 加速
    if (use_gpu) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "Recognizer: GPU acceleration enabled" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Recognizer: GPU not available, using CPU" << std::endl;
        }
    }

    // Windows 需要宽字符路径
    std::wstring wmodel_path(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, wmodel_path.c_str(), session_options);

    std::cout << "Recognizer model loaded successfully" << std::endl;
}

cv::Mat Recognizer::preprocess(const cv::Mat& face_image) {
    return Preprocessor::preprocess_face(face_image);
}

std::vector<float> Recognizer::extract_embedding(const cv::Mat& face_image) {
    cv::Mat tensor = preprocess(face_image);

    // 创建输入张量
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, 3, Preprocessor::INPUT_SIZE, Preprocessor::INPUT_SIZE};

    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, reinterpret_cast<float*>(tensor.data),
        3 * Preprocessor::INPUT_SIZE * Preprocessor::INPUT_SIZE,
        shape.data(), shape.size());

    // 运行推理
    const char* input_names[] = {"input"};
    const char* output_names[] = {"embedding"};
    auto output = session_->Run(Ort::RunOptions{},
        input_names, &input, 1, output_names, 1);

    // 获取输出
    float* data = output[0].GetTensorMutableData<float>();
    std::vector<float> emb(data, data + embedding_size_);

    // L2 归一化
    float norm = 0.0f;
    for (float v : emb) norm += v * v;
    norm = std::sqrt(norm);

    if (norm > 0) {
        for (float& v : emb) v /= norm;
    }

    return emb;
}

std::vector<std::vector<float>> Recognizer::extract_embedding_batch(
    const std::vector<cv::Mat>& face_images) {

    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(face_images.size());

    for (const auto& face : face_images) {
        embeddings.push_back(extract_embedding(face));
    }

    return embeddings;
}

} // namespace face_engine
