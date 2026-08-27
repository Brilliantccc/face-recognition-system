/**
 * detector.cpp - YOLO 人脸检测器实现
 */
#include "detector.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cstring>

namespace face_engine {

Detector::Detector(const std::string& model_path, bool use_gpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "detector") {

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(4);

    // 启用 GPU 加速
    if (use_gpu) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "Detector: GPU acceleration enabled" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cout << "Detector: GPU failed [" << e.GetOrtErrorCode() << "] " << e.what() << std::endl;
            std::cout << "Detector: Falling back to CPU" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Detector: GPU error: " << e.what() << std::endl;
            std::cout << "Detector: Falling back to CPU" << std::endl;
        }
    }

    // Windows 需要宽字符路径
    std::wstring wmodel_path(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, wmodel_path.c_str(), session_options);

    // 获取输入输出名称
    auto allocator = Ort::AllocatorWithDefaultOptions();

    // YOLOv8 输入: [batch, 3, 640, 640]
    input_size_ = cv::Size(640, 640);

    std::cout << "Detector model loaded successfully" << std::endl;
}

cv::Mat Detector::preprocess(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, input_size_);

    // BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // 转换为 float 并归一化到 [0, 1]
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // HWC -> CHW
    std::vector<cv::Mat> channels;
    cv::split(float_img, channels);

    cv::Mat tensor(1, 3 * input_size_.height * input_size_.width, CV_32FC1);
    for (int c = 0; c < 3; ++c) {
        int offset = c * input_size_.height * input_size_.width;
        memcpy(tensor.data + offset * sizeof(float),
               channels[c].data,
               input_size_.height * input_size_.width * sizeof(float));
    }

    std::vector<int> new_shape = {1, 3, input_size_.height, input_size_.width};
    cv::Mat result = tensor.reshape(1, new_shape);
    return result;
}

std::vector<Detection> Detector::detect(const cv::Mat& frame) {
    cv::Mat tensor = preprocess(frame);

    // 创建输入张量
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, 3, input_size_.height, input_size_.width};

    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, reinterpret_cast<float*>(tensor.data),
        3 * input_size_.height * input_size_.width,
        shape.data(), shape.size());

    // 运行推理
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    auto output = session_->Run(Ort::RunOptions{},
        input_names, &input, 1, output_names, 1);

    // 获取输出
    float* output_data = output[0].GetTensorMutableData<float>();
    auto output_shape = output[0].GetTensorTypeAndShapeInfo().GetShape();

    // YOLOv8 输出: [1, 84, 8400] -> 转置为 [8400, 84]
    int num_detections = output_shape[2];  // 8400
    int num_features = output_shape[1];    // 84 (4 bbox + 80 classes)

    return postprocess(output_data, num_detections, frame.cols, frame.rows);
}

std::vector<Detection> Detector::postprocess(
    const float* output_data,
    int output_size,
    int frame_width,
    int frame_height,
    float conf_threshold,
    float iou_threshold) {

    std::vector<Detection> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    // 计算缩放比例
    float x_scale = static_cast<float>(frame_width) / input_size_.width;
    float y_scale = static_cast<float>(frame_height) / input_size_.height;

    for (int i = 0; i < output_size; ++i) {
        // YOLOv8 输出格式: [1, 84, 8400]
        // 行优先存储: output_data[c * output_size + i] 是第 i 个检测框的第 c 个特征
        float cx = output_data[0 * output_size + i];      // x center
        float cy = output_data[1 * output_size + i];      // y center
        float w = output_data[2 * output_size + i];       // width
        float h = output_data[3 * output_size + i];       // height

        // 获取类别分数（对于人脸检测，通常只有一类）
        float max_score = 0;
        for (int c = 4; c < 84; ++c) {
            float score = output_data[c * output_size + i];
            if (score > max_score) {
                max_score = score;
            }
        }

        if (max_score < conf_threshold) {
            continue;
        }

        // 转换为边界框格式
        int x1 = static_cast<int>((cx - w / 2) * x_scale);
        int y1 = static_cast<int>((cy - h / 2) * y_scale);
        int width = static_cast<int>(w * x_scale);
        int height = static_cast<int>(h * y_scale);

        // 裁剪到图像范围
        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        width = std::min(width, frame_width - x1);
        height = std::min(height, frame_height - y1);

        boxes.push_back(cv::Rect(x1, y1, width, height));
        scores.push_back(max_score);
    }

    // NMS
    std::vector<int> indices = nms(boxes, scores, iou_threshold);

    for (int idx : indices) {
        Detection det;
        det.bbox = boxes[idx];
        det.confidence = scores[idx];
        detections.push_back(det);
    }

    return detections;
}

std::vector<int> Detector::nms(
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& scores,
    float threshold) {

    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 按分数降序排序
    std::sort(indices.begin(), indices.end(),
        [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);

    for (int i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;

        keep.push_back(idx);

        for (int j = i + 1; j < indices.size(); ++j) {
            int jdx = indices[j];
            if (suppressed[jdx]) continue;

            // 计算 IoU
            cv::Rect intersection = boxes[idx] & boxes[jdx];
            float intersection_area = intersection.area();
            float union_area = boxes[idx].area() + boxes[jdx].area() - intersection_area;

            if (intersection_area / union_area > threshold) {
                suppressed[jdx] = true;
            }
        }
    }

    return keep;
}

std::vector<std::vector<Detection>> Detector::detect_batch(const std::vector<cv::Mat>& frames) {
    std::vector<std::vector<Detection>> results;
    for (const auto& frame : frames) {
        results.push_back(detect(frame));
    }
    return results;
}

} // namespace face_engine
