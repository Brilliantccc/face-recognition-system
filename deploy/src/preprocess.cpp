/**
 * preprocess.cpp - 图像预处理实现
 */
#include "preprocess.h"
#include <algorithm>
#include <cstring>

namespace face_engine {

cv::Rect Preprocessor::expand_bbox(const cv::Rect& bbox, int image_width, int image_height, float expand_ratio) {
    int mx = static_cast<int>(bbox.width * expand_ratio);
    int my = static_cast<int>(bbox.height * expand_ratio);

    int x1 = std::max(0, bbox.x - mx);
    int y1 = std::max(0, bbox.y - my);
    int x2 = std::min(image_width, bbox.x + bbox.width + mx);
    int y2 = std::min(image_height, bbox.y + bbox.height + my);

    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

cv::Mat Preprocessor::normalize(const cv::Mat& float_img) {
    cv::Mat normalized = cv::Mat::zeros(float_img.size(), CV_32FC3);

    for (int c = 0; c < 3; ++c) {
        cv::Mat ch;
        cv::extractChannel(float_img, ch, c);
        ch = (ch - MEAN[c]) / STD[c];
        cv::insertChannel(ch, normalized, c);
    }

    return normalized;
}

cv::Mat Preprocessor::hwc_to_chw(const cv::Mat& normalized) {
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);

    int height = normalized.rows;
    int width = normalized.cols;
    cv::Mat tensor(1, 3 * height * width, CV_32FC1);

    for (int c = 0; c < 3; ++c) {
        int offset = c * height * width;
        memcpy(tensor.data + offset * sizeof(float),
               channels[c].data,
               height * width * sizeof(float));
    }

    std::vector<int> new_shape = {1, 3, height, width};
    return tensor.reshape(1, new_shape);
}

cv::Mat Preprocessor::preprocess(const cv::Mat& image, const cv::Rect& bbox) {
    // 1. 扩大人脸区域 20%
    cv::Rect roi = expand_bbox(bbox, image.cols, image.rows, 0.2f);
    cv::Mat face = image(roi).clone();

    return preprocess_face(face);
}

cv::Mat Preprocessor::preprocess_face(const cv::Mat& face_image) {
    // 1. Resize 到 112x112
    cv::Mat resized;
    cv::resize(face_image, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));

    // 2. 转换为 float
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 3. 归一化
    cv::Mat normalized = normalize(float_img);

    // 4. HWC -> CHW
    return hwc_to_chw(normalized);
}

} // namespace face_engine
