/**
 * preprocess.h - 图像预处理模块
 *
 * ⚠️ 必须与训练时完全一致！
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace face_engine {

class Preprocessor {
public:
    // ImageNet 归一化参数（与训练一致）
    static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3] = {0.229f, 0.224f, 0.225f};
    static constexpr int INPUT_SIZE = 112;

    /**
     * 扩大人脸区域
     *
     * @param bbox 原始边界框
     * @param image_width 图像宽度
     * @param image_height 图像高度
     * @param expand_ratio 扩展比例（默认0.2，即20%）
     * @return 扩展后的边界框
     */
    static cv::Rect expand_bbox(const cv::Rect& bbox, int image_width, int image_height, float expand_ratio = 0.2f);

    /**
     * 预处理人脸图像
     *
     * @param image 原始图像
     * @param bbox 人脸边界框（可选，如果为空则使用整个图像）
     * @return 预处理后的张量 [1, 3, 112, 112]
     */
    static cv::Mat preprocess(const cv::Mat& image, const cv::Rect& bbox);

    /**
     * 仅预处理人脸区域（不扩展边界框）
     *
     * @param face_image 人脸图像
     * @return 预处理后的张量 [1, 3, 112, 112]
     */
    static cv::Mat preprocess_face(const cv::Mat& face_image);

private:
    // 归一化
    static cv::Mat normalize(const cv::Mat& float_img);

    // HWC -> CHW
    static cv::Mat hwc_to_chw(const cv::Mat& normalized);
};

} // namespace face_engine
