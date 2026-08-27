/**
 * database.h - 人脸数据库接口
 */
#pragma once

#include <string>
#include <vector>
#include <utility>
#include <tuple>

namespace face_engine {

struct PersonRecord {
    std::string name;
    std::vector<float> embedding;
};

class FaceDatabase {
public:
    FaceDatabase() = default;
    ~FaceDatabase() = default;

    // 加载二进制数据库文件
    bool load(const std::string& db_path);

    // 匹配人脸嵌入向量
    // 返回: (是否匹配, 相似度, 人员姓名)
    std::tuple<bool, float, std::string> match(
        const std::vector<float>& embedding,
        float threshold = 0.6f) const;

    // 批量匹配
    std::vector<std::tuple<bool, float, std::string>> match_batch(
        const std::vector<std::vector<float>>& embeddings,
        float threshold = 0.6f) const;

    // 获取人员数量
    int get_person_count() const { return static_cast<int>(records_.size()); }

private:
    // 余弦相似度
    float cosine_similarity(
        const std::vector<float>& a,
        const std::vector<float>& b) const;

    std::vector<PersonRecord> records_;
};

} // namespace face_engine
