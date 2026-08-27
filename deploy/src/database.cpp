/**
 * database.cpp - 人脸数据库实现
 */
#include "database.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>

namespace face_engine {

bool FaceDatabase::load(const std::string& db_path) {
    std::ifstream file(db_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open database: " << db_path << std::endl;
        return false;
    }

    // 读取文件头
    uint32_t version, person_count, embedding_dim;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&person_count), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&embedding_dim), sizeof(uint32_t));

    std::cout << "Database: version=" << version
              << ", persons=" << person_count
              << ", dim=" << embedding_dim << std::endl;

    if (version != 1 || embedding_dim != 128) {
        std::cerr << "Unsupported database format" << std::endl;
        return false;
    }

    // 读取所有 embedding
    std::vector<float> all_embeddings(person_count * embedding_dim);
    file.read(reinterpret_cast<char*>(all_embeddings.data()),
              person_count * embedding_dim * sizeof(float));

    // 读取所有人名
    records_.resize(person_count);
    for (uint32_t i = 0; i < person_count; ++i) {
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));

        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        records_[i].name = name;
        records_[i].embedding.assign(
            all_embeddings.begin() + i * embedding_dim,
            all_embeddings.begin() + (i + 1) * embedding_dim);
    }

    std::cout << "Database loaded successfully: " << person_count << " persons" << std::endl;
    return true;
}

float FaceDatabase::cosine_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b) const {

    if (a.size() != b.size()) return 0.0f;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denominator < 1e-6f) return 0.0f;

    return dot / denominator;
}

std::tuple<bool, float, std::string> FaceDatabase::match(
    const std::vector<float>& embedding,
    float threshold) const {

    bool is_known = false;
    float max_similarity = 0.0f;
    std::string best_name = "";

    for (const auto& record : records_) {
        float similarity = cosine_similarity(embedding, record.embedding);
        if (similarity > max_similarity) {
            max_similarity = similarity;
            best_name = record.name;
        }
    }

    is_known = (max_similarity >= threshold);

    return std::make_tuple(is_known, max_similarity, best_name);
}

std::vector<std::tuple<bool, float, std::string>> FaceDatabase::match_batch(
    const std::vector<std::vector<float>>& embeddings,
    float threshold) const {

    std::vector<std::tuple<bool, float, std::string>> results;
    results.reserve(embeddings.size());

    for (const auto& emb : embeddings) {
        results.push_back(match(emb, threshold));
    }

    return results;
}

} // namespace face_engine
