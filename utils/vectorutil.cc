/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#include "vectorutil.h"

namespace knowledgeembedding {
namespace utils {
float DowRow(const vector<float> &vec1, const vector<float> &vec2) {
    assert(vec1.size() == vec2.size());
    float res = 0;
    for (uint32_t i = 0; i < vec1.size(); i++) {
        res += vec1[i] * vec2[i];
    }
    return res;
}

float Norm(const vector<float> &vec) {
    float sum = 0;
    for (uint32_t i = 0; i < vec.size(); i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}
} // namespace utils
} // namespace knowledgeembedding
