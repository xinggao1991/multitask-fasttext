/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#include "matrixutil.h"

namespace knowledgeembedding {
namespace utils {
// print matrix at index: i
void Print(float *data, uint32_t i, uint32_t col) {
    string res = "";
    for (uint32_t j = 0; j < col; j++) {
        uint64_t idx = uint64_t(i) * uint64_t(col) + uint64_t(j);
        res += to_string(data[idx]) + " ";
    }
    StringTrim(&res);
    cerr << res << endl;
}

void MatrixMul(float *data,
               const vector<float> &hidden_vec,
               const vector<float> &mask_vec,
               vector<float> &result_vec,
               uint32_t row,
               uint32_t col) {
    assert(hidden_vec.size() == col);
    assert(result_vec.size() == row);
    for (uint32_t i = 0; i < row; i++) {
        for (uint32_t j = 0; j < col; j++) {
            uint64_t idx = uint64_t(i) * uint64_t(col) + uint64_t(j);
            result_vec[i] += data[idx] * hidden_vec[j] * mask_vec[j];
        }
    }
}

// dow row function
float MatrixDowRow(float *data1,
                   uint32_t idx1,
                   float *data2,
                   uint32_t idx2,
                   uint32_t col) {
    float res = 0;
    for (uint32_t i = 0; i < col; i++) {
        uint64_t i1 = uint64_t(idx1) * uint64_t(col) + uint64_t(i);
        uint64_t i2 = uint64_t(idx2) * uint64_t(col) + uint64_t(i);
        res += data1[i1] * data2[i2];
    }
    return res;
}
float MatrixDowRow(float *data1,
                   uint32_t idx1,
                   const vector<float> &vec,
                   uint32_t col) {
    assert(col == vec.size());
    float res = 0;
    for (uint32_t i = 0; i < col; i++) {
        uint64_t i1 = uint64_t(idx1) * uint64_t(col) + uint64_t(i);
        res += data1[i1] * vec[i];
    }
    return res;
}
float MatrixDowRow(float *data1,
                   uint32_t idx1,
                   const vector<float> &vec,
                   uint32_t col,
                   const vector<float> &mask_vec) {
    assert(col == vec.size());
    assert(col == mask_vec.size());
    float res = 0;
    for (uint32_t i = 0; i < col; i++) {
        uint64_t i1 = uint64_t(idx1) * uint64_t(col) + uint64_t(i);
        res += data1[i1] * vec[i] * mask_vec[i];
    }
    return res;
}
// maxtirx update function
void MatrixAdd(float *dest_data,
               uint32_t dest_idx,
               float *src_data,
               uint32_t src_idx,
               uint32_t col,
               float rate) {
    for (uint32_t i = 0; i < col; i++) {
        uint64_t idest = uint64_t(dest_idx) * uint64_t(col) + uint64_t(i);
        uint64_t isrc = uint64_t(src_idx) * uint64_t(col) + uint64_t(i);
        dest_data[idest] += rate * src_data[isrc];
    }
}
void MatrixAdd(float *dest_data,
               uint32_t dest_idx,
               const vector<float> &vec,
               uint32_t col,
               float rate) {
    assert(col == vec.size());
    for (uint32_t i = 0; i < col; i++) {
        uint64_t idest = uint64_t(dest_idx) * uint64_t(col) + uint64_t(i);
        dest_data[idest] += rate * vec[i];
    }
}
void MatrixAdd(float *dest_data,
               uint32_t dest_idx,
               const vector<float> &vec,
               uint32_t col,
               float rate,
               const vector<float> &mask_vec) {
    assert(col == vec.size());
    for (uint32_t i = 0; i < col; i++) {
        uint64_t idest = uint64_t(dest_idx) * uint64_t(col) + uint64_t(i);
        dest_data[idest] += rate * vec[i] * mask_vec[i];
    }
}


// get vector from matrix
void MatrixGetVec(vector<float> *vec,
                  float *src_data,
                  uint32_t src_idx,
                  uint32_t col,
                  float rate) {
    assert(col == vec->size());
    for (uint32_t i = 0; i < col; i++) {
        uint64_t isrc = uint64_t(src_idx) * uint64_t(col) + uint64_t(i);
        (*vec)[i] += rate * src_data[isrc];
    }
}
void MatrixGetVec(vector<float> *vec,
                  float *src_data,
                  uint32_t src_idx,
                  uint32_t col,
                  float rate,
                  const vector<float> &mask_vec) {
    assert(col == vec->size());
    assert(col == mask_vec.size());
    for (uint32_t i = 0; i < col; i++) {
        uint64_t isrc = uint64_t(src_idx) * uint64_t(col) + uint64_t(i);
        (*vec)[i] += rate * src_data[isrc] * mask_vec[i];
    }
}
float MatrixNorm(float *dest_data,
                 uint32_t dest_idx,
                 uint32_t col) {
    float res = 0;
    for (uint32_t i = 0; i < col; i++) {
        uint64_t idest = uint64_t(dest_idx) * uint64_t(col) + uint64_t(i);
        res += dest_data[idest] * dest_data[idest];
    }
    return sqrt(res);
}
} // namespace utils
} // namespace knowledgeembedding
