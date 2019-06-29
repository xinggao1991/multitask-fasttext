/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#ifndef KNOWLEDGE_EMBEDDING_UTILS_MATRIXUTIL_H
#define KNOWLEDGE_EMBEDDING_UTILS_MATRIXUTIL_H
#include <string>
#include <vector>
#include "basicutil.h"

namespace knowledgeembedding {
namespace utils {
    // print matrix at index: i
    void Print(uint32_t i);
    // get matrix mul vec
    void MatrixMul(float *data,
                   const vector<float> &hidden_vec,
                   const vector<float> &mask_vec,
                   vector<float> &result_vec,
                   uint32_t row,
                   uint32_t col);
    // dow row function
    float MatrixDowRow(float *data1,
                       uint32_t idx1,
                       float *data2,
                       uint32_t idx2,
                       uint32_t col);
    float MatrixDowRow(float *data1,
                       uint32_t idx1,
                       const vector<float> &vec,
                       uint32_t col);
    float MatrixDowRow(float *data1,
                       uint32_t idx1,
                       const vector<float> &vec,
                       uint32_t col,
                       const vector<float> &mask_vec);
    // maxtirx update function
    void MatrixAdd(float *dest_data,
                   uint32_t dest_idx,
                   float *src_data,
                   uint32_t src_idx,
                   uint32_t col,
                   float rate);
    void MatrixAdd(float *dest_data,
                   uint32_t dest_idx,
                   const vector<float> &vec,
                   uint32_t col,
                   float rate);
    void MatrixAdd(float *dest_data,
                   uint32_t dest_idx,
                   const vector<float> &vec,
                   uint32_t col,
                   float rate,
                   const vector<float> &mask_vec);
    // get vector from matrix
    void MatrixGetVec(vector<float> *vec,
                      float *src_data,
                      uint32_t src_idx,
                      uint32_t col,
                      float rate);
    void MatrixGetVec(vector<float> *vec,
                      float *src_data,
                      uint32_t src_idx,
                      uint32_t col,
                      float rate,
                      const vector<float> &mask_vec);
    float MatrixNorm(float *dest_data,
                     uint32_t dest_idx,
                     uint32_t col);
} // namespace utils
} // namespace knowledgeembedding

#endif // KNOWLEDGE_EMBEDDING_UTILS_MATRIXUTIL_H
