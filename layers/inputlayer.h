/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#ifndef KNOWLEDGE_EMBEDDING_LAYERS_INPUTLAYER_H
#define KNOWLEDGE_EMBEDDING_LAYERS_INPUTLAYER_H

#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "../utils/basicutil.h"
#include "../utils/hashtable.h"
#include "../utils/matrixutil.h"
#include "../utils/textutil.h"
#include "../utils/vectorutil.h"

namespace knowledgeembedding {
class InputLayer {
    public:
        InputLayer(shared_ptr<ArgsConf> args_conf,
                shared_ptr<HashTable> hash_table);
        ~InputLayer();
        void Init();
        // get index vector from text
        void GetIdxVec(const string &text,
                       vector<int32_t> &idx_vec,
                       float boost_freq_sample = 10000,
                       bool usephrase = true);
        // get vector from data
        void GetLayerByIdxs(int32_t word_idx,
                            vector<float> &layer,
                            float rate = 1.0);
        void GetLayerByIdxs(const vector<int32_t> &word_idx_vec,
                            vector<float> &layer,
                            float boost_freq_sample,
                            bool use_discard_rate = false);
        // update word vector data
        void UpdateData(int32_t input_idx,
                        vector<float> &add_vec,
                        float rate = 1);
        void UpdateData(const vector<int32_t> &input_vec,
                        vector<float> &add_vec,
                        float rate = 1);
        // get top nearest
        void GetNearestNeighbor(const vector<int32_t> &idx_vec,
                                priority_queue<pair<float, uint32_t>> &heap,
                                const string &query_type);
        // write data
        void Save();
        void Load();

    public:
        float* data_;

    private:
        shared_ptr<HashTable> hash_table_;
        shared_ptr<ArgsConf> args_conf_;
        uint32_t row_ = 0;
        uint32_t col_ = 0;

        minstd_rand rng_;
        uniform_real_distribution<> uniform_;
}; // InputLayer
} // namespace knowledgeembedding

#endif // KNOWLEDGE_EMBEDDING_LAYERS_INPUTLAYER_H
