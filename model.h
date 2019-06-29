/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#ifndef KNOWLEDGE_EMBEDDING_MODEL_H
#define KNOWLEDGE_EMBEDDING_MODEL_H

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "layers/inputlayer.h"
#include "layers/outputlayer.h"
#include "utils/argsconf.h"
#include "utils/basicutil.h"
#include "utils/matrixutil.h"
#include "utils/textutil.h"
#include "utils/vectorutil.h"

#define SIGMOID_TABLE_SIZE 512
#define MAX_SIGMOID 8
#define LOG_TABLE_SIZE 512
#define NEG_TABLE_SIZE 10000000

namespace knowledgeembedding {
class Model {
    public:
        Model(shared_ptr<ArgsConf> args_conf,
              ModelName n,
              int32_t cls_number,
              shared_ptr<InputLayer> input_layer,
              const string &tag,
              shared_ptr<HashTable> hash_table);
        ~Model();
        // init model
        void Init();
        // init mask vector
        void RandomMask(vector<float> &mask_vec);
        // get random number
        float GetRandFloat(float min, float max);
        int32_t GetRandInt(int32_t min, int32_t max);
        // init negative sampling table
        void InitNegTable();
        void InitNegTable(const map<string, int32_t> &tag_count_map);
        uint32_t GetNegativeLabel(uint32_t positive_label);
        uint32_t GetNegativeLabel(const map<uint32_t, uint32_t> &pos_counter,
                                  uint32_t max_find_times = 50);
        // init sigmoid table
        void InitSigmoid();
        float GetSigmoid(float x);
        // init log table
        void InitLog();
        float GetLog(float x);
        // get model mean loss
        float GetLoss();
        // soft max function
        void SoftMax(const vector<float> &hidden_vec,
                     uint32_t target,
                     vector<float> &grad,
                     vector<float> &mask_vec);
        // update base function
        void UpdateBatch(const vector<float> &hidden_vec,
                         uint32_t output,
                         uint32_t label,
                         vector<float> &grad,
                         vector<float> &mask_vec);
        // negtive sampling update
        void UpdateNeg(const vector<int32_t> &input_vec,
                       const vector<float> &hidden_vec,
                       vector<float> &grad,
                       vector<float> &mask_vec,
                       uint32_t output,
                       map<uint32_t, uint32_t> &pos_counter,
                       bool use_neg = true,
                       uint32_t label = 1);
        // train skip model
        void UpdateSkip(const string &text);
        // train classify model
        void UpdateCls(const string &text, uint32_t label);
        // train pair model
        void UpdatePair(const string &text_1,
                        const string &text_2,
                        uint32_t label);
        float PredictPair(const vector<int32_t> &input_idx_vec_1,
                          const vector<int32_t> &input_idx_vec_2);
        // predict the label of example
        int32_t PredictCls(const vector<int32_t> &input_idx_vec);
        // predict score
        void PredictClsScore(const vector<int32_t> &input_idx_vec,
                             vector<pair<int32_t, float>> &predict);

        // save and load
        void Save(bool save_common_data);
        void Load(const map<string, int32_t> &tag_count_map);

    public:
        bool use_as_skip_example_ = false;
        float boost_freq_sample_ = 1.0;

    private:
        shared_ptr<ArgsConf> args_conf_;
        ModelName name_;
        LossFun loss_fun_; 
        uint32_t cls_number_;
        string class_tag_;

        float boost_ = 1.0;
        int32_t neg_sample_ = 5;

        int64_t total_loss_num_ = 0;
        double total_loss_value_ = 0.0;

        uint64_t rand_long_ = 1;
        minstd_rand rng_;
        uniform_real_distribution<> uniform_;

        uint64_t neg_table_index_ = 0;
        vector<int32_t> neg_table_;

        float *sigmoid_table_;
        float *log_table_;

        shared_ptr<HashTable> hash_table_;
        shared_ptr<InputLayer> input_layer_;
        shared_ptr<OutputLayer> output_layer_;
}; // Model
} // namespace knowledgeembedding
#endif // KNOWLEDGE_EMBEDDING_MODEL_H
