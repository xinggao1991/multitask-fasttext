/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#ifndef KNOWLEDGE_EMBEDDING_EMBEDDING_H
#define KNOWLEDGE_EMBEDDING_EMBEDDING_H

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "model.h"

namespace knowledgeembedding {
class Embedding {
    public:
        Embedding() {}
        ~Embedding() {}
        void InitArgs(const string &confpath);
        // load train file word/phrase vocab
        void AddVocab(const vector<string> &parts,
                      const string &text,
                      HashTable* hash_word,
                      HashTable* hash_phrase,
                      uint64_t &word_counter,
                      uint64_t &phrase_counter);
        void LoadTrainVocab(bool only_count);
        // load eval example
        void LoadEvalExample();
        // init model
        void InitModel();
        bool HasWord(const string &word);
        bool GetWordVec(const string &input_word,
                        vector<float> &res_vec,
                        float default_val = 0);
        // word vec distance
        void GetDistance(const string &word,
                         int32_t top_size = 20,
                         const string &query_type_ = "_word");
        void Distance(int32_t top_size = 20);
        // get sentence vec of pipline
        void GetSentenceVec();
        // predict example
        void PredictCls(const pair<vector<int32_t>, string> &example,
                        pair<int32_t, float> &predict_res);
        float PredictPair(pair<pair<vector<int32_t>, vector<int32_t>>,
                              string> &example);
        void Predict(const string &line, string &res);
        void Predict();
        // check model with dev example
        void EvalCls(map<string, pair<int32_t, int32_t>> *result);
        void EvalPair(map<string, pair<int32_t, int32_t>> *result);
        // print eval infos while training
        void PrintEvalInfo(float progress, bool is_eval);
        // train thread
        void TrainThread(int32_t thread_id);
        // train model
        void Train();
        // save and load
        void SaveMap(map<string, int32_t> &the_map, const string &file);
        void Save();
        void LoadMap(map<string, int32_t> &the_map, const string &file);
        void Load();
        // main process function
        void MainProcess();

    private:
        shared_ptr<ArgsConf> args_conf_;
        shared_ptr<Model> skip_model_;
        shared_ptr<Model> kb_model_;
        map<string, shared_ptr<Model>> cls_model_map_;
        map<string, shared_ptr<Model>> pair_model_map_;

        map<string, int32_t> cls_tag_map_;
        map<string, int32_t> cls_tag_count_map_;
        map<string, int32_t> pair_tag_map_;
        map<string, int32_t> pair_tag_count_map_;

        shared_ptr<InputLayer> input_layer_;
        shared_ptr<HashTable> hash_table_;
        vector<pair<vector<int32_t>, string>> cls_eval_;
        vector<pair<pair<vector<int32_t>, vector<int32_t>>, string>> pair_eval_;

        string query_type_ = "_all";
}; // Embedding
} // namespace knowledgeembedding
#endif // KNOWLEDGE_EMBEDDING_EMBEDDING_H
