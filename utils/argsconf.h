/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#ifndef KNOWLEDGE_EMBEDDING_UTILS_ARGSCONF_H
#define KNOWLEDGE_EMBEDDING_UTILS_ARGSCONF_H

#include <map>
#include <string>
#include <vector>

#include "basicutil.h"
#include "fileutil.h"

namespace knowledgeembedding {
    enum class ModelName : int {skip = 1, cls, kb, pair};
    enum class LossFun : int {ng = 1, softmax};
    class ArgsConf {
        public:
            ArgsConf();
            ~ArgsConf();
            bool Init(const string &conf_file);
        void SetOutputDir();
            void PrintArgs();
            void CheckArgs();
            template<typename T>
            void CheckMin(const T &param, T minval, const string &err);
            float GetParamNum(const string &key);
            string GetParamStr(const string &key);

        public: // user set conf
            map<string, string *> param_str_;
            map<string, int *> param_int_;
            map<string, float *> param_float_;
            map<string, bool *> param_bool_;
            map<string, string> params_map_;
            string process_ = "";
            string modeldir_ = "";
            string trainfile_ = "";
            string evalfile_ = "";

            int minlen_ = 3;
            int maxlen_ = 10000;
            int maxvocabsize_ = 30000000;
            int maxphrasesize_ = 30000000;

            int minwordfreq_ = 10;
            int minphrasefreq_ = 10;

            int dim_ = 64;
            int ngram_ = 2;
            int subngram_ = 2;
            int phrasefreqthreshold_ = 3;
            int windowsize_ = 5;

            int thread_ = 20;
            int getlossevery_ = 100;
            int evalevery_ = 10000;
            int epoch_ = 1;

            float learnrate_ = 0.05;
            float freqsample_ = 0.0001;
            float dropoutkeeprate_ = 1.0;

            bool useskipgram_ = true;
            bool usecls_ = true;
            bool usepair_ = true;

        public: // loaded confs
            atomic<uint64_t> totallinenum_;
            atomic<uint64_t> curlinenum_;
            atomic<float> curlearnrate_;

            string outputdir_ = "";
    };
} // namespace knowledgeembedding

#endif // KNOWLEDGE_EMBEDDING_UTILS_ARGSCONF_H
