/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#ifndef KNOWLEDGE_EMBEDDING_UTILS_VECTORUTIL_H
#define KNOWLEDGE_EMBEDDING_UTILS_VECTORUTIL_H

#include <string>
#include <vector>
#include "basicutil.h"

namespace knowledgeembedding
{
namespace utils
{
    float DowRow(const vector<float> &vec1, const vector<float> &vec2);
    float Norm(const vector<float> &vec);
    template<typename T>
    void ParseVec(const vector<string> &vec_str, vector<T> &vec_num) {
        vec_num.clear();
        for (uint32_t i = 0; i < vec_str.size(); i++) {
            T val = 0;
            if (!utils::StringToNumber(vec_str[i], &val)) {
                cerr << "cannot parse number " << vec_str[i] << endl;
                assert(utils::StringToNumber(vec_str[i], &val));
            }
            vec_num.push_back(val);
        }
    }
    template<typename T>
    string JoinVector(const vector<T> &vec_num, const string &connector) {
        string res = "";
        for (uint32_t i = 0; i < vec_num.size(); i++) {
            res += connector + to_string(vec_num[i]);
        }
        if (res.size() >= connector.size()) {
            res = res.substr(connector.size());
        }
        return res;
    }
    template<typename T>
    void Print(const vector<T> &vec) {
        string res = "";
        for (uint32_t i = 0; i < vec.size(); i++) {
            res += to_string(vec[i]) + " ";
        }
        utils::StringTrim(&res);
        cerr << res << endl;
    }
} // namespace utils
} // namespace knowledgeembedding
#endif // KNOWLEDGE_EMBEDDING_UTILS_VECTORUTIL_H
