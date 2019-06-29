/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#ifndef KNOWLEDGE_EMBEDDING_UTILS_TEXTUTIL_H
#define KNOWLEDGE_EMBEDDING_UTILS_TEXTUTIL_H

#include <string>
#include <vector>

#include "argsconf.h"
#include "basicutil.h"

namespace knowledgeembedding {
namespace utils {
    // get word list of seged text
    void GetSegedWordList(const string &text, vector<string> &word_list);
    // get word list of ngram from text,
    // return ngram(res_str)
    string GetNgramWord(const vector<string> &word_list,
                        const vector<int32_t> &word_idx_vec,
                        unsigned int start,
                        unsigned int end);
    void GetNgramWordList(const vector<string> &word_list,
                          vector<string> &ngram_list,
                          uint32_t ngram);
    void GetNgramWordList(const string &text,
                          vector<string> &ngram_list,
                          uint32_t ngram);
    // get word size:
    // size(one english word) = 1
    // size(one chinese word) = 1
    uint32_t GetWordSize(const string &word, bool &is_single_byte);
    void PrintWordList(const vector<string> &word_list);
} // namespace utils
} // namespace knowledgeembedding
#endif // KNOWLEDGE_EMBEDDING_UTILS_TEXTUTIL_H
