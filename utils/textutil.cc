/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#include "textutil.h"

namespace knowledgeembedding {
namespace utils {
void GetSegedWordList(const string &text, vector<string> &word_list) {
    word_list.clear();
    StringSplit(text, " ", word_list);
    word_list.erase(remove_if(word_list.begin(), word_list.end(),
                    [&](const string& e) {
                        return e == "";
                    })
                , word_list.end());
    word_list.shrink_to_fit();
}

string GetNgramWord(const vector<string> &word_list,
                    const vector<int32_t> &word_idx_vec,
                    unsigned int start,
                    unsigned int end) {
    assert(word_list.size() == word_idx_vec.size());
    assert(end < word_idx_vec.size());
    assert(start <= end);
    if (end >= word_list.size() || start > end) {
        return "";
    }
    string res_str = word_list[start];
    for (uint32_t i = start+1; i <= end; i++) {
        res_str += "_" + word_list[i];
        if (word_idx_vec[i] < 0) {
            return "";
        }
    }
    return res_str;
}

void GetNgramWordList(const vector<string> &word_list,
                      vector<string> &ngram_list,
                      uint32_t ngram) {
    ngram_list.clear();
    if (ngram <= 1) {
        return;
    }
    for (uint32_t i = 0; i < word_list.size(); i++) {
        string ngram_str = word_list[i];
        for (uint32_t j = i+1; j < word_list.size() && j < i+ngram; j++) {
            ngram_str += "_" + word_list[j];
            ngram_list.push_back(ngram_str);
        }
    }
}

void GetNgramWordList(const string &text,
                      vector<string> &ngram_list,
                      uint32_t ngram) {
    ngram_list.clear();
    if (ngram <= 1) {
        return;
    }
    vector<string> parts;
    StringSplit(text, " ", parts);
    if (parts.size() <= 1) {
        return;
    }
    GetNgramWordList(parts, ngram_list, ngram);
}

uint32_t GetWordSize(const string &word, bool &is_single_byte) {
    if (word.size() == 0) {
        return 0;
    }
    uint32_t multi_counter = 0;
    uint32_t single_counter = 0;
    bool single_tag = false;
    for (uint32_t i = 0; i < word.size(); i++) {
        // deal single byte chars
        if ((word[i] & 0x80) == 0x00) {
            if (single_tag == false) {
                single_counter++;
            }
            single_tag = true;
        } else {
            // deal multi byte chars
            if ((word[i] & 0xC0) == 0xC0) {
                multi_counter++;
            }
            single_tag = false;
        }
    }
    is_single_byte = (multi_counter == 0);
    return single_counter + multi_counter;
}

void PrintWordList(const vector<string> &word_list) {
    for (uint32_t i = 0; i < word_list.size(); i++) {
        cerr << i << "\t" << word_list[i] << endl;
    }
}

} // namespace utils
} // namespace knowledgeembedding
