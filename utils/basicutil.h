/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#ifndef KNOWLEDGE_EMBEDDING_UTILS_BASICUTIL_H
#define KNOWLEDGE_EMBEDDING_UTILS_BASICUTIL_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using std::atomic;
using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::flush;
using std::ifstream;
using std::ios;
using std::istream;
using std::map;
using std::max;
using std::min;
using std::minstd_rand;
using std::make_pair;
using std::make_shared;
using std::ofstream;
using std::priority_queue;
using std::setw;
using std::shared_ptr;
using std::streamoff;
using std::streampos;
using std::string;
using std::thread;
using std::to_string;
using std::uniform_real_distribution;
using std::vector;
using std::pair;

namespace knowledgeembedding {
namespace utils {
    bool StringToNumber(const string &text, float *val);
    bool StringToNumber(const string &text, int32_t *val);
    bool StringToNumber(const string &text, uint32_t *val);
    void StringTrim(string* str);
    string StringTrim(const string& str);
    void StringToLower(string* s);
    // split string and store it in vector
    // only support single char splitor
    void StringSplit(const string &dest_string,
                     const string &splitor,
                     vector<string> &result_vec);
    // trim string vector
    void TrimVector(vector<string> *str_vec);
    // judge whether "str_prefix" is prefix of "str_source"
    bool StartWith(const string &str_source, const string &str_prefix);
    // change number to format str
    string GetFormatStr(float num, uint32_t width = 7);
} // namespace utils
} // namespace knowledgeembedding

#endif // KNOWLEDGE_EMBEDDING_UTILS_BASICUTIL_H
