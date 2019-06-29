/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#include "basicutil.h"

namespace knowledgeembedding {
namespace utils {
bool StringToNumber(const string &text, float *val) {
   *val = atof(text.c_str());
   return true;
}
bool StringToNumber(const string &text, int32_t *val) {
   *val = atoi(text.c_str());
   return true;
}
bool StringToNumber(const string &text, uint32_t *val) {
   *val = atoi(text.c_str());
   return true;
}
void StringTrim(string* str) {
    size_t start_pos = 0;
    size_t end_pos = str->length();
    while (start_pos != end_pos && (str->at(start_pos) == ' '
       || str->at(start_pos) == '\t' || str->at(start_pos) == '\n'))
        start_pos++;
    if (start_pos == end_pos)
    {
        str->clear();
        return;
    }
    end_pos--;
    while (str->at(end_pos) == ' ' || str->at(end_pos) == '\t' || str->at(end_pos) == '\n') // end_pos always >= 0
        end_pos--;
    *str = str->substr(start_pos, end_pos - start_pos + 1);
}

string StringTrim(const string& str) {
    string s = str;
    StringTrim(&s);
    return s;
}

void StringToLower(string* s) {
    string::iterator end = s->end();
    for (string::iterator i = s->begin(); i != end; ++i)
        *i = tolower(static_cast<unsigned char>(*i));
}

void StringSplit(const string &dest_string,
                 const string &splitor,
                 vector<string> &result_vec) {
    assert(splitor.size() == 1);
    result_vec.clear();
    if (dest_string == "") {
        result_vec.push_back("");
    } else if (dest_string == splitor) {
        result_vec.push_back("");
        result_vec.push_back("");
    } else {
        // SplitString(dest_string, splitor.c_str(), &result_vec);
        char splitc = splitor[0];
        char c;
        std::stringbuf sb;
        sb.str(dest_string.c_str());
        string word = "";
        while ((c = sb.sbumpc()) != EOF) {
            if (c == splitc) {
                result_vec.push_back(word);
                word = "";
            } else {
                word.push_back(c);
            }
        }
        result_vec.push_back(word);
    }
}

// void StringSplit(const string &dest_string,
//                  const string &splitor,
//                  vector<string> &result_vec) {
//     assert(splitor.size() == 1);
//     result_vec.clear();
//     if (dest_string == "") {
//         result_vec.push_back("");
//     } else if (dest_string == splitor) {
//         result_vec.push_back("");
//         result_vec.push_back("");
//     } else {
//         SplitString(dest_string, splitor.c_str(), &result_vec);
//     }
// }

void TrimVector(vector<string> *str_vec) {
    for (unsigned int i = 0; i < str_vec->size(); i++) {
        utils::StringTrim(&((*str_vec)[i]));
    }
}

bool StartWith(const string &str_source, const string &str_prefix) {
    if (str_source == "" || str_prefix == ""
        || str_prefix.size() > str_source.size()) {
        return false;
    }
    if (str_source.substr(0, str_prefix.size()) == str_prefix) {
        return true;
    }
    return false;
}

string GetFormatStr(float num, uint32_t width) {
    string res = to_string(num);
    uint32_t len = res.size();
    for (uint32_t i = len; i <= width; i++) {
        res += " ";
    }
    if (res.size() == len) {
        res += " ";
    }
    return res;
}

} // namespace utils
} // namespace knowledgeembedding

