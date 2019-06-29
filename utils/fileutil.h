/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#ifndef KNOWLEDGE_EMBEDDING_UTILS_FILEUTIL_H
#define KNOWLEDGE_EMBEDDING_UTILS_FILEUTIL_H

#include <string>

#include "basicutil.h"

namespace knowledgeembedding {
namespace utils {
    int64_t Size(ifstream *ifs);
    void Seek(ifstream *ifs, int64_t pos);
    bool GetLine(istream &ifs, string &line);
    void ReadLine(istream *ifs, string *line, int32_t thread_id);
    void OpenOutFile(const string &outputdir,
                     const string &filename,
                     ofstream &ofs);
    void WriteLine(ofstream &ofs, const string &line);
    void CloseOutFile(ofstream *ofs);
    void CloseInFile(ifstream *ifs);
    uint64_t GetFileLineNumber(const string &file_path);
} // namespace utils
} // namespace knowledgeembedding
#endif // KNOWLEDGE_EMBEDDING_UTILS_FILEUTIL_H
