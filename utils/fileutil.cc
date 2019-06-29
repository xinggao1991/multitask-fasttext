/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#include "fileutil.h"

namespace knowledgeembedding {
namespace utils {

int64_t Size(ifstream *ifs) {
    ifs->seekg(streamoff(0), ios::end);
    return ifs->tellg();
}

void Seek(ifstream *ifs, int64_t pos) {
    ifs->clear();
    ifs->seekg(streampos(pos));
}

bool GetLine(istream &ifs, string &line) {
    bool res = !getline(ifs, line).fail();
    StringToLower(&line);
    return res;
}

void ReadLine(istream *ifs, string *line, int32_t thread_id) {
    if (ifs->eof()) {
        ifs->clear();
        ifs->seekg(streampos(0));
    }
    *line = "";
    getline(*ifs, *line);
    StringTrim(line);
}

void OpenOutFile(const string &outputdir,
                 const string &filename,
                 ofstream &ofs) {
    string file = outputdir + "/" + filename;
    ofs.open(file);
    if (!ofs.is_open()) {
        cerr << "Error : cannot create file " + file << endl;
        assert(ofs.is_open());
    }
}

void WriteLine(ofstream &ofs, const string &line) {
    string lin = line;
    StringTrim(&lin);
    ofs << lin << endl;
    ofs << flush;
}

void CloseOutFile(ofstream *ofs) {
    (*ofs) << flush;
    ofs->close();
}

void CloseInFile(ifstream *ifs) {
    ifs->close();
}

uint64_t GetFileLineNumber(const string &file_path) {
    ifstream fin(file_path);
    if (!fin.is_open()) {
        cerr << "can not open file: " << file_path << endl;
        assert(fin.is_open());
    }
    uint64_t line_counter = 0;
    string line = "";
    while (getline(fin, line)) {
        line_counter++;
    }
    fin.close();
    return line_counter;
}

} // namespace utils
} // namespace knowledgeembedding
