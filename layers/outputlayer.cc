/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#include "outputlayer.h"

namespace knowledgeembedding {

OutputLayer::OutputLayer(shared_ptr<HashTable> hash_table,
                        shared_ptr<ArgsConf> args_conf,
                        ModelName n,
                        int cls_number,
                        const string &tag): name_(n), class_tag_(tag) {
    assert(cls_number > 1);
    hash_table_ = hash_table;
    args_conf_ = args_conf;
    row_ = uint32_t(cls_number);
    col_ = args_conf_->dim_;
    data_ = NULL;
    Init();
}

OutputLayer::~OutputLayer() {
    delete data_;
    data_ = NULL;
}

void OutputLayer::Init() {
    uint64_t datasize = uint64_t(row_) * uint64_t(col_);
    data_ = new float[datasize];
    for (uint32_t i = 0; i < row_; i++) {
        for (uint32_t j = 0; j < col_; j++) {
            uint64_t idx = uint64_t(i) * uint64_t(col_) + uint64_t(j);
            data_[idx] = 0;
        }
    }
}

void OutputLayer::Save() {
    ofstream ofs;
    utils::OpenOutFile(args_conf_->outputdir_, "layer.output."
                + to_string(static_cast<int>(name_)) + "." + class_tag_, ofs);
    utils::WriteLine(ofs, to_string(row_));
    utils::WriteLine(ofs, to_string(col_));
    for (uint32_t i = 0; i < row_; i++) {
        string line = "";
        for (uint32_t j = 0; j < col_; j++) {
            uint64_t idx = uint64_t(i) * uint64_t(col_) + uint64_t(j);
            line += "\t" + to_string(data_[idx]);
        }
        utils::StringTrim(&line);
        utils::WriteLine(ofs, line);
    }
    utils::CloseOutFile(&ofs);
}

void OutputLayer::Load() {
    string output_layer_file = args_conf_->modeldir_ + "/layer.output."
                            + to_string(static_cast<int>(name_)) + "." + class_tag_;
    ifstream fin(output_layer_file);
    assert(fin.is_open());
    string line;
    vector<string> parts;
    utils::GetLine(fin, line);
    assert(utils::StringToNumber(line, &row_));
    utils::GetLine(fin, line);
    assert(utils::StringToNumber(line, &col_));

    uint32_t count = 0;
    uint64_t datasize = row_ * col_;
    data_ = new float[datasize];
    vector<float> vec_num;
    while (utils::GetLine(fin, line)) {
        utils::StringTrim(&line);
        if (line == "") continue;
        utils::StringSplit(line, "\t", parts);
        utils::TrimVector(&parts);
        if (parts.size() != col_) {
            cerr << "output layer not valid col_ number : "
                << parts.size() << endl;
            cerr << "line ("<< (count + 1) <<"): " << line << endl;
            assert(parts.size() == col_);
        }
        utils::ParseVec(parts, vec_num);
        for (uint32_t j = 0; j < vec_num.size(); j++) {
            uint64_t idx = uint64_t(count) * uint64_t(col_) + uint64_t(j);
            data_[idx] = vec_num[j];
        }
        count++;
    }
    if (count != row_) {
        cerr << "output layer vector size(" << count
            << ") != row_("<< row_ <<") " << endl;
        assert(count == row_);
    }
    utils::CloseInFile(&fin);
}
} // namespace knowledgeembedding
