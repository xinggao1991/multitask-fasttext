/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#include "inputlayer.h"

namespace knowledgeembedding {

InputLayer::InputLayer(shared_ptr<ArgsConf> args_conf,
                    shared_ptr<HashTable> hash_table): rng_(1), uniform_(0, 1) {
    args_conf_ = args_conf;
    hash_table_ = hash_table;
    col_ = uint32_t(args_conf_->dim_);
    row_ = uint32_t((hash_table_->wordvec_).size());
    data_ = NULL;
    Init();
}

InputLayer::~InputLayer() {
    delete data_;
    data_ = NULL;
}

void InputLayer::Init() {
    uint64_t datasize = uint64_t(row_) * uint64_t(col_);
    data_ = new float[datasize];
    minstd_rand rng_(1);
    uniform_real_distribution<> init_uniform(-1.0/col_, 1.0/col_);
    for (uint32_t i = 0; i < row_; i++) {
        for (uint32_t j = 0; j < col_; j++) {
            uint64_t idx = uint64_t(i) * uint64_t(col_) + uint64_t(j);
            data_[idx] = init_uniform(rng_);
        }
    }
}

void InputLayer::GetIdxVec(const string &text,
                           vector<int32_t> &idx_vec,
                           float boost_freq_sample,
                           bool usephrase) {
    idx_vec.clear();
    vector<string> word_list;
    vector<int32_t> word_idx_vec;
    vector<int32_t> subword_idx_vec;
    vector<int32_t> phrase_idx_vec;

    utils::GetSegedWordList(text, word_list);
    if (static_cast<int>(word_list.size()) < args_conf_->minlen_
        || static_cast<int>(word_list.size()) > args_conf_->maxlen_) {
        return;
    }

    hash_table_->RandomDiscard(&word_list, word_idx_vec, boost_freq_sample);
    hash_table_->GetSubWordList(word_list, word_idx_vec,subword_idx_vec, args_conf_->subngram_);
    word_idx_vec.erase(remove_if(word_idx_vec.begin(), word_idx_vec.end(),
                    [&](int32_t idx){
                        return idx < 0;
                    }), word_idx_vec.end());
    word_idx_vec.shrink_to_fit();

    if (usephrase && args_conf_->ngram_ > 1) {
        vector<string> ngram_list;
        utils::GetNgramWordList(word_list, ngram_list, args_conf_->ngram_);
        hash_table_->GetWordPos(ngram_list, phrase_idx_vec);
        hash_table_->RandomDiscard(&phrase_idx_vec, boost_freq_sample);
    }

    idx_vec.insert(idx_vec.end(), word_idx_vec.begin(), word_idx_vec.end());
    idx_vec.insert(idx_vec.end(), subword_idx_vec.begin(), subword_idx_vec.end());
    idx_vec.insert(idx_vec.end(), phrase_idx_vec.begin(), phrase_idx_vec.end());
}

void InputLayer::GetLayerByIdxs(int32_t word_idx,
                                vector<float> &layer,
                                float rate) {
    assert(layer.size() == col_);
    if (word_idx < 0 || static_cast<uint32_t>(word_idx) >= row_) {
        return;
    }
    utils::MatrixGetVec(&layer, data_, word_idx, args_conf_->dim_, rate);
}
void InputLayer::GetLayerByIdxs(const vector<int32_t> &word_idx_vec,
                                vector<float> &layer,
                                float boost_freq_sample,
                                bool use_discard_rate) {
    assert(layer.size() == col_);
    float size = static_cast<float>(word_idx_vec.size());
    for (uint32_t i = 0; i < word_idx_vec.size(); i++) {
        if (use_discard_rate && uniform_(rng_) >
            hash_table_->GetDiscardRate(word_idx_vec[i], boost_freq_sample)) {
            continue;
        }
        GetLayerByIdxs(word_idx_vec[i], layer);
        size++;
    }
    if (size > 1) {
        for (uint32_t i = 0; i < col_; i++) {
            layer[i] /= size;
        }
    }
}

void InputLayer::UpdateData(int32_t input_idx,
                            vector<float> &add_vec,
                            float rate) {
    if (static_cast<uint32_t>(input_idx) >= row_) {
        return;
    }
    utils::MatrixAdd(data_, input_idx, add_vec, args_conf_->dim_, rate);
}
void InputLayer::UpdateData(const vector<int32_t> &input_vec,
                            vector<float> &add_vec,
                            float rate) {
    if (input_vec.size() <= 0) {
        return;
    }
    rate = rate / input_vec.size();
    for (uint32_t i = 0; i < input_vec.size(); i++) {
        UpdateData(input_vec[i], add_vec, rate);
    }
}

// get top nearest
void InputLayer::GetNearestNeighbor(const vector<int32_t> &idx_vec,
                                    priority_queue<pair<float, uint32_t>> &heap,
                                    const string &query_type) {
    if (idx_vec.size() == 0) {
        return;
    }
    vector<float> query_vec(col_, 0);
    for (uint32_t i = 0; i < idx_vec.size(); i++) {
        if (idx_vec[i] < 0 || static_cast<uint32_t>(idx_vec[i]) >= row_) {
            continue;
        }
        for (uint32_t j = 0; j < col_; j++) {
            uint64_t idx = uint64_t(idx_vec[i]) * uint64_t(col_) + uint64_t(j);
            query_vec[j] += data_[idx];
        }
    }
    float query_norm = utils::Norm(query_vec);
    query_norm = (abs(query_norm) < 1e-6) ? 1 : query_norm;
    for (uint32_t i = 0; i < row_; i++) {
        if (i >= hash_table_->wordvec_.size()) {
            break;
        }
        string word = hash_table_->wordvec_[i].word;
        if (query_type == "_word" && word.find("_") != string::npos) {
            continue;
        }
        if (query_type == "_phrase" && word.find("_") == string::npos) {
            continue;
        }
        float dnorm = utils::MatrixNorm(data_, i, col_);
        dnorm = (abs(dnorm) < 1e-6) ? 1 : dnorm;
        float val = utils::MatrixDowRow(data_, i, query_vec, col_);
        heap.push(make_pair(val / query_norm / dnorm, i));
    }
}

void InputLayer::Save() {
    ofstream ofs;
    utils::OpenOutFile(args_conf_->outputdir_, "layer.input", ofs);
    utils::WriteLine(ofs, to_string(row_));
    utils::WriteLine(ofs, to_string(col_));
    for (uint32_t i = 0; i < row_; i++) {
        string line = "";
        for (uint32_t j = 0; j < col_; j++) {
            uint64_t idx = uint64_t(i) * uint64_t(col_) + uint64_t(j);
            line += to_string(data_[idx]) + "\t";
        }
        utils::StringTrim(line);
        utils::WriteLine(ofs, line);
    }
    utils::CloseOutFile(&ofs);
}

void InputLayer::Load() {
    string input_layer_file = args_conf_->modeldir_ + "/layer.input";
    ifstream fin(input_layer_file);
    assert(fin.is_open());

    string line;
    vector<string> parts;
    utils::GetLine(fin, line);
    utils::StringTrim(&line);
    assert(utils::StringToNumber(line, &row_));
    assert(row_ > 0);

    utils::GetLine(fin, line);
    utils::StringTrim(&line);
    assert(utils::StringToNumber(line, &col_));
    assert(col_ > 0);

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
            cerr << "input layer not valid col_ number : "
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
        cerr << "input layer vector size(" << count
            << ") != row_("<< row_ <<") " << endl;
        assert(count == row_);
    }
    utils::CloseInFile(&fin);
}
} // namespace knowledgeembedding
