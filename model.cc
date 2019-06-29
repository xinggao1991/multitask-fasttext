/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#include "model.h"

namespace knowledgeembedding {

Model::Model(shared_ptr<ArgsConf> args_conf,
             ModelName n,
             int32_t cls_number,
             shared_ptr<InputLayer> input_layer,
             const string &tag,
             shared_ptr<HashTable> hash_table):
    args_conf_(args_conf),
    name_(n),
    loss_fun_(LossFun::ng),
    cls_number_(cls_number),
    class_tag_(tag),
    rng_(static_cast<int>(n) + cls_number),
    uniform_(0, 1) {
    hash_table_ = hash_table;
    input_layer_ = input_layer;
    output_layer_ = make_shared<OutputLayer>(hash_table, args_conf, n, cls_number, tag);
    Init();
}
Model::~Model() {
    delete sigmoid_table_;
    neg_table_.clear();
}

void Model::Init() {
    assert(cls_number_ > 1);
    if (name_ == ModelName::cls) {
        float val = args_conf_->GetParamNum("cls_" + class_tag_ + "_boost");
        boost_ = val;
        boost_freq_sample_ = args_conf_->GetParamNum("cls_" + class_tag_ + "_boost_freq_sample");
        val = args_conf_->GetParamNum("cls_" + class_tag_ + "_neg_sample");
        neg_sample_ = val >= 0 ? int32_t(val) : 5;
        use_as_skip_example_ = args_conf_->GetParamStr(
                    "cls_" + class_tag_ + "_use_as_skip_example") == "true" ?
                    true : false;
        if (args_conf_->GetParamStr("cls_" + class_tag_ + "_loss") == "softmax") {
            loss_fun_ = LossFun::softmax;
        }
    } else if (name_ ==  ModelName::pair) {
        float val = args_conf_->GetParamNum("pair_" + class_tag_ + "_boost");
        boost_ = val;
        boost_freq_sample_ = args_conf_->GetParamNum(
                    "pair_" + class_tag_ + "_boost_freq_sample");
        use_as_skip_example_ = args_conf_->GetParamStr(
                    "pair_" + class_tag_ + "_use_as_skip_example") == "true" ?
                    true : false;
    } else if (name_ == ModelName::skip) {
        float val = args_conf_->GetParamNum("skipgram_boost");
        boost_ = val >= 0 ? val : 1.0;
        boost_freq_sample_ = args_conf_->GetParamNum("skipgram_boost_freq_sample");
        val = args_conf_->GetParamNum("skipgram_neg_sample");
        neg_sample_ = val >= 0 ? int32_t(val) : 5;
    }
    string name_str = "";
    if (name_ == ModelName::cls) name_str = "cls";
    if (name_ == ModelName::pair) name_str = "pair";
    cerr << "---------------- model " << name_str
        << " " << class_tag_ << " params ------------------\n"
        << std::left << setw(30) << "loss: "
        << (loss_fun_ == LossFun::softmax ? "softmax" : "ng") << "\n"
        << std::left << setw(30) << "cls_number_:" << cls_number_ << "\n"
        << std::left << setw(30) << "boost_:" << boost_ << "\n"
        << std::left << setw(30) << "boost_freq_sample_:"
        << boost_freq_sample_ << "\n"
        << std::left << setw(30) << "real output size:" << output_layer_->row_
        << endl;
    if (name_ == ModelName::cls || name_ == ModelName::pair) {
        cerr << std::left << setw(30) << "use_as_skip_example_:"
            << (use_as_skip_example_ ? "true" : "false") << endl;
    }
    InitSigmoid();
    InitLog();
}

void Model::RandomMask(vector<float> &mask_vec) {
    assert(args_conf_->dropoutkeeprate_ > 0);
    assert(args_conf_->dropoutkeeprate_ <= 1);
    assert(mask_vec.size() == args_conf_->dim_);
    for (uint32_t i = 0; i < mask_vec.size(); i++) {
        float r = uniform_(rng_);
        mask_vec[i] = r < args_conf_->dropoutkeeprate_ ? 1 / args_conf_->dropoutkeeprate_ : 0;
    }
    // utils::Print(mask_vec);
    // exit(1);
}

float Model::GetRandFloat(float min, float max) {
    assert(max > min);
    float r = uniform_(rng_);
    r = r * (max - min) + min;
    return r;
}

int32_t Model::GetRandInt(int32_t min, int32_t max) {
    if (min >= max) {
        return min;
    }
    rand_long_ = rand_long_ * (static_cast<uint64_t>(25214903917)) + 11;
    return (rand_long_ >> 16) % (max - min + 1) + min;
}

void Model::InitNegTable() {
    float sum = 0;
    for (uint32_t i = 0; i < hash_table_->wordvec_.size(); i++) {
        // only use seged word (no subword or ngramstr)
        if (hash_table_->wordvec_[i].subwords.size() <= 0) {
            continue;
        }
        sum += pow(hash_table_->wordvec_[i].freq, 0.5);
    }
    for (uint32_t i = 0; i < hash_table_->wordvec_.size(); i++) {
        // only use seged word (no subword or ngramstr)
        if (hash_table_->wordvec_[i].subwords.size() <= 0) {
            continue;
        }
        float c = pow(hash_table_->wordvec_[i].freq, 0.5);
        for (uint32_t j = 0; j < c * NEG_TABLE_SIZE / sum; j++) {
            neg_table_.push_back(i);
        }
    }
    shuffle(neg_table_.begin(), neg_table_.end(), rng_);
}
void Model::InitNegTable(const map<string, int32_t> &tag_count_map) {
    cerr << "init neg table with map.size: " << tag_count_map.size() << endl;
    vector<string> parts;
    float sum = 0;
    for (auto it = tag_count_map.begin(); it != tag_count_map.end(); it++) {
        utils::StringSplit(it->first, "\t", parts);
        string tag = parts[0];
        if (tag == class_tag_) {
            sum += pow(it->second, 0.5);
        }
    }
    assert(sum > 0);
    for (auto it = tag_count_map.begin(); it != tag_count_map.end(); it++) {
        utils::StringSplit(it->first, "\t", parts);
        if (parts.size() != 2) {
            continue;
        }
        int32_t label = -1;
        string tag = parts[0];
        if (!utils::StringToNumber(parts[1], &label)) {
            continue;
        }
        if (label < 0) {
            continue;
        }
        if (tag == class_tag_) {
            float num = pow(it->second, 0.5);
            for (int32_t i = 0; i < (num * NEG_TABLE_SIZE / sum); i++) {
                neg_table_.push_back(uint32_t(label));
            }
        }
    }
    shuffle(neg_table_.begin(), neg_table_.end(), rng_);
}

uint32_t Model::GetNegativeLabel(uint32_t positive_label) {
    uint32_t neg_label = positive_label;
    do {
        neg_table_index_ = (neg_table_index_ + 1) % neg_table_.size();
        neg_label = neg_table_[neg_table_index_];
    } while (neg_label == positive_label);
    return neg_label;
}

uint32_t Model::GetNegativeLabel(const map<uint32_t, uint32_t> &pos_counter,
                                 uint32_t max_find_times) {
    uint32_t neg_label = 0;
    uint32_t find_times = 0;
    do {
        neg_table_index_ = (neg_table_index_ + 1) % neg_table_.size();
        neg_label = neg_table_[neg_table_index_];
        find_times++;
    } while (pos_counter.find(neg_label) != pos_counter.end()
             && find_times < max_find_times);
    return neg_label;
}

void Model::InitSigmoid() {
    sigmoid_table_ = new float[SIGMOID_TABLE_SIZE + 1];
    for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
        float x = static_cast<float>(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
        sigmoid_table_[i] = 1.0 / (1.0 + std::exp(-x));
    }
}

float Model::GetSigmoid(float x) {
    if (x < -MAX_SIGMOID) {
        return 0.0;
    } else if (x > MAX_SIGMOID) {
        return 1.0;
    } else {
        int i = static_cast<int>((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
        return sigmoid_table_[i];
    }
}

void Model::InitLog() {
    log_table_ = new float[LOG_TABLE_SIZE];
    for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
        float x = (static_cast<float>(i) + 1e-5) / LOG_TABLE_SIZE;
        log_table_[i] = std::log(x);
    }
}

float Model::GetLog(float x) {
    if (x > 1.0) {
        return 0.0;
    }
    int i = static_cast<int>(x * LOG_TABLE_SIZE);
    return log_table_[i];
}

float Model::GetLoss() {
    if (total_loss_num_ <= 1) {
        return -1;
    } else {
        float res = total_loss_value_ / total_loss_num_;
        return res;
    }
}

void Model::SoftMax(const vector<float> &hidden_vec,
                    uint32_t target,
                    vector<float> &grad,
                    vector<float> &mask_vec) {
    // compute yi = exp(i) / sum(exp(j))
    vector<float> mul_vec(output_layer_->row_, 0);
    utils::MatrixMul(output_layer_->data_, hidden_vec, mask_vec, mul_vec,
                     output_layer_->row_, output_layer_->col_);
    float maxval = mul_vec[0];
    for (auto item : mul_vec) {
        maxval = max(maxval, item);
    }
    float z = 0;
    for (uint32_t i = 0; i < mul_vec.size(); i++) {
        mul_vec[i] = exp(mul_vec[i] - maxval);
        z += mul_vec[i];
    }
    for (uint32_t i = 0; i < mul_vec.size(); i++) {
        mul_vec[i] /= z;
    }

    // update output and get grad
    for (uint32_t i = 0; i < mul_vec.size(); i++) {
        float label = (i == target) ? 1.0 : 0.0;
        float alpha = boost_ * args_conf_->curlearnrate_ * (label - mul_vec[i]);
        utils::MatrixGetVec(&grad, output_layer_->data_, i,
                            args_conf_->dim_, alpha, mask_vec);
        utils::MatrixAdd(output_layer_->data_, i, hidden_vec,
                         args_conf_->dim_, alpha, mask_vec);
    }
    total_loss_value_ += -GetLog(mul_vec[target]);
    total_loss_num_ += 1;
}

void Model::UpdateBatch(const vector<float> &hidden_vec,
                        uint32_t output,
                        uint32_t label,
                        vector<float> &grad,
                        vector<float> &mask_vec) {
    float dow_val = utils::MatrixDowRow(output_layer_->data_, output,
                        hidden_vec, args_conf_->dim_, mask_vec);
    float score = GetSigmoid(dow_val);
    double loss = (label == 1) ? -GetLog(score) : -GetLog(1.0 - score);
    total_loss_value_ += loss;

    float alpha = boost_ * args_conf_->curlearnrate_ * (static_cast<float>(label) - score);
    utils::MatrixGetVec(&grad, output_layer_->data_, output,
                        args_conf_->dim_, alpha, mask_vec);
    utils::MatrixAdd(output_layer_->data_, output, hidden_vec,
                     args_conf_->dim_, alpha, mask_vec);
}

void Model::UpdateNeg(const vector<int32_t> &input_vec,
                      const vector<float> &hidden_vec,
                      vector<float> &grad,
                      vector<float> &mask_vec,
                      uint32_t output,
                      map<uint32_t, uint32_t> &pos_counter,
                      bool use_neg,
                      uint32_t label) {
    if (input_vec.size() == 0) {
        return;
    }
    UpdateBatch(hidden_vec, output, label, grad, mask_vec);
    if (use_neg) {
        for (int32_t neg = 0; neg < neg_sample_; neg++) {
            uint32_t negOutput = GetNegativeLabel(pos_counter);
            UpdateBatch(hidden_vec, negOutput, 0, grad, mask_vec);
        }
    }
    total_loss_num_ += 1;
}

void Model::UpdateSkip(const string &text) {
    assert(args_conf_->ngram_ >= 1);
    vector<string> word_list;
    utils::GetSegedWordList(text, word_list);
    if (static_cast<int>(word_list.size()) < args_conf_->minlen_
        || static_cast<int>(word_list.size()) > args_conf_->maxlen_) {
        return;
    }
    map<uint32_t, uint32_t> pos_counter;
    vector<int32_t> input_vec;
    vector<int32_t> word_idx_vec;
    hash_table_->RandomDiscard(&word_list, word_idx_vec, boost_freq_sample_);
    string ngram_str;
    
    for (uint32_t i = 0; i < word_list.size(); i++) {
        for (uint32_t n = 1; n <= args_conf_->ngram_; n++) {
            uint32_t end = i + n - 1;
            if (end >= word_list.size()) {
                break;
            }
            ngram_str = utils::GetNgramWord(word_list, word_idx_vec, i, end);
            if (ngram_str == "") {
                break;
            }
            int32_t ngram_pos = hash_table_->GetWordPos(ngram_str);
            if (ngram_pos < 0) {
                break;
            }

            // prepare input vector
            input_vec.clear();
            input_vec.push_back(ngram_pos);
            if (n == 1) {
                input_vec.insert(input_vec.end(),
                    hash_table_->wordvec_[ngram_pos].subwords.begin(),
                    hash_table_->wordvec_[ngram_pos].subwords.end());
            }
            // use hidden vector
            vector<float> hidden_vec(args_conf_->dim_, 0);
            input_layer_->GetLayerByIdxs(input_vec, hidden_vec, 1);
            vector<float> grad(args_conf_->dim_, 0);
            
            // random drop out
            vector<float> mask_vec(args_conf_->dim_, 0);
            RandomMask(mask_vec);

            pos_counter.clear();
            pos_counter[ngram_pos] = 1;
            int32_t boundary = GetRandInt(1, args_conf_->windowsize_);

            // update left
            int32_t lbound = boundary;
            for (int32_t j = i-1; j >= 0 && j >= i-lbound; j--) {
                if (word_idx_vec[j] < 0) {
                    lbound++;
                    continue;
                }
                UpdateNeg(input_vec, hidden_vec, grad, mask_vec,
                          word_idx_vec[j], pos_counter);
            }

            // update right
            uint32_t rbound = boundary;
            for (uint32_t j = end+1; j < word_idx_vec.size()
                                && j <= end+rbound; j++) {
                if (word_idx_vec[j] < 0) {
                    rbound++;
                    continue;
                }
                UpdateNeg(input_vec, hidden_vec, grad, mask_vec,
                        word_idx_vec[j], pos_counter);
            }

            // update grad to input layer
            input_layer_->UpdateData(input_vec, grad);
        }
    }
}

void Model::UpdateCls(const string &text, uint32_t label) {
    vector<int32_t> word_idx_vec;
    input_layer_->GetIdxVec(text, word_idx_vec);

    if (word_idx_vec.size() < 1 || boost_ <= 0.000001) {
        return;
    }
    assert(label < cls_number_);
    vector<float> hidden_vec(args_conf_->dim_, 0);
    input_layer_->GetLayerByIdxs(word_idx_vec, hidden_vec, 1);
    vector<float> grad(args_conf_->dim_, 0);
    map<uint32_t, uint32_t> pos_counter;
    pos_counter[label] = 1;

    // random drop out
    vector<float> mask_vec(args_conf_->dim_, 0);
    RandomMask(mask_vec);

    if (loss_fun_ == LossFun::softmax) {
        SoftMax(hidden_vec, label, grad, mask_vec);
    } else {
        UpdateNeg(word_idx_vec, hidden_vec, grad, mask_vec, label, pos_counter);
    }
    input_layer_->UpdateData(word_idx_vec, grad);
}

void Model::UpdatePair(const string &text_1,
                        const string &text_2,
                        uint32_t label) {
    vector<int32_t> word_idx_vec_1;
    input_layer_->GetIdxVec(text_1, word_idx_vec_1);
    vector<int32_t> word_idx_vec_2;
    input_layer_->GetIdxVec(text_2, word_idx_vec_2);

    if (word_idx_vec_1.size() < 1 || word_idx_vec_2.size() < 1
        || boost_ <= 0.000001) {
        return;
    }
    assert(label < cls_number_);
    vector<float> hidden_vec_1(args_conf_->dim_, 0);
    vector<float> hidden_vec_2(args_conf_->dim_, 0);
    input_layer_->GetLayerByIdxs(word_idx_vec_1, hidden_vec_1, 1);
    input_layer_->GetLayerByIdxs(word_idx_vec_2, hidden_vec_2, 1);

    float dow_val = utils::DowRow(hidden_vec_1, hidden_vec_2);
    float score = GetSigmoid(dow_val);
    double loss = (label == 1) ? -GetLog(score) : -GetLog(1.0 - score);
    total_loss_value_ += loss;
    total_loss_num_ += 1;

    float alpha = boost_ * args_conf_->curlearnrate_ * (static_cast<float>(label) - score);
    input_layer_->UpdateData(word_idx_vec_1, hidden_vec_2, alpha);
    input_layer_->UpdateData(word_idx_vec_2, hidden_vec_1, alpha);
}

float Model::PredictPair(const vector<int32_t> &input_idx_vec_1,
                         const vector<int32_t> &input_idx_vec_2) {
    vector<float> hidden_vec_1(args_conf_->dim_, 0);
    vector<float> hidden_vec_2(args_conf_->dim_, 0);
    input_layer_->GetLayerByIdxs(input_idx_vec_1, hidden_vec_1, 1);
    input_layer_->GetLayerByIdxs(input_idx_vec_2, hidden_vec_2, 1);
    float dow_val = utils::DowRow(hidden_vec_1, hidden_vec_2);
    float score = GetSigmoid(dow_val);
    return score;
}

int32_t Model::PredictCls(const vector<int32_t> &input_idx_vec) {
    vector<float> hidden_layer(args_conf_->dim_, 0);
    // input_layer_->GetLayerByIdxs(input_idx_vec, hidden_layer, boost_freq_sample_, true);
    input_layer_->GetLayerByIdxs(input_idx_vec, hidden_layer, 1);
    float max_score = -1000000;
    int32_t label = -1;
    for (uint32_t i = 0; i < output_layer_->row_; i++) {
        float score = utils::MatrixDowRow(output_layer_->data_, i,
                                          hidden_layer, args_conf_->dim_);
        if (score > max_score) {
            max_score = score;
            label = int32_t(i);
        }
    }
    return label;
}

void Model::PredictClsScore(const vector<int32_t> &input_idx_vec,
                              vector<pair<int32_t, float>> &predict) {
    predict.clear();
    float max_score = -1000000;
    vector<float> hidden_layer(args_conf_->dim_, 0);
    // input_layer_->GetLayerByIdxs(input_idx_vec, hidden_layer, boost_freq_sample_, true);
    input_layer_->GetLayerByIdxs(input_idx_vec, hidden_layer, 1);
    for (uint32_t i = 0; i < output_layer_->row_; i++) {
        float score = utils::MatrixDowRow(output_layer_->data_, i,
                                          hidden_layer, args_conf_->dim_);
        pair<int32_t, float> p(int32_t(i), score);
        predict.push_back(p);
        max_score = (i == 0) ? score : max(max_score, score);
    }
    float sum = 0;
    for (uint32_t i = 0; i < predict.size(); i++) {
        predict[i].second = exp(predict[i].second - max_score);
        sum += predict[i].second;
    }
    if (sum > 0) {
        for (uint32_t i = 0; i < predict.size(); i++) {
            predict[i].second /= sum;
        }
    }
    stable_sort(predict.begin(), predict.end(),
        [](const pair<int32_t, float> &p1, const pair<int32_t, float> &p2) {
            return p1.second > p2.second;
        }
    );
}

void Model::Save(bool save_common_data) {
    if (save_common_data) {
        hash_table_->Save(args_conf_);
        input_layer_->Save();
    }
    if (name_ == ModelName::pair
       || name_ == ModelName::cls
       || (name_ == ModelName::skip && args_conf_->useskipgram_)) {
        output_layer_->Save();
    }
}
void Model::Load(const map<string, int32_t> &tag_count_map) {
    if (name_ == ModelName::cls
       || (name_ == ModelName::skip && args_conf_->useskipgram_)) {
        output_layer_->Load();
    }
    if (name_ == ModelName::skip) {
        InitNegTable();
    } else if (name_ == ModelName::cls) {
        InitNegTable(tag_count_map);
    } else if (name_ == ModelName::pair) {
        cerr << "pair do not need initNegTable" << endl;
    }
}
} // namespace knowledgeembedding
