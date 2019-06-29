/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#include "embedding.h"

namespace knowledgeembedding {
void Embedding::InitArgs(const string &confpath) {
    args_conf_ = make_shared<ArgsConf>();
    assert(args_conf_->Init(confpath));
    cerr << "------------------ user set params -------------------" << endl;
    args_conf_->PrintArgs();
    cerr << "------------------------------------------------------" << endl;
    args_conf_->CheckArgs();
}

void Embedding::AddVocab(const vector<string> &parts,
                         const string &text,
                         HashTable* hash_word,
                         HashTable* hash_phrase,
                         uint64_t &word_counter,
                         uint64_t &phrase_counter) {
    if (utils::StringTrim(text) == "") {
        return;
    }
    // add word list
    vector<string> word_list;
    utils::GetSegedWordList(text, word_list);
    if ((parts[0] == "cls" || parts[0] == "skip" || parts[0] == "pair")
        && (static_cast<int>(word_list.size()) < args_conf_->minlen_
        || static_cast<int>(word_list.size()) > args_conf_->maxlen_)) {
        return;
    }
    word_counter += word_list.size();
    hash_word->AddWord(word_list, true);

    // add ngram list
    vector<string> ngram_list;
    utils::GetNgramWordList(word_list, ngram_list, args_conf_->ngram_);
    phrase_counter += ngram_list.size();
    hash_phrase->AddWord(ngram_list);
}

void Embedding::LoadTrainVocab(bool only_count) {
    cerr << "load train vocab with file : " << args_conf_->trainfile_ << endl;
    ifstream fin(args_conf_->trainfile_);
    assert(fin.is_open());

    HashTable* hash_word =
        new HashTable(args_conf_, args_conf_->maxvocabsize_);
    HashTable* hash_phrase =
        new HashTable(args_conf_, args_conf_->maxphrasesize_);
    vector<string> parts;
    string line;
    vector<string> word_list;
    uint64_t line_counter = 0;
    uint64_t word_counter = 0;
    uint64_t phrase_counter = 0;

    while (utils::GetLine(fin, line)) {
        line_counter++;
        utils::StringTrim(&line);
        utils::StringToLower(&line);
        utils::StringSplit(line, "\t", parts);
        utils::TrimVector(&parts);
        string text = "";
        if (parts.size() == 2 && parts[0] == "skip"
            && args_conf_->useskipgram_) {
            text = parts[1];
        } else if (parts.size() == 5 && parts[0] == "pair"
            && args_conf_->usepair_) {
            string pair_tag = parts[1];
            string label = parts[2];
            text = parts[3] + " . " + parts[4];
            pair_tag_map_[pair_tag] = 1;
            string key = pair_tag + "\t" + label;
            if (pair_tag_count_map_.find(key) == pair_tag_count_map_.end()) {
                pair_tag_count_map_[key] = 1;
            } else {
                pair_tag_count_map_[key] += 1;
            }
        } else if (parts.size() == 4 && parts[0] == "cls"
            && args_conf_->usecls_) {
            text = parts[3];
            string cls_tag = parts[1];
            int32_t label = 0;
            if (!utils::StringToNumber(parts[2], &label) || label < 0) {
                continue;
            }
            if (cls_tag_map_.find(cls_tag) == cls_tag_map_.end()) {
                cls_tag_map_[cls_tag] = label;
                cls_tag_count_map_[cls_tag + "\t" + to_string(label)] = 1;
            } else {
                cls_tag_map_[cls_tag] = max(label, cls_tag_map_[cls_tag]);
            }
            string tag_label = cls_tag + "\t" + to_string(label);
            if (cls_tag_count_map_.find(tag_label)
            == cls_tag_count_map_.end()) {
                cls_tag_count_map_[tag_label] = 1;
            } else {
                cls_tag_count_map_[tag_label] += 1;
            }
        }
        if (text == "" || only_count) {
            continue;
        }
        AddVocab(parts, text, hash_word, hash_phrase,
                 word_counter, phrase_counter);

        if (line_counter % 1000 == 0) {
            cerr.flags(ios::left);
            cerr << "\rRead line: " << setw(12) << line_counter
                << "  words(M): " << setw(10) << (word_counter / 1000000.0)
                << "  phrase(M): " << setw(10) << (phrase_counter / 1000000.0)
                << flush;
        }
    }
    fin.close();
    cerr.flags(ios::left);
    cerr << "\rRead line: " << setw(12) << line_counter
        << "  words(M): " << setw(10) << (word_counter / 1000000.0)
        << "  phrase(M): " << setw(10) << (phrase_counter / 1000000.0)
        << endl;

    hash_word->Rebuild(args_conf_->minwordfreq_);
    hash_phrase->Rebuild(args_conf_->minphrasefreq_);
    cerr << "phrase table size (before filter) : "
     << hash_phrase->wordsize_ << endl;
    hash_phrase->FilterPhraseFromNgram(hash_word, args_conf_);
    cerr << "phrase table size (after filter) : "
     << hash_phrase->wordsize_ << endl;

    hash_table_ = make_shared<HashTable>(args_conf_,
                args_conf_->maxvocabsize_ + args_conf_->maxphrasesize_);
    hash_table_->CombineWordVec(hash_word->wordvec_);
    hash_table_->CombineWordVec(hash_phrase->wordvec_);

    hash_table_->Rebuild(-1);
    hash_table_->InitDiscardTable(args_conf_->freqsample_);
    cerr << "hash_table_.size: " << hash_table_->wordvec_.size() << endl;

    cerr << "After filter words(M): "
        << setw(10) << (hash_word->wordvec_).size() / 1000000.0
        << "  phrase(M): "
        << setw(10) << (hash_phrase->wordvec_).size() / 1000000.0
        << "  combine(M): "
        << setw(10) << (hash_table_->wordvec_).size() / 1000000.0
        << endl;
    args_conf_->totallinenum_ = line_counter;
    delete hash_word;
    hash_word = NULL;
    delete hash_phrase;
    hash_phrase = NULL;
}

void Embedding::LoadEvalExample() {
    if (args_conf_->usecls_ == false
        && args_conf_->usepair_ == false) {
        return;
    }
    cls_eval_.clear();
    cerr << "load eval file : " << args_conf_->evalfile_ << endl;
    ifstream fin(args_conf_->evalfile_);
    assert(fin.is_open());

    vector<string> parts;
    string line;
    vector<int32_t> idx_vec;
    vector<int32_t> idx_vec_1;
    vector<int32_t> idx_vec_2;
    while (utils::GetLine(fin, line)) {
        utils::StringTrim(&line);
        utils::StringToLower(&line);
        utils::StringSplit(line, "\t", parts);
        utils::TrimVector(&parts);

        if (parts.size() == 4 && parts[0] == "cls") {
            string cls_tag = parts[1];
            int32_t label = 0;
            if (!utils::StringToNumber(parts[2], &label) || label < 0) {
                continue;
            }

            string text = parts[3];
            input_layer_->GetIdxVec(text, idx_vec);
            if (idx_vec.size() > 0) {
                pair<vector<int32_t>, string> p(
                            idx_vec, cls_tag + "\t" + to_string(label));
                cls_eval_.push_back(p);
            }
        } else if (parts.size() == 5 && parts[0] == "pair") {
            string pair_tag = parts[1];
            utils::StringTrim(&pair_tag);
            int32_t label = 0;
            if (!utils::StringToNumber(parts[2], &label) || label < 0) {
                continue;
            }

            string text_1 = parts[3];
            utils::StringTrim(&text_1);
            input_layer_->GetIdxVec(text_1, idx_vec_1);

            string text_2 = parts[3];
            utils::StringTrim(&text_2);
            input_layer_->GetIdxVec(text_2, idx_vec_2);

            if (idx_vec_1.size() > 0 && idx_vec_2.size() > 0) {
                pair<vector<int32_t>, vector<int32_t>> text_pair(
                            idx_vec_1, idx_vec_2);
                pair<pair<vector<int32_t>, vector<int32_t>>, string> eval_pair(
                            text_pair, pair_tag + "\t" + to_string(label));
                pair_eval_.push_back(eval_pair);
            }
        }
    }
    fin.close();
    cerr << "cls_eval_ size : " << cls_eval_.size() << endl;
    cerr << "pair_eval_ size : " << pair_eval_.size() << endl;
}

void Embedding::InitModel() {
    input_layer_ = make_shared<InputLayer>(args_conf_, hash_table_);
    skip_model_ = make_shared<Model>(args_conf_, ModelName::skip,
                hash_table_->wordvec_.size(),
        input_layer_, "skip", hash_table_);
    skip_model_->InitNegTable();

    for (auto it = cls_tag_map_.begin(); it != cls_tag_map_.end(); it++) {
        cerr << "initing cls-" << it->first
            << "(" << it->second << ")" << endl;
        shared_ptr<Model> clsi = make_shared<Model>(args_conf_, ModelName::cls,
                    it->second + 1, input_layer_, it->first, hash_table_);
        clsi->InitNegTable(cls_tag_count_map_);
        cls_model_map_[it->first] = clsi;
    }
    for (auto it = pair_tag_map_.begin(); it != pair_tag_map_.end(); it++) {
        shared_ptr<Model> pairi =
        make_shared<Model>(args_conf_, ModelName::pair,
                    2, input_layer_, it->first, hash_table_);
        pair_model_map_[it->first] = pairi;
    }
    cerr << "------------------------------------------------------" << endl;
}

bool Embedding::HasWord(const string &word) {
    int32_t pos = hash_table_->GetWordPos(utils::StringTrim(word));
    return pos >= 0;
}

bool Embedding::GetWordVec(const string &input_word,
                           vector<float> &res_vec,
                           float default_val) {
    res_vec.clear();
    int32_t pos = hash_table_->GetWordPos(utils::StringTrim(input_word));
    if (pos >= 0) {
        for (int32_t i = 0; i < args_conf_->dim_; i++) {
            res_vec.push_back(0);
        }
        input_layer_->GetLayerByIdxs(pos, res_vec);
        return true;
    } else {
        for (int32_t i = 0; i < args_conf_->dim_; i++) {
            res_vec.push_back(default_val);
        }
        return false;
    }
}

void Embedding::GetDistance(const string &input_word,
                            int32_t top_size,
                            const string &query_type_) {
    vector<string> words;
    vector<int32_t> idx_vec;
    vector<string> parts;
    utils::StringSplit(input_word, " ", parts);
    for (uint32_t i = 0; i < parts.size(); i++) {
        utils::StringTrim(&parts[i]);
        if (parts[i] != "") {
            words.push_back(parts[i]);
        }
    }
    hash_table_->GetWordPos(words, idx_vec);
    if (idx_vec.size() >= 1) {
        priority_queue<pair<float, uint32_t>> heap;
        input_layer_->GetNearestNeighbor(idx_vec, heap, query_type_);
        int32_t i = 0;
        while (i < top_size && heap.size() > 0) {
            if (hash_table_->wordvec_[heap.top().second].word != input_word) {
                string word = hash_table_->wordvec_[heap.top().second].word;
                cerr << word << "\t\t" << heap.top().first << endl;
                i++;
            }
            heap.pop();
        }
    }
}

void Embedding::Distance(int32_t top_size) {
    string word;
    cerr << "---------------- " << query_type_
        << "(_all / _word / _phrase) input word('exit' to quit) : " << flush;
    while (utils::GetLine(cin, word)) {
        utils::StringTrim(&word);
        if (word == "_all" || word == "_word" || word == "_phrase") {
            query_type_ = word;
        } else if (word == "exit") {
            return;
        } else {
            GetDistance(word, top_size, query_type_);
        }
        cerr << "---------------- " << query_type_
            << "(_all / _word / _phrase) input word('exit' to quit) : "
            << flush;
    }
}

void Embedding::GetSentenceVec() {
    string sentence;
    vector<string> word_list;
    vector<int32_t> word_idx_vec;
    vector<string> ngram_list;
    vector<int32_t> phrase_idx_vec;

    while (utils::GetLine(cin, sentence)) {
        word_list.clear();
        word_idx_vec.clear();
        ngram_list.clear();
        phrase_idx_vec.clear();

        utils::StringTrim(&sentence);
        utils::GetSegedWordList(sentence, word_list);
        hash_table_->GetWordPos(word_list, word_idx_vec);

        if (args_conf_->ngram_ > 0) {
            utils::GetNgramWordList(word_list, ngram_list, args_conf_->ngram_);
            hash_table_->GetWordPos(ngram_list, phrase_idx_vec);
            word_idx_vec.insert(word_idx_vec.end(),
                   phrase_idx_vec.begin(), phrase_idx_vec.end());
        }
        if (word_idx_vec.size() > 0) {
            vector<float> hiddenVec(args_conf_->dim_, 0);
            input_layer_->GetLayerByIdxs(word_idx_vec, hiddenVec, 1, false);
            cout << sentence << "\t"
                << utils::JoinVector(hiddenVec, " ") << endl;
        }
    }
}

void Embedding::PredictCls(const pair<vector<int32_t>, string> &example,
                           pair<int32_t, float> &predict_res) {
    predict_res.first = -1;
    predict_res.second = 0;
    vector<string> parts;
    vector<pair<int32_t, float>> pred;
    utils::StringSplit(example.second, "\t", parts);
    if (parts.size() == 2
        && cls_model_map_.find(parts[0]) != cls_model_map_.end()) {
        cls_model_map_[parts[0]]->PredictClsScore(example.first, pred);
        if (pred.size() > 0) {
            predict_res.first = pred[0].first;
            predict_res.second = pred[0].second;
        }
    }
}

float Embedding::PredictPair(pair<pair<vector<int32_t>,
                             vector<int32_t>>, string> & example) {
    float score = -1;
    vector<string> parts;
    utils::StringSplit(example.second, "\t", parts);
    if (parts.size() == 2
        && pair_model_map_.find(parts[0]) != pair_model_map_.end()) {
        score = pair_model_map_[parts[0]]->PredictPair(
                    example.first.first, example.first.second);
    }
    return score;
}

void Embedding::Predict(const string &line, string &res) {
    res = line;
    pair<int32_t, float> predict_res(-1, 0);
    vector<string> parts;
    vector<int32_t> idx_vec;
    vector<int32_t> idx_vec_1;
    vector<int32_t> idx_vec_2;
    utils::StringSplit(line, "\t", parts);

    if (parts.size() == 4 && parts[0] == "cls") {
        string text = parts[3];
        string cls_tag = parts[1];
        utils::StringTrim(&cls_tag);
        utils::StringTrim(&text);
        input_layer_->GetIdxVec(text, idx_vec);

        if (idx_vec.size() > 0) {
            pair<vector<int32_t>, string> p(
        idx_vec, cls_tag + "\t" + to_string(-1));
            PredictCls(p, predict_res);
            int32_t pre = predict_res.first;
            parts[2] = to_string(pre);
            parts.push_back(parts[3]);
            parts[3] = to_string(predict_res.second);
        }
        if (parts.size() == 4) {
            parts.push_back(parts[3]);
            parts[3] = "-1";
        }
        res = "";
        for (uint32_t i = 0; i < 5; i++) {
            res += parts[i] + "\t";
        }
        utils::StringTrim(&res);
    }
    if (parts.size() == 5 && parts[0] == "pair") {
        parts[2] = to_string(-1);
        string text_1 = parts[3];
        string text_2 = parts[4];
        string pair_tag = parts[1];
        utils::StringTrim(&pair_tag);
        utils::StringTrim(&text_1);
        utils::StringTrim(&text_2);
        input_layer_->GetIdxVec(text_1, idx_vec_1);
        input_layer_->GetIdxVec(text_2, idx_vec_2);

        if (idx_vec_1.size() > 0 && idx_vec_2.size() > 0) {
            pair<vector<int32_t>, vector<int32_t>> simPair(
                idx_vec_1, idx_vec_2);
            pair<pair<vector<int32_t>, vector<int32_t>>, string> p(
                        simPair, pair_tag + "\t" + to_string(-1));
            parts[2] = to_string(PredictPair(p));
        }
        res = "";
        for (uint32_t i = 0; i < 5; i++) {
            res += parts[i] + "\t";
        }
        utils::StringTrim(&res);
    }
}

void Embedding::Predict() {
    cerr << "predicting ... " << endl;
    string res = "";
    string line = "";
    while (utils::GetLine(cin, line)) {
        utils::StringToLower(&line);
        Predict(line, res);
        if (res != "") {
            cout << res << endl;
        }
    }
}

void Embedding::EvalCls(map<string, pair<int32_t, int32_t>> *result) {
    result->clear();
    vector<string> parts;
    pair<int32_t, float> predict_res(-1, 0);
    for (uint32_t i = 0; i < cls_eval_.size(); i++) {
        int32_t label = -1;
        utils::StringSplit(cls_eval_[i].second, "\t", parts);
        if (parts.size() == 2
            && utils::StringToNumber(parts[1], &label)
            && label >= 0) {
            PredictCls(cls_eval_[i], predict_res);
            int32_t pre = predict_res.first;
            if (pre >= 0) {
                string cls_tag = parts[0];
                if (result->find(cls_tag) == result->end()) {
                    pair<int32_t, int32_t> p(0, 0);
                    p.first += (pre != label) ? 1 : 0;
                    p.second += (pre == label) ? 1 : 0;
                    (*result)[cls_tag] = p;
                } else {
                    (*result)[cls_tag].first += (pre != label) ? 1 : 0;
                    (*result)[cls_tag].second += (pre == label) ? 1 : 0;
                }
            }
        }
    }
}

void Embedding::EvalPair(map<string, pair<int32_t, int32_t>> *result) {
    result->clear();
    vector<string> parts;
    for (uint32_t i = 0; i < pair_eval_.size(); i++) {
        int32_t label = -1;
        utils::StringSplit(pair_eval_[i].second, "\t", parts);
        if (parts.size() == 2 &&
        utils::StringToNumber(parts[1], &label) && label >= 0) {
            string pair_tag = parts[0];
            float score = PredictPair(pair_eval_[i]);
            int32_t pre = (score > 0.5) ? 1 : 0;
            if (result->find(pair_tag) == result->end()) {
                pair<int32_t, int32_t> p(0, 0);
                p.first += (pre != label) ? 1 : 0;
                p.second += (pre == label) ? 1 : 0;
                (*result)[pair_tag] = p;
            } else {
                (*result)[pair_tag].first += (pre != label) ? 1 : 0;
                (*result)[pair_tag].second += (pre == label) ? 1 : 0;
            }
        }
    }
}

void Embedding::PrintEvalInfo(float progress, bool is_eval) {
    string res = "Progress: " + utils::GetFormatStr(progress)
        + " learn-rate: " + utils::GetFormatStr(args_conf_->curlearnrate_)
        + " skip-loss: " + utils::GetFormatStr(skip_model_->GetLoss());
    for (auto it = cls_model_map_.begin(); it != cls_model_map_.end(); it++) {
        res += " cls-" + it->first + "-loss: "
            + utils::GetFormatStr(it->second->GetLoss());
    }
    for (auto it = pair_model_map_.begin(); it != pair_model_map_.end(); it++) {
        res += " pair-" + it->first + "-loss: "
            + utils::GetFormatStr(it->second->GetLoss());
    }
    if (is_eval) {
        string eval_cls_str = "";
        string eval_pair_str = "";
        map<string, pair<int32_t, int32_t>> eval_result;
        if (args_conf_->usecls_) {
            EvalCls(&eval_result);
            for (auto it = eval_result.begin(); it != eval_result.end(); it++) {
                float acc = it->second.second /
                    (it->second.first + it->second.second + 0.0);
                eval_cls_str += " cls-" + it->first + "-acc: "
                            + utils::GetFormatStr(acc);
            }
            utils::StringTrim(&eval_cls_str);
        }
        if (args_conf_->usepair_) {
            EvalPair(&eval_result);
            for (auto it = eval_result.begin(); it != eval_result.end(); it++) {
                float acc = it->second.second /
                    (it->second.first + it->second.second + 0.0);
                eval_pair_str += " pair-" + it->first + "-acc: "
                            + utils::GetFormatStr(acc);
            }
            utils::StringTrim(&eval_pair_str);
        }
        res += "\n\n---" + eval_cls_str + "  " + eval_pair_str  + "\n";
    }
    cerr << res << endl;
}

void Embedding::TrainThread(int32_t thread_id) {
    assert(args_conf_->totallinenum_ > 0);
    assert(args_conf_->epoch_ > 0);
    ifstream fin(args_conf_->trainfile_);
    assert(fin.is_open());

    utils::Seek(&fin, thread_id * utils::Size(&fin) / args_conf_->thread_);
    vector<string> parts;
    string line;
    uint32_t line_counter = 0;
    // jump the line after seek
    utils::ReadLine(&fin, &line, thread_id);
    while (args_conf_->curlinenum_ <
       args_conf_->totallinenum_ * args_conf_->epoch_) {
        line_counter += 1;
        args_conf_->curlinenum_ += 1;
        float progress = args_conf_->curlinenum_ /
                (args_conf_->totallinenum_ * args_conf_->epoch_ * 1.0);
        if (line_counter % args_conf_->getlossevery_ == 0) {
            args_conf_->curlearnrate_ = args_conf_->learnrate_ * (1 - progress);
        }
        utils::ReadLine(&fin, &line, thread_id);
        utils::StringToLower(&line);
        if (line == "") {
            continue;
        }
        utils::StringSplit(line, "\t", parts);
        utils::TrimVector(&parts);
        string text = "";
        string cls_tag = "";
        int32_t label = -1;
        if (parts.size() == 2 && parts[0] == "skip"
            && args_conf_->useskipgram_) {
            text = parts[1];
            if (text == "") continue;
            skip_model_->UpdateSkip(text);
        } else if (parts.size() == 4 && parts[0] == "cls"
                   && args_conf_->usecls_) {
            cls_tag = parts[1];
            if (!utils::StringToNumber(parts[2], &label) || label < 0
                || cls_model_map_.find(cls_tag) == cls_model_map_.end()) {
                continue;
            }
            text = parts[3];
            if (text == "") continue;
            cls_model_map_[cls_tag]->UpdateCls(text, uint32_t(label));
            if (cls_model_map_[cls_tag]->use_as_skip_example_
                && args_conf_->useskipgram_) {
                skip_model_->UpdateSkip(text);
            }
        } else if (parts.size() == 5 && parts[0] == "pair"
                   && args_conf_->usepair_) {
            string pair_tag = parts[1];
            if (!utils::StringToNumber(parts[2], &label) || label < 0
                || pair_model_map_.find(pair_tag) == pair_model_map_.end()) {
                continue;
            }
            string text = parts[3];
            string text_2 = parts[4];
            if (text == "" || text_2 == "") continue;
            pair_model_map_[pair_tag]->UpdatePair(
                text, text_2, uint32_t(label));
            if (pair_model_map_[pair_tag]->use_as_skip_example_
                && args_conf_->useskipgram_) {
                skip_model_->UpdateSkip(text);
                skip_model_->UpdateSkip(text_2);
            }
        }
        if (thread_id == 0) {
            if (line_counter >= static_cast<uint32_t>(args_conf_->evalevery_)) {
                line_counter = 0;
                PrintEvalInfo(progress, true);
            } else if (line_counter % args_conf_->getlossevery_ == 0) {
                PrintEvalInfo(progress, false);
            }
        }
    }
    fin.close();
    if (thread_id == 0) {
        PrintEvalInfo(1, true);
    }
}

void Embedding::Train() {
    vector<thread> threads;
    for (int32_t i = 0; i < args_conf_->thread_; i++) {
        threads.push_back(thread([=]() {
                        TrainThread(i);
                        }));
    }
    for (auto it = threads.begin(); it != threads.end(); it++) {
        it->join();
    }
    cerr << endl;
}

void Embedding::SaveMap(map<string, int32_t> &the_map, const string &file) {
    ofstream ofs;
    utils::OpenOutFile(args_conf_->outputdir_, file, ofs);
    for (auto it = the_map.begin(); it != the_map.end(); it++) {
        utils::WriteLine(ofs, it->first + "\t" + to_string(it->second));
    }
    utils::CloseOutFile(&ofs);
}

void Embedding::Save() {
    args_conf_->SetOutputDir();
    SaveMap(cls_tag_map_, "cls_tag_map_.out");
    SaveMap(cls_tag_count_map_, "cls_tag_count_map_.out");
    SaveMap(pair_tag_map_, "pair_tag_map_.out");
    cerr << "saving skip model ... " << endl;
    skip_model_->Save(true);
    // kb_model_->save(false);
    for (auto it = cls_model_map_.begin(); it != cls_model_map_.end(); it++) {
        cerr << "saving cls model " << it->first << " ... " << endl;
        it->second->Save(false);
    }
    for (auto it = pair_model_map_.begin(); it != pair_model_map_.end(); it++) {
        cerr << "saving pair model " << it->first << " ... " << endl;
        it->second->Save(false);
    }
}

void Embedding::LoadMap(map<string, int32_t> &the_map, const string &file) {
    the_map.clear();
    string dirfile = args_conf_->modeldir_ + "/" + file;
    ifstream fin(dirfile);
    assert(fin.is_open());

    string line;
    vector<string> parts;
    while (utils::GetLine(fin, line)) {
        utils::StringTrim(&line);
        if (line == "") continue;
        utils::StringSplit(line, "\t", parts);
        utils::TrimVector(&parts);
        if (parts.size() >= 2) {
            string key = "";
            for (uint32_t i = 0; i < parts.size()-1; i++) {
                key += parts[i] + "\t";
            }
            utils::StringTrim(&key);
            int32_t num = 0;
            assert(utils::StringToNumber(parts[parts.size()-1], &num));
            the_map[key] = num;
        }
    }
    utils::CloseInFile(&fin);
}

void Embedding::Load() {
    LoadMap(cls_tag_map_, "cls_tag_map_.out");
    LoadMap(cls_tag_count_map_, "cls_tag_count_map_.out");
    LoadMap(pair_tag_map_, "pair_tag_map_.out");
    cerr << "loading hash table ... " << endl;
    hash_table_ = make_shared<HashTable>(args_conf_,
                args_conf_->maxvocabsize_ + args_conf_->maxphrasesize_);
    hash_table_->Load(args_conf_->modeldir_ + "/hashtable.out",
                args_conf_->freqsample_);

    cerr << "loading input layer ... " << endl;
    input_layer_ = make_shared<InputLayer>(args_conf_, hash_table_);
    input_layer_->Load();

    cerr << "loading skip model ... " << endl;
    skip_model_ = make_shared<Model>(args_conf_, ModelName::skip,
                hash_table_->wordvec_.size(),
        input_layer_, "skip", hash_table_);
    skip_model_->Load(cls_tag_count_map_);

    // kb_model_ = make_shared<Model>(args_conf_, ModelName::kb,
    //            0, input_layer_, "kb", hash_table_);
    for (auto it = cls_tag_map_.begin(); it != cls_tag_map_.end(); it++) {
        cerr << "loading cls model " << it->first << " ..." << endl;
        shared_ptr<Model> clsi = make_shared<Model>(args_conf_, ModelName::cls,
                    it->second + 1, input_layer_, it->first, hash_table_);
        // clsi->initNegTable(cls_tag_count_map_);
        clsi->Load(cls_tag_count_map_);
        cls_model_map_[it->first] = clsi;
    }
    for (auto it = pair_tag_map_.begin(); it != pair_tag_map_.end(); it++) {
        cerr << "loading pair model " << it->first << " ... " << endl;
        shared_ptr<Model> pairi = make_shared<Model>(args_conf_,
        ModelName::pair, 2, input_layer_, it->first, hash_table_);
        pairi->Load(pair_tag_count_map_);
        pair_model_map_[it->first] = pairi;
    }
    cerr << "finished load model" << endl;
}

void Embedding::MainProcess() {
    if (args_conf_->modeldir_ != "") {
        Load();
    }
    if (args_conf_->process_ == "train") {
        LoadTrainVocab(args_conf_->modeldir_ != "");
        InitModel();
        LoadEvalExample();
        Train();
        Save();
    } else {
        assert(args_conf_->modeldir_ != "");
        assert(input_layer_ != NULL);
        if (args_conf_->process_ == "distance") {
            Distance();
        } else if (args_conf_->process_ == "predict") {
            Predict();
        } else if (args_conf_->process_ == "sentence_vec") {
            GetSentenceVec();
        } else {
            cerr << "error process : " << args_conf_->process_ << endl;
            exit(1);
        }
    }
}
}  // end of namespace knowledgeembedding
