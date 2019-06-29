/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#include "argsconf.h"

namespace knowledgeembedding {

ArgsConf::ArgsConf() {
    params_map_.clear();
    totallinenum_ = 0;
    curlinenum_ = 0;
    curlearnrate_ = 0.05;

    // string
    param_str_["process"] = &process_;
    param_str_["modeldir"] = &modeldir_;
    param_str_["trainfile"] = &trainfile_;
    param_str_["evalfile"] = &evalfile_;
    // int
    param_int_["minlen"] = &minlen_;
    param_int_["maxlen"] = &maxlen_;
    param_int_["maxvocabsize"] = &maxvocabsize_;
    param_int_["maxphrasesize"] = &maxphrasesize_;
    param_int_["minwordfreq"] = &minwordfreq_;
    param_int_["minphrasefreq"] = &minphrasefreq_;
    param_int_["dim"] = &dim_;
    param_int_["ngram"] = &ngram_;
    param_int_["subngram"] = &subngram_;
    param_int_["phrasefreqthreshold"] = &phrasefreqthreshold_;
    param_int_["windowsize"] = &windowsize_;
    param_int_["thread"] = &thread_;
    param_int_["getlossevery"] = &getlossevery_;
    param_int_["evalevery"] = &evalevery_;
    param_int_["epoch"] = &epoch_;
    // float
    param_float_["learnrate"] = &learnrate_;
    param_float_["freqsample"] = &freqsample_;
    param_float_["dropoutkeeprate"] = &dropoutkeeprate_;
    // bool
    param_bool_["useskipgram"] = &useskipgram_;
    param_bool_["usecls"] = &usecls_;
    param_bool_["usepair"] = &usepair_;
}

ArgsConf::~ArgsConf() {
    params_map_.clear();
}

bool ArgsConf::Init(const string &conf_file) {
    cerr << "conf file is : " << conf_file << endl;
    ifstream fin(conf_file.c_str());
    if ( !fin ) {
        cerr << "can not open file : " << conf_file << endl;
        return false;
    }
    string line = "";
    vector<string> parts;
    while (utils::GetLine(fin, line)) {
    utils::StringTrim(&line);
        if (utils::StartWith(line, "#")) {
            continue;
        }
        utils::StringSplit(line, "=", parts);
        utils::TrimVector(&parts);
        if (parts.size() != 2 || parts[0] == "" || parts[1] == "") {
            continue;
        }

        if (param_str_.find(parts[0]) != param_str_.end()) {
            *param_str_[parts[0]] = parts[1];
        }
        else if (param_int_.find(parts[0]) != param_int_.end()) {
            int val = 0;
            if (utils::StringToNumber(parts[1], &val)) {
                *param_int_[parts[0]] = val;
            } else {
                cerr << "Error: param(" << parts[0] << ")" << endl;
                assert(utils::StringToNumber(parts[1], &val));
            }
        }
        else if (param_float_.find(parts[0]) != param_float_.end()) {
            float val = 0;
            if (utils::StringToNumber(parts[1], &val)) {
                *param_float_[parts[0]] = val;
            } else {
                cerr << "Error: param(" << parts[0] << ")" << endl;
                assert(utils::StringToNumber(parts[1], &val));
            }
        }
        else if (param_bool_.find(parts[0]) != param_bool_.end()) {
        utils::StringToLower(&parts[1]);
            *param_bool_[parts[0]] = (parts[1] == "true");
        } else {
        utils::StringToLower(&parts[0]);
        utils::StringToLower(&parts[1]);
            params_map_[parts[0]] = parts[1];
        }
    }
    fin.close();
    return true;
}

void ArgsConf::SetOutputDir() {
    if (outputdir_ == "") {
        int32_t i = 0;
        do {
            string dir = "0000" + to_string(i);
            dir = "model_" + dir.substr(dir.size()-4, 4);
            if (access(dir.c_str(), 0) != 0) {
                outputdir_ = dir;
                if (mkdir(dir.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO) != 0) {
                    cerr << "Error : cannot mkdir " + dir << endl;
                    exit(1);
                }
                i = -100;
            }
            i++;
        } while (i >= 0 && i < 10000);
    }
}

void ArgsConf::PrintArgs() {
    cerr << std::left << setw(30) << "process:" << process_ << endl;
    cerr << std::left << setw(30) << "modeldir:" << modeldir_ << endl;
    cerr << std::left << setw(30) << "trainfile:" << trainfile_ << endl;
    cerr << std::left << setw(30) << "evalfile:" << evalfile_ << endl;
    cerr << std::left << setw(30) << "minlen:" << minlen_ << endl;
    cerr << std::left << setw(30) << "maxlen:" << maxlen_ << endl;
    cerr << std::left << setw(30) << "maxvocabsize:" << maxvocabsize_ << endl;
    cerr << std::left << setw(30) << "maxphrasesize:" << maxphrasesize_ << endl;
    cerr << std::left << setw(30) << "minwordfreq:" << minwordfreq_ << endl;
    cerr << std::left << setw(30) << "minphrasefreq:" << minphrasefreq_ << endl;
    cerr << std::left << setw(30) << "dim:" << dim_ << endl;
    cerr << std::left << setw(30) << "ngram:" << ngram_ << endl;
    cerr << std::left << setw(30) << "subngram:" << subngram_ << endl;
    cerr << std::left << setw(30) << "phrasefreqthreshold:" << phrasefreqthreshold_ << endl;
    cerr << std::left << setw(30) << "windowsize:" << windowsize_ << endl;
    cerr << std::left << setw(30) << "useskipgram:" << (useskipgram_ ? "true" : "false") << endl;
    cerr << std::left << setw(30) << "usecls:" << (usecls_ ? "true" : "false") << endl;
    cerr << std::left << setw(30) << "usepair:" << (usepair_ ? "true" : "false") << endl;
    cerr << std::left << setw(30) << "thread:" << thread_ << endl;
    cerr << std::left << setw(30) << "getlossevery:" << getlossevery_ << endl;
    cerr << std::left << setw(30) << "evalevery:" << evalevery_ << endl;
    cerr << std::left << setw(30) << "epoch:" << epoch_ << endl;
    cerr << std::left << setw(30) << "learnrate:" << learnrate_ << endl;
    cerr << std::left << setw(30) << "freqsample:" << freqsample_ << endl;
    cerr << std::left << setw(30) << "dropoutkeeprate:" << dropoutkeeprate_ << endl;

    for (auto it = params_map_.begin(); it != params_map_.end(); it++) {
        cerr << std::left << setw(30) << (it->first + ":")
            << std::left << setw(20) << it->second << endl;
    }
}

void ArgsConf::CheckArgs() {
    if (useskipgram_ == false && usecls_ == false && usepair_ == false) {
        cerr << "nothing to do with param: \n"
            << "useskipgram_: false\n"
            << "usecls_: false\n"
            << "usepair_: false\n"
            << endl;
        exit(1);
    }
    CheckMin(minlen_, 1, "minlen number error");
    CheckMin(maxlen_, 1, "maxlen number error");
    CheckMin(maxvocabsize_, 10000, "maxvocabsize number error");
    CheckMin(maxphrasesize_, 10000, "maxphrasesize number error");
    CheckMin(minwordfreq_, 1, "minphrasefreq number error");
    CheckMin(minphrasefreq_, 1, "minphrasefreq number error");
    CheckMin(dim_, 1, "dim number error");
    CheckMin(ngram_, 1, "ngram number error");
    CheckMin(subngram_, 1, "subngram number error");
    CheckMin(phrasefreqthreshold_, 1, "phrasefreqthreshold number error");
    CheckMin(windowsize_, 1, "windowsize number error");
    CheckMin(thread_, 1, "thread number error");
    CheckMin(getlossevery_, 1, "get loss every number error");
    CheckMin(evalevery_, 1, "evalevery number error");
    CheckMin(epoch_, 1, "epoch number error");
    CheckMin(learnrate_, static_cast<float>(0.0), "learn rate error");
    CheckMin(freqsample_, static_cast<float>(0.0), "learn rate error");
    CheckMin(dropoutkeeprate_, static_cast<float>(0.0), "learn rate error");
}
template<typename T>
void ArgsConf::CheckMin(const T &param, T minval, const string &err) {
    if (param < minval) {
        cerr << "Error: " << err << endl;
        assert(param >= minval);
    }
}

float ArgsConf::GetParamNum(const string &key) {
    float val = -1;
    if (params_map_.find(key) != params_map_.end()) {
        utils::StringToNumber(params_map_[key], &val);
    }
    return val;
}

string ArgsConf::GetParamStr(const string &key) {
    string k = key;
    utils::StringToLower(&k);
    if (params_map_.find(k) != params_map_.end()) {
        return  params_map_[k];
    }
    return "NULL";
}
} // end of namespace knowledgeembedding
