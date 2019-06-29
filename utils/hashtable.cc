/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */

#include "hashtable.h"

namespace knowledgeembedding {

HashTable::HashTable(shared_ptr<ArgsConf> args_conf, int vocab_size):
    args_conf_(args_conf),
    wordidx_(vocab_size, -1),
    rng_(vocab_size),
    uniform_(0, 1) {
    wordvec_.clear();
    max_vocab_size_ = vocab_size;
    wordsize_ = 0;
    word_filter_freq_ = 1;
}

HashTable::~HashTable() {
    wordvec_.clear();
    wordidx_.clear();
}

void HashTable::GetSubWordList(const string &word,
                               vector<int32_t> &subword_list,
                               uint32_t subngram,
                               bool is_build_hash_table,
                   bool enable_add) {
    subword_list.clear();
    
    // chech if already record subword
    int32_t pos = GetWordPos(word);
    if (pos >= 0 && is_build_hash_table == false) {
        subword_list.insert(subword_list.end(), wordvec_[pos].subwords.begin(),
                    wordvec_[pos].subwords.end());
        return;
    }

    // compute subword from word str
    string token = "<" + word + ">>";
    for (uint32_t i = 0; i < token.size(); i++) {
        // jump the middle char
        if ((token[i]& 0xC0) == 0x80) continue;
        string substr = "";
        uint32_t gram_num = 1;
        uint32_t j = i;
        while (j < token.size() && gram_num <= subngram) {
            if ((token[j] & 0xC0) != 0x80 && substr != "") {
                gram_num++;
                // filte '<' and '>'
                if (j > i+1 || (0 < i && i < token.size()-2)) {
                    if (substr != word) {
                        // cerr << substr << endl;
                        pos = GetWordPos(substr);
                        if (pos >= 0) {
                            subword_list.push_back(pos);
                        } else if (is_build_hash_table && enable_add) {
                            AddWord(substr, 0, false, false, false);
                            pos = GetWordPos(substr);
                            if (pos >= 0) {
                subword_list.push_back(pos);
                            }
                        }
                    }
                }
            }
            substr.push_back(token[j]);
            j++;
        }
    }
}

void HashTable::GetSubWordList(const vector<string> &word_list,
                               const vector<int32_t> &word_idx_list,
                               vector<int32_t> &subword_list,
                               uint32_t subngram) {
    assert(word_idx_list.size() == word_list.size());
    subword_list.clear();
    vector<int32_t> subword;
    for (uint32_t i = 0; i < word_list.size(); i++) {
        if (word_idx_list[i] < 0) {
            continue;
        }
        GetSubWordList(word_list[i], subword, subngram);
        subword_list.insert(subword_list.end(), subword.begin(), subword.end());
    }
}

uint32_t HashTable::GetWordHash(const string &word) {
    uint32_t h = 2166136261;
    for (uint32_t i = 0; i < word.size(); i++) {
        h = h ^ uint32_t(word[i]);
        h = h * 16777619;
    }
    return h % max_vocab_size_;
}

uint32_t HashTable::GetWordIdx(const string &word) {
    uint32_t hval = GetWordHash(word);
    while (wordidx_[hval] >= 0 && wordvec_[wordidx_[hval]].word != word) {
        hval = (hval + 1) % max_vocab_size_;
    }
    return hval;
}

int32_t HashTable::GetWordPos(const string &word) {
    uint32_t idx = GetWordIdx(word);
    if (wordidx_[idx] >= 0 && uint32_t(wordidx_[idx]) < wordvec_.size()) {
        return wordidx_[idx];
    }
    return -1;
}

void HashTable::GetWordPos(const vector<string> &words,
                            vector<int32_t> &idx_vec,
                            bool keepout) {
    idx_vec.clear();
    for (uint32_t i = 0; i < words.size(); i++) {
        int32_t pos = GetWordPos(words[i]);
        if (pos >= 0 || keepout) {
            idx_vec.push_back(pos);
        }
    }
}

float HashTable::GetWordFreq(const string &word) {
    uint32_t idx = GetWordIdx(word);
    if (wordidx_[idx] >= 0) {
        return wordvec_[wordidx_[idx]].freq;
    }
    return 0.0;
}

bool HashTable::HasWord(const string &word) {
    uint32_t idx = GetWordIdx(word);
    return wordidx_[idx] >= 0;
}

void HashTable::AddWord(const string &word_ori,
                        float default_freq,
                        bool enable_rebuild,
                        bool add_freq,
                        bool add_subword) {
    string word = word_ori;
    utils::StringTrim(&word);
    if (word == "") {
        return;
    }
    uint32_t idx = GetWordIdx(word);
    if (wordidx_[idx] >= 0) {
        if (add_freq == true) {
            // add word frequence
            wordvec_[wordidx_[idx]].freq += default_freq;
            // add subword frequence
            if (add_subword) {
                for (auto subidx : wordvec_[wordidx_[idx]].subwords) {
                    wordvec_[subidx].freq += default_freq;
                }
            }
        }
    } else {
        Item item;
        item.word = word;
        item.freq = default_freq;
        if (add_subword) {
            GetSubWordList(word, item.subwords, args_conf_->subngram_, true);
            // cerr << word << "\t" << item.subwords.size() << endl;
            for (auto subidx : item.subwords) {
                wordvec_[subidx].freq += default_freq;
            }
        }

        wordvec_.push_back(item);
        wordidx_[idx] = wordsize_;
        wordsize_++;
        if (wordsize_ > 0.7 * max_vocab_size_ && enable_rebuild) {
            word_filter_freq_++;
            Rebuild(word_filter_freq_);
        }
    }
}

void HashTable::AddWord(const vector<string> &word_list, bool add_subword) {
    for (uint32_t i = 0; i < word_list.size(); i++) {
        AddWord(word_list[i], 1, true, true, add_subword);
    }
}

void HashTable::Rebuild(int min_word_freq) {
    if (min_word_freq > 0) {
        wordvec_.erase(remove_if(wordvec_.begin(), wordvec_.end(),
                        [&](const Item &e) {
                            return static_cast<int>(e.freq) < min_word_freq;
                        }),
            wordvec_.end());
    }
    wordvec_.shrink_to_fit();
    stable_sort(wordvec_.begin(), wordvec_.end(),
                [](const Item &e1, const Item &e2) {
                    return e1.freq > e2.freq;
            });
    fill(wordidx_.begin(), wordidx_.end(), -1);
    wordsize_ = wordvec_.size();
    // rebuild word index
    for (uint32_t i = 0; i < wordvec_.size(); i++) {
        uint32_t idx = GetWordIdx(wordvec_[i].word);
        wordidx_[idx] = i;
    }
    wordsize_ = wordvec_.size();
    // rebuild subword index
    for (uint32_t i = 0; i < wordvec_.size(); i++) {
        if (wordvec_[i].subwords.size() <= 0) {
            continue;
        }
        GetSubWordList(wordvec_[i].word, wordvec_[i].subwords,
                       args_conf_->subngram_, true, false);
    }
    wordsize_ = wordvec_.size();
    train_words_ = 0;
    for (uint32_t i = 0; i < wordvec_.size(); i++) {
        train_words_ += wordvec_[i].freq;
    }
}

void HashTable::InitDiscardTable(float freq_sample) {
    hash_freq_sample_ = freq_sample;
    uint64_t total = 0;
    for (uint32_t i = 0; i < wordvec_.size(); i++) {
        total += uint64_t(wordvec_[i].freq);
    }
    train_words_ = total;
    discard_table_.clear();
    if (total <= 0) {
        return;
    }
    for (uint32_t i = 0; i < wordvec_.size(); i++) {
        float rate = wordvec_[i].freq / total;
        float disrate = sqrt(freq_sample / rate) + freq_sample / rate;
        discard_table_.push_back(disrate);
    }
}

float HashTable::GetDiscardRate(uint32_t wordpos, float boost_freq_sample) {
    if (wordpos >= discard_table_.size()) {
        return 0;
    }
    if (boost_freq_sample < 0.99 || boost_freq_sample > 1.01) {
        if (train_words_ <= 0) {
            return 1; // key all word
        }
        float freq_sample = hash_freq_sample_ * boost_freq_sample;
        float rate = wordvec_[wordpos].freq / train_words_;
        float disrate = sqrt(freq_sample / rate) + freq_sample / rate;
        return disrate;
    }
    return discard_table_[wordpos];
}

void HashTable::RandomDiscard(vector<int32_t> *word_idx_vec,
                              float boost_freq_sample) {
    if (boost_freq_sample < 0.99 || boost_freq_sample > 1.01) {
        if (train_words_ <= 0) {
            return;
        }
        float freq_sample = hash_freq_sample_ * boost_freq_sample;
        word_idx_vec->erase(remove_if(word_idx_vec->begin(), word_idx_vec->end(),
                [&](int32_t idx) {
                    float rate = wordvec_[idx].freq / train_words_;
                    float disrate = sqrt(freq_sample / rate) + freq_sample / rate;
                    return uniform_(rng_) > disrate;
                }),
            word_idx_vec->end());
    } else {
        word_idx_vec->erase(remove_if(word_idx_vec->begin(), word_idx_vec->end(),
                [&](int32_t idx) {
                    return uniform_(rng_) > discard_table_[idx];
                }),
            word_idx_vec->end());
    }
    word_idx_vec->shrink_to_fit();
}

void HashTable::RandomDiscard(vector<string> *words,
                                vector<int32_t> &word_idx_vec,
                                float boost_freq_sample) {
    GetWordPos(*words, word_idx_vec, true);
    if (boost_freq_sample < 0.99 || boost_freq_sample > 1.01) {
        if (train_words_ <= 0) {
            return;
        }
        float freq_sample = hash_freq_sample_ * boost_freq_sample;
        for (uint32_t i = 0; i < word_idx_vec.size(); i++) {
            float rate = wordvec_[word_idx_vec[i]].freq / train_words_;
            float disrate = sqrt(freq_sample / rate) + freq_sample / rate;
            if (uniform_(rng_) > disrate) {
                word_idx_vec[i] = -1;
            }
        }
    } else {
        for (uint32_t i = 0; i < word_idx_vec.size(); i++) {
            if (uniform_(rng_) > discard_table_[i]) {
                word_idx_vec[i] = -1;
            }
        }
    }
}

void HashTable::PrintHashTable() {
    cerr << "------------ index -------------" << endl;
    for (uint32_t i = 0; i < wordidx_.size(); i++) {
        if (wordidx_[i] >= 0) {
            cerr << i << "\t" << wordidx_[i] << endl;
        }
    }
    cerr << "----------- data ---------------" << endl;
    for (uint32_t i = 0; i < wordvec_.size(); i++) {
        cerr << i << "\t" << wordvec_[i].word << "\t" << wordvec_[i].freq << endl;
    }
    cerr << "--------------------------------" << endl;
}

void HashTable::CombineWordVec(const vector<Item> &word_vec) {
    for (uint32_t i = 0; i < word_vec.size(); i++) {
        Item item = word_vec[i];
        AddWord(item.word, item.freq);
        if (item.subwords.size() > 0) {
            int32_t pos = GetWordPos(item.word);
            if (pos >= 0) {
                wordvec_[pos].subwords.push_back(0);
            }
        }
    }
}

void HashTable::FilterPhraseFromNgram(HashTable *word_hash_table,
                                      shared_ptr<ArgsConf> args_conf) {
    vector<string> parts;
    vector<bool> is_discard(wordvec_.size(), false);
    uint32_t total_size = wordvec_.size();
    for (uint32_t i = 0; i < wordvec_.size(); i++) {
        if (i % 100 == 0) {
            cerr << "\rfiltering (" << total_size << ") : "
                << setw(12) << i << flush;
        }
        string word = wordvec_[i].word;
        float freq = wordvec_[i].freq;
        utils::StringSplit(word, "_", parts);
        if (parts.size() == 0) {
            is_discard[i] = true;
        } else if (parts.size() == 1) {
            if (freq < args_conf->minphrasefreq_) {
                is_discard[i] = true;
            }
        } else {
            double word_freq = 1;
            for (uint32_t j = 0; j < parts.size(); j++) {
                double ifreq = word_hash_table->GetWordFreq(parts[j]);
                word_freq *= max(1.0, ifreq);
            }
            double rate = (freq - args_conf->minphrasefreq_)
                    * pow(word_hash_table->train_words_, parts.size()-1)
                    / word_freq;
            if (rate < args_conf->phrasefreqthreshold_) {
                is_discard[i] = true;
            }
        }
    }
    cerr << endl;
    vector<Item> temp_vec;
    temp_vec.insert(temp_vec.end(), wordvec_.begin(), wordvec_.end());
    wordvec_.clear();
    for (uint32_t i = 0; i < temp_vec.size(); i++) {
        if (i % 100 == 0) {
            cerr << "\rfiltered (" << total_size << ") : "
                << setw(12) << i << flush;
        }
        if (is_discard[i] == false) {
            wordvec_.push_back(temp_vec[i]);
        }
    }
    cerr << endl;
    wordvec_.shrink_to_fit();
    Rebuild(-1);
}

void HashTable::Save(shared_ptr<ArgsConf> args_conf) {
    ofstream ofs;
    utils::OpenOutFile(args_conf->outputdir_, "hashtable.out", ofs);
    utils::WriteLine(ofs, to_string(max_vocab_size_));
    utils::WriteLine(ofs, to_string(wordsize_));
    utils::WriteLine(ofs, to_string(word_filter_freq_));
    for (uint64_t i = 0; i < wordvec_.size(); i++) {
        string subwordstr = utils::JoinVector(wordvec_[i].subwords, "|");
        utils::WriteLine(ofs, wordvec_[i].word
                    + "\t" + to_string(wordvec_[i].freq)
                    + "\t" + subwordstr);
        // cerr << wordvec_[i].subwords.size() << "\t" << subwordstr << endl;
    }
    utils::CloseOutFile(&ofs);
}

void HashTable::Load(const string &hash_table_file, float freq_sample) {
    ifstream fin(hash_table_file);
    assert(fin.is_open());

    string line;
    vector<string> parts;
    vector<string> subword_parts;
    utils::GetLine(fin, line);
    utils::StringTrim(&line);
    assert(utils::StringToNumber(line, &max_vocab_size_));
    assert(max_vocab_size_ > 0);

    utils::GetLine(fin, line);
    utils::StringTrim(&line);
    assert(utils::StringToNumber(line, &wordsize_));
    assert(wordsize_ > 0);

    wordsize_ = 0;
    utils::GetLine(fin, line);
    utils::StringTrim(&line);
    assert(utils::StringToNumber(line, &word_filter_freq_));
    assert(word_filter_freq_ > 0);

    uint32_t count = 0;
    while (utils::GetLine(fin, line)) {
        utils::StringTrim(&line);
        if (line == "") continue;
        utils::StringSplit(line, "\t", parts);
        utils::TrimVector(&parts);
        if (parts.size() < 2) {
            cerr << "hash table not valid col number : " << parts.size() << endl;
            cerr << "line ("<< (count+1) <<"): " << line << endl;
            assert(parts.size() >= 2);
        }
        float freq = -1;
        if (!utils::StringToNumber(parts[1], &freq) || freq < 0) {
            cerr << "can not parse hash table freq: " << parts[1] << endl;
            assert(utils::StringToNumber(parts[1], &freq));
            assert(freq >= 0);
            exit(1);
        }
        AddWord(parts[0], freq, false);
        int32_t pos = GetWordPos(parts[0]);
        if (pos >= 0 && parts.size() >= 3) {
            utils::StringSplit(parts[2], "|", subword_parts);
            utils::ParseVec(subword_parts, wordvec_[pos].subwords);
        }
        count++;
    }
    if (count != wordsize_) {
        cerr << "input layer vector size(" << count
            << ") != wordSize("<< wordsize_ <<") " << endl;
        assert(count == wordsize_);
    }
    InitDiscardTable(freq_sample);
    utils::CloseInFile(&fin);
    cerr << "finish load hash table " << endl;
}
} // namespace knowledgeembedding
