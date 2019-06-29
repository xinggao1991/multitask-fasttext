/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */

#ifndef KNOWLEDGE_EMBEDDING_UTILS_HASHTABLE_H
#define KNOWLEDGE_EMBEDDING_UTILS_HASHTABLE_H

#include <string>
#include <vector>

#include "argsconf.h"
#include "basicutil.h"
#include "fileutil.h"
#include "textutil.h"
#include "vectorutil.h"

namespace knowledgeembedding {
struct Item {
    string word;
    vector<int32_t> subwords;
    float freq;
};
class HashTable
{
    public:
        explicit HashTable(shared_ptr<ArgsConf> args_conf, int vocab_size);
        ~HashTable();
        // get sub word list
        void GetSubWordList(const string &word,
                            vector<int32_t> &subword_list,
                            uint32_t subngram,
                            bool is_build_hash_table = false,
                            bool enable_add = true);
        void GetSubWordList(const vector<string> &word_list,
                            const vector<int32_t> &word_idx_list,
                            vector<int32_t> &subword_list,
                            uint32_t subngram);
        // get the hash value of word
        uint32_t GetWordHash(const string &word);
        // get word index number of wordIdx
        // return hash index of word or the pos that can insert word
        uint32_t GetWordIdx(const string &word);
        // filter word that in wordvec, and get the word indexs
        int32_t GetWordPos(const string &word);
        void GetWordPos(const vector<string> &words,
                        vector<int32_t> &idx_vec,
                        bool keepout = false);
        // get frequence of word
        float GetWordFreq(const string &word);
        // judeg whether the word is in this hash table
        bool HasWord(const string &word);
        // add a word to this hash table
        void AddWord(const string &word,
                     float default_freq = 1,
                     bool enable_rebuild = true,
                     bool add_freq = true,
                     bool add_subword = false);
        // add all the word of a list to this hash table
        void AddWord(const vector<string> &word_list, bool add_subword = false);
        // rebuild this hash table, filter low frequence word,
        // and index the high frequent word first
        void Rebuild(int min_word_freq);
        // init discard table
        void InitDiscardTable(float freq_sample);
        // get the number of discard rate
        float GetDiscardRate(uint32_t wordpos, float boost_freq_sample = 1);
        // random discard some high freq word
        void RandomDiscard(vector<int32_t> *word_idx_vec,
                           float boost_freq_sample = 1);
        void RandomDiscard(vector<string> *words,
                           vector<int32_t> &word_idx_vec,
                           float boost_freq_sample = 1);
        // the infos of index and word
        void PrintHashTable();
        // combine a wordvec to this hash table
        void CombineWordVec(const vector<Item> &word_vec);
        // get phrase from ngram
        void FilterPhraseFromNgram(HashTable *word_hash_table,
                                   shared_ptr<ArgsConf> args_conf);

        // save and load
        void Save(shared_ptr<ArgsConf> args_conf);
        void Load(const string &hash_table_file, float freq_sample);

    public:
        vector<Item> wordvec_;
        uint32_t wordsize_ = 0;
        uint64_t train_words_ = 0;

    private:
        shared_ptr<ArgsConf> args_conf_;
        vector<int32_t> wordidx_;
        vector<float> discard_table_;
        uint32_t max_vocab_size_ = 0;
        uint32_t word_filter_freq_ = 0;
        float hash_freq_sample_ = 0;

        minstd_rand rng_;
        uniform_real_distribution<> uniform_;
}; // HashTable
} // namespace knowledgeembedding
#endif // KNOWLEDGE_EMBEDDING_UTILS_HASHTABLE_H
