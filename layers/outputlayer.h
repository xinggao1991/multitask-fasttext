/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#ifndef KNOWLEDGE_EMBEDDING_LAYERS_OUTPUTLAYER_H
#define KNOWLEDGE_EMBEDDING_LAYERS_OUTPUTLAYER_H

#include <map>
#include <string>
#include <vector>

#include "../utils/basicutil.h"
#include "../utils/hashtable.h"
#include "../utils/vectorutil.h"

namespace knowledgeembedding {
class OutputLayer {
    public:
        OutputLayer(shared_ptr<HashTable> hash_table,
                    shared_ptr<ArgsConf> args_conf,
                    ModelName n,
                    int cls_number,
                    const string &tag);
        ~OutputLayer();
        void Init();
        // save and load
        void Save();
        void Load();

    public:
        float* data_;
        uint32_t row_;
        uint32_t col_;

    private:
        shared_ptr<HashTable> hash_table_;
        shared_ptr<ArgsConf> args_conf_;
        ModelName name_;
        string class_tag_;
}; // OutputLayer
} // namespace knowledgeembedding

#endif // KNOWLEDGE_EMBEDDING_LAYERS_OUTPUTLAYER_H
