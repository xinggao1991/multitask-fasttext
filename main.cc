/*
 * Copyright (c) 2019. All rights reserved.
 * Author: xinggao1991
 */
#include "embedding.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Usage: ./embedding <confpath>" << endl;
        exit(1);
    }
    string confpath = reinterpret_cast<char *>(argv[1]);
    knowledgeembedding::Embedding emb;
    emb.InitArgs(confpath);
    emb.MainProcess();
}
