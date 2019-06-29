# 
# Copyright (c) 2019. All rights reserved.
# Author: xinggao1991
# 

CXX = c++
# CXXFLAGS = -pthread -std=c++0x
CXXFLAGS = -pthread -std=gnu++0x
OBJS = basicutil.o argsconf.o fileutil.o hashtable.o matrixutil.o textutil.o vectorutil.o inputlayer.o outputlayer.o model.o embedding.o 
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops
opt: embedding

debug: CXXFLAGS += -g -O0 -fno-inline
debug: embedding

basicutil.o: utils/basicutil.cc utils/basicutil.h
	$(CXX) $(CXXFLAGS) -c utils/basicutil.cc

fileutil.o: utils/fileutil.cc utils/fileutil.h utils/basicutil.h
	$(CXX) $(CXXFLAGS) -c utils/fileutil.cc

argsconf.o: utils/argsconf.cc utils/argsconf.h utils/basicutil.h utils/fileutil.h
	$(CXX) $(CXXFLAGS) -c utils/argsconf.cc

hashtable.o: utils/hashtable.cc utils/hashtable.h utils/argsconf.h utils/basicutil.h utils/fileutil.h utils/textutil.h utils/vectorutil.h
	$(CXX) $(CXXFLAGS) -c utils/hashtable.cc

matrixutil.o: utils/matrixutil.cc utils/matrixutil.h utils/basicutil.h
	$(CXX) $(CXXFLAGS) -c utils/matrixutil.cc

textutil.o: utils/textutil.cc utils/textutil.h utils/basicutil.h utils/argsconf.h
	$(CXX) $(CXXFLAGS) -c utils/textutil.cc

vectorutil.o: utils/vectorutil.cc utils/vectorutil.h utils/basicutil.h
	$(CXX) $(CXXFLAGS) -c utils/vectorutil.cc

inputlayer.o: layers/inputlayer.cc layers/inputlayer.h utils/basicutil.h utils/hashtable.h utils/matrixutil.h utils/textutil.h utils/vectorutil.h
	$(CXX) $(CXXFLAGS) -c layers/inputlayer.cc

outputlayer.o: layers/outputlayer.cc layers/outputlayer.h utils/basicutil.h utils/hashtable.h utils/vectorutil.h
	$(CXX) $(CXXFLAGS) -c layers/outputlayer.cc

model.o: model.cc model.h layers/inputlayer.h layers/outputlayer.h utils/argsconf.h utils/basicutil.h utils/matrixutil.h utils/textutil.h utils/vectorutil.h
	$(CXX) $(CXXFLAGS) -c model.cc

embedding.o: embedding.cc *.h model.h
	$(CXX) $(CXXFLAGS) -c embedding.cc

embedding: $(OBJS) embedding.cc
	$(CXX) $(CXXFLAGS) $(OBJS) main.cc -o embedding	

clean:
	rm -rf *.o embedding
