## multitask-fasttext
What is [multitask-fasttext](https://github.com/xinggao1991/multitask-fasttext)?
* It's not just a copy of fasttext, multitask-fasttext is a totally different multitask-model.
* Compare with supporting only one task (fasttext support cbow/skip/supervise), multitask-fasttext does support to train several task together(**do not need change code**).
* cross-lingual support

## Requirements
* latest g++

## Getting the source code
```
$ git clone https://github.com/xinggao1991/multitask-fasttext
```

## Building code
```
$ ./make.sh
```

## How to support multitask and cross-lingual?
It support multitask and cross-lingual by **data format** and **config**
### Preparing data
Currently, it support three data format.

**format.1**: to run word embedding with skip-gram 
```
$ skip  \t  text
```
**format.2**: to run classification model
```
$ cls  \t  tasktag  \t  label  \t  text
```
**format.3**: to run pair model (similarity or not)
```
$ pair  \t  tasktag  \t  label  \t  text1  \t  text2
```
skip/cls/pair: it is a tag which can be used to distinguish different model

tasktag: it is used to distinguish different task

    for example, if you have three different classification task, you can use:
```
$ cls  \t  a  \t  1  \t  I'm a man ...
$ cls  \t  b  \t  5  \t  The movie is very good ...
$ cls  \t  c  \t  2  \t  NLP is very popular now ...
```

label: a number from 0-n. (if use 'pair' model, it is 0/1)

text: seged sentence. you can use [sentencepiece](https://github.com/google/sentencepiece) to support cross-lingual

### Multi-lingual support
If want use cross-lingual support, the **pair model and skip model** can be useful with following config:
```
$ skip  \t  language1
$ skip  \t  language2
$ pair  \t  task1  \t  label  \t  langual1  \t langual2
```
which means embedding different language respectively, and mapping them into the same vector space.

### Writing config
There is an example in the conf directory. Specially:
* set **useskipgram / usecls / usepair** with the value **true** to enable different model.
* set **process** with the value **train / preict / ...** to start different process.
* set trainfile and evalfile

## Training model
```
$ ./embedding ./conf/embedding.conf
```
set process=train

## Testing model
```
$ ./embedding ./conf/embedding.conf
```
set process=predict and set modeldir

