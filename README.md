Fork of Guillaume Lample's NER tagger, which was originally presented in https://arxiv.org/abs/1603.01360 .
This fork adds confidence estimates to the output of tagger.py, which are obtained using the constrained forward-backward algorithm introduced by http://www.aclweb.org/anthology/N04-4028 .

The output format of ```tagger.py``` is changed to \<token\>\_\_O if no entity was predicted, or \<token\>\_\_\<tag\>\_\_\<confidence\> for predicted entities. Scores are per entities and a higher score generally implies a higher confidence of the model.

A utility script ``` 	drop_low_score_entities.py``` is provided, which can be used to drop a certain percentage of the lowest confidence predictions to increase precision and convert the output to CoNLL format.

Please note that for training the original version of the repository has to be used.

For convenience the ```README.md``` of the forked repository is provided below.

------------------------------------

## NER Tagger

NER Tagger is an implementation of a Named Entity Recognizer that obtains state-of-the-art performance in NER on the 4 CoNLL datasets (English, Spanish, German and Dutch) without resorting to any language-specific knowledge or resources such as gazetteers. Details about the model can be found at: http://arxiv.org/abs/1603.01360


## Initial setup

To use the tagger, you need Python 2.7, with Numpy and Theano installed.


## Tag sentences

The fastest way to use the tagger is to use one of the pretrained models:

```
./tagger.py --model models/english/ --input input.txt --output output.txt
```

The input file should contain one sentence by line, and they have to be tokenized. Otherwise, the tagger will perform poorly.


## Train a model

To train your own model, you need to use the train.py script and provide the location of the training, development and testing set:

```
./train.py --train train.txt --dev dev.txt --test test.txt
```

The training script will automatically give a name to the model and store it in ./models/
There are many parameters you can tune (CRF, dropout rate, embedding dimension, LSTM hidden layer size, etc). To see all parameters, simply run:

```
./train.py --help
```

Input files for the training script have to follow the same format than the CoNLL2003 sharing task: each word has to be on a separate line, and there must be an empty line after each sentence. A line must contain at least 2 columns, the first one being the word itself, the last one being the named entity. It does not matter if there are extra columns that contain tags or chunks in between. Tags have to be given in the IOB format (it can be IOB1 or IOB2).
