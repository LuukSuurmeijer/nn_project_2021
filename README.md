# Neural Networks: Implementation and Application 2021 - Project


## Table of contents
* [General Info](#general-info)
* [Installing](#installing)
* [Data Preprocessing](#data-Preprocessing)
* [Data Loading](#data-loading)
* [Tokenizing](#tokenizing)
* [Embeddings](#embeddings)
* [Running the model](#running-the-model)

## General info

Language is often represented as sequential data - e.g., each word in a sentence is one data point in a series of data points. Therefore, sequence modelling for neural networks in NLP is common. For Part-of-Speech tagging, the context of a tag (i.e. words and tags to the left and right of the word we are looking to tag) is critical, which makes sequence modelling particularly applicable for this task. 
There are different neural network types for sequential modelling, e.g. vanilla recurrent neural networks (RNNs)  or long-short-term-memory (LSTM) models, which are a special kind of RNNs. RNNs suffer from vanishing gradient problems as they backpropagte error through time. LSTMs were created to especially avoid the vanishing gradient problem that comes with regular RNNs and can better capture long-term dependencies.

We implemented two two different model architectures (RNN and LSTM) for POS tagging and measure our results in accuracy to be able to directly compare their differences in performance.

## Installing

Use Conda to create/activate the environment:
```
$ conda env create -f lara_luuk_annalena.yml
$ conda activate lara_luuk_annalena
```

## Data Preprocessing

The data for training our model is the data from the [Ontonotes 4.0](https://catalog.ldc.upenn.edu/LDC2011T03) dataset. Running `cat *.gold_conll >> [OUTPUT FILE]` or `type *.gold_conll > [OUTPUT FILE]` (on Windows) concatenates the files into one file which can subsequently be fed to the data preprocessing script.

Running `python Data_Preprocessing.py [INPUT FILE] [OUTPUT DIR]` will create a directory with the files `data.tsv` (containing data) and a `data.info` file with information about the data.

Run `python generate_split.py [CLEANED DATA FILE] --outdir train_test_split --split 0.9` to create a new directory with the train and test splits.

## Data loading

`TagDataset.py` contains the data loading class for our data. It can be called as follows

`dataset = datasets.load_dataset('TagDataset.py', data_dir='train_test_split/')`

**Example of use:**
```
>>> d = datasets.load_dataset('TagDataset.py', data_dir='train_test_split/')
>>> d
> DatasetDict({
    train: Dataset({
        features: ['index', 'word', 'tag'],
        num_rows: 5217
    })
    test: Dataset({
        features: ['index', 'word', 'tag'],
        num_rows: 569
    })
})

>>> d['train']
> Dataset({
    features: ['index', 'word', 'tag'],
    num_rows: 5217
})

>>> d['train'][0]
> {'index': 0, 'tag': 'NNP', 'word': 'xinhua'}

```


## Tokenizing

Words are lowercased and punctuation from dates and numbers etc. is removed in lines 60-80 in `TagDataset.py`

The method `tokenize_and_align_labels` in `embeddings.py` is adapted from [huggingface](https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py). Tokens are mapped to their Word IDs and sentences are padded to the longest sentence. Labels are aligned with the padded sentences, while taking care of the splitting of words in several parts with the same ID (e.g. #embedding #s) and special tokens ([SEP] and [CLS]) by assigning those the label `-100`. 

## Embeddings

The method `create_embeddings` is given a tokenized sentences and an embedding model and returns the embedded sentence (dimensions: 1 x max sentence length x 768). The method is called during training and creates embeddings one by one/batch by batch. 

## Running the model

`model.py` contains the method to initialize the model. There is a simple RNN model and an LSTM model available.

To train and test the model call `train.py`. The following arguments can be specified:

- `--num_layers`   number of recurrent layers, default: `1`

- `--epochs`       number of epochs, default: `3`

- `--hiddens`      number of hidden units per layer, default: `100`

- `--type`         model type (LSTM or RNN), no default, this must be specified

- `--batchsize`   batch size, default: `1`

- `--lr`          learning rate, default: `0.001`

- `--loginterval` interval at which logs should be printed to the console, default: `5`

-  `--output`      name under which the model should be saved, default: `None`, if nothing is specified, the model will not be saved

To see a list of all arguments do `python train.py -h`. 

Running `python train.py --type [MODEL TYPE]` will initialize the default model. Data will be loaded and tokenized and the model will be trained for 15 epochs (mini-batchsize of 1 sentence) using Cross Entropy Loss. If CUDA is available, the model will be trained on CUDA, else on GPU. After training, the model will be saved to `model/[NAME OF MODEL].model` and loss and accuracy will be plotted. 

After training, the model will immediately be evaluated on the test data, and average test loss and accuracy will be shown.

We implemented a method `look_at_test_example()` in `train.py` which takes an id of a testing example, loads the model from the `*.model` file and prints the sentence, the predicted tags as well as the correct tags to the console. However, due to reasons of time we were not able to use the method and find interesting examples.
