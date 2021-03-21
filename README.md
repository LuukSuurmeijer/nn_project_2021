# Neural Networks: Implementation and Application 2021 - Project


## Table of contents
* [General Info](#general-info)
* [Installing](#installing)
* [Data Preprocessing](#data-Preprocessing)
* [Data Loading](#data-loading)

## General info
Short description of the project.

## Installing
Use Conda to create/activate the environment:
```
$ conda env create -f lara_luuk_annalena.yml
$ conda activate lara_luuk_annalena
```

## Data Preprocessing
Running `python Data_Preprocessing.py [INPUT FILE] [OUTPUT DIR]` will create a directory with the files `sample.tsv` and `cleaned.conll` (containing data) and a `sample.info` file with information about the data.

Run `python generate_split.py sample_cleaned.conll/sample.tsv --outdir train_test_split --split 0.9` will create a new directory with the train and test splits.

## Data loading

`TagDataset.py` contains the data loading class for our data. It can be called as follows

`dataset = datasets.load_dataset('TagDataset.py', data_dir='train_test_split/')`

Example of use:

`d = datasets.load_dataset('TagDataset.py', data_dir='train_test_split/')`

`d`

`>DatasetDict({
    train: Dataset({
        features: ['index', 'word', 'tag'],
        num_rows: 5217
    })
    test: Dataset({
        features: ['index', 'word', 'tag'],
        num_rows: 569
    })
})`

`d['train']`

`>Dataset({
    features: ['index', 'word', 'tag'],
    num_rows: 5217
})`

`d['train'][0]`

`>{'index': 0, 'tag': 'NNP', 'word': 'xinhua'}`

### Tokenizing

Words are lowercased and punctuation from dates and numbers etc. is removed in lines 60-80 in `TagDataset.py`

The method `tokenize_and_align_labels` in `embeddings.py` is adapted from [huggingface](https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py). Tokens are mapped to der Word IDs and sentences are padded to the longest sentence. Labels are aligned with the padded sentences taking care of the padding, the splitting of words in several parts with the same ID (e.g. #embedding #s) and special tokens ([SEP] and [CLS]) by assigning those the lable -100. 

### Embeddings

The method `create_embeddings` is given a tokenized sentences and an embedding model and returns the embedded sentence (dimensions 1 x max sentence length x 768). The method is called during training and creates embeddings one by one. 

### Model

TO-DO: We want to add a second class for an LSTM 

`Model.py` initializes the models used for training. There are either `RNNTagger` or ??? available.

Shoud we describe the models here or is this part of the report?

### Training

To train the model call `train.py`. The following arguments can be specified:
`-- num_layers  number of recurrent layers          default: 1`
`--epochs       number of epochs                    default: 15`
`--hiddens      number of hidden units per layer    default: 200`
`--type         model type (LSTM or RNN)            default: 'RNN'`
`--batchsize    batch size                          default: 1`
`--lr          learning rate                       default: 0.001`

To see a list of all arguments do `python train.py -h`

Running `python train.py` will initialize the default model. Data will be loaded and tokenized and the model will be trained for 15 epochs (mini-batchsize of 1 sentence) using Cross Entropy Loss. The model will be saved to `model/rnn.model` and error will be plotted. 
TO-DO: AT THE MOMENT CALLING `train.py` also runs the testing, we might want to change that

### Testing

TO-DO

`
