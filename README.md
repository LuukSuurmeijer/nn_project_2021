# Neural Networks: Implementation and Application 2021 - Project


## Table of contents
* [General Info](#general-info)
* [Installing](#installing)
* [Data Preprocessing](#data-Preprocessing)

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

`>{'index': 0, 'tag': 'NNP', 'word': 'Xinhua'}`

# Tokenizing

Important for later: I have no idea what is going on with the tokenization in the example that Anna provided, so for now i just lowercase the words and remove punctuation from dates and numbers etc. This all happens in lines 60-80 in `TagDataset.py`.


`
