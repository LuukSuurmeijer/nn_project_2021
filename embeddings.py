import datasets
from transformers import BertModel
from transformers import AutoTokenizer
import torch


# adapted from https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
def tokenize_and_align_labels(examples, text_column_name, label_column_name):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        # TODO: choose good padding size
        padding= False, #"max_length",
        truncation=True,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
                # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
                #label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_label2id_list(data, label_column):
    label_list = set()
    for i in data:
        for label in i[label_column]:
            label_list.add(label)
    label_to_id = {l: i+1 for i, l in enumerate(label_list)}
    return label_to_id


def create_embeddings(tokenized_inputs):
    tokenized_inputs['last_hidden_states'] = []
    for i, ids in enumerate(tokenized_inputs.input_ids):
        outputs = model(torch.tensor(ids).unsqueeze(0))
        tokenized_inputs['last_hidden_states'].append(outputs[0])
    return tokenized_inputs


d = datasets.load_dataset('TagDataset.py', data_dir='train_test_split/')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, is_split_into_words=True)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

label_to_id = get_label2id_list(d['train'], 'tags')
tokenized_train = tokenize_and_align_labels(d['train'], 'words', 'tags')
create_embeddings(tokenized_train)

vocab = list(tokenizer.vocab.keys())
