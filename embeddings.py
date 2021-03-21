import datasets
from transformers import BertModel
from transformers import AutoTokenizer
import torch


# adapted from https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(d, label_to_id, tokenizer, text_column_name, label_column_name):
    tokenized_inputs = tokenizer(
        d[text_column_name],
        padding="longest",
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(d[label_column_name]):
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
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_label2id_list(data, label_column):
    label_list = set()
    for split in data.keys():
        for i in data[split]:
            for label in i[label_column]:
                label_list.add(label)
    label_to_id = {l: i for i, l in enumerate(label_list)}
    return label_to_id


def create_embeddings(model, sent):
    return model(sent).last_hidden_state
