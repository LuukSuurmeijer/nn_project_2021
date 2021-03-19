import datasets
from transformers import BertModel
from transformers import AutoTokenizer
from embeddings import *
from model import RNNTagger
import torch.nn as nn
import numpy as np



def accuracy(preds, targets):
    max_preds = preds.argmax(dim = 1, keepdim = True).squeeze() # get the index of the max probability
    targets = targets
    correct = [i for i in range(len(targets)) if max_preds[i] == targets[i] and targets[i] != -100] #correct labels except the pads
    num_correct = len(correct)
    return np.round(num_correct / len(targets), 3)

d = datasets.load_dataset('TagDataset.py', data_dir='train_test_split/')
criterion = nn.CrossEntropyLoss(ignore_index=-100) #ignore padding ?



tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, is_split_into_words=True)
embedding_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

#dataset
print("...Tokenizing data...")
label_to_id = get_label2id_list(d['test'], 'tags')
id_to_label = {i:l for l,i in label_to_id.items()}
tokenized_train = tokenize_and_align_labels(label_to_id, tokenizer, d['test'], 'words', 'tags')
examples = tokenized_train['input_ids']


HIDDEN_DIM = 100
EMBEDDING_DIM = 768
TAGSET_SIZE = 40

model = RNNTagger(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, tagset_size=TAGSET_SIZE)
model.load_state_dict(torch.load('model/rnn.model'))
model.eval()

running_loss = 0.0
running_acc = 0.0

with torch.no_grad():
    for id, example in enumerate(tokenized_train['input_ids']):
        #model.initHidden()
        input = create_embeddings(embedding_model, tokenized_train['input_ids'][id]) #shape: (1, 86, 768)
        target = tokenized_train['labels'][id]

        pred = model(input) #forward pass

        #reshape targets and predictions
        pred = pred.view(-1, pred.shape[-1])
        target = target.unsqueeze(0).view(-1)

        label_ids = torch.argmax(pred, dim=1)
        #print([id_to_label[i.item()] for i in label_ids])


        loss = criterion(pred, target) #calculate loss
        acc = accuracy(pred, target)

        running_loss += loss.item()
        running_acc += acc

print(f'Average Loss per example: {running_loss/len(examples)}')
print(f'Average accuracy per example: {running_acc/len(examples)}')
