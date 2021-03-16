import torch
import torch.optim as optim
import torch.nn as nn
from embeddings import *
from model import RNNTagger
import datasets
from transformers import BertModel
from transformers import AutoTokenizer

import matplotlib.pyplot as plt

HIDDEN_DIM = 100
EMBEDDING_DIM = 768
TAGSET_SIZE = 40
EPOCHS = 15

model = RNNTagger(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, tagset_size=TAGSET_SIZE)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=-100) #ignore padding ?

#load the data
d = datasets.load_dataset('TagDataset.py', data_dir='train_test_split/')

#tokenize the data
print("Generating embeddings...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, is_split_into_words=True)
embedding_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

#dataset
print("Tokenizing data...")
label_to_id = get_label2id_list(d['train'], 'tags')
tokenized_train = tokenize_and_align_labels(label_to_id, tokenizer, d['train'], 'words', 'tags')

examples = tokenized_train['input_ids']

train_losses = []
train_counter = []
avg_epoch_loss = []
for epoch in range(EPOCHS):
    running_loss = 0.0
    model.train()
    for id, example in enumerate(tokenized_train['input_ids']):
        #model.initHidden()
        input = create_embeddings(embedding_model, tokenized_train['input_ids'][id]) #shape: (1, 86, 768)
        target = tokenized_train['labels'][id]
        optimizer.zero_grad()

        pred = model(input) #forward pass

        #reshape targets and predictions
        pred = pred.view(-1, pred.shape[-1])
        target = target.unsqueeze(0).view(-1)

        #print(target.shape)
        #print(pred.shape)

        loss = criterion(pred, target) #calculate loss
        loss.backward() #backward pass
        optimizer.step() #update weights

        # print statistics
        running_loss += loss.item()
        train_losses.append(loss.item())
        train_counter.append((epoch*len(examples)) + id)
        if id % 41 == 0:
            print(f"Epoch: {epoch+1} | example: {id}/{len(examples)} | loss: {loss.item()}")

    print(f"Loss after epoch {epoch+1}: {running_loss}")
    avg_epoch_loss.append(running_loss/len(examples))


plt.plot(train_counter, train_losses, color='blue')
plt.scatter(list([i * len(examples) for i in range(EPOCHS)]), avg_epoch_loss, color='red')
plt.xlabel('Number of training examples seen')
plt.ylabel('Cross Entropy Loss')
plt.show()
