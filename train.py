#pytorch stuff
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
#from torchsummary import summary
#preprocessing and embeddings
import datasets
from transformers import BertModel
from transformers import AutoTokenizer
#our functions
from embeddings import *
from model import RNNTagger
#plotting outputs
import matplotlib.pyplot as plt

def summarize(model):
    data = {name: [name, [*param.data.shape], param.numel()] for name, param in model.named_parameters() if param.requires_grad}
    print("{:<25} {:<25} {:<25}".format("Layer", "Dim", "Number of parameters"))
    print(("="*25)*3)
    for key, value in data.items():
        name, shape, num_param = value
        print("{:<25} {:<25} {:<25}".format(name, str(shape), num_param))
    print(("="*25)*3)
    total = sum([param[2] for param in data.values()])
    print(f"Total trainable parameters: {total}" )
    print(f"Estimated memory required: {(total * 4) * (10**-6)} MB")
    print("\n")

#adapted from https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20for%20PoS%20Tagging.ipynb
def accuracy(preds, targets):
    max_preds = preds.argmax(dim = 1, keepdim = True).squeeze() # get the index of the max probability
    targets = targets
    correct = [i for i in range(len(targets)) if max_preds[i] == targets[i] and targets[i] != -100] #correct labels except the pads
    num_correct = len(correct)
    return np.round(num_correct / len(targets), 3)


HIDDEN_DIM = 100
EMBEDDING_DIM = 768
TAGSET_SIZE = 40
EPOCHS = 15


#create model, define loss function and optimizer
model = RNNTagger(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, tagset_size=TAGSET_SIZE)

summarize(model)


optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=-100) #ignore padding ?

#load the data
d = datasets.load_dataset('TagDataset.py', data_dir='train_test_split/')

#tokenize the data
print("...Generating embeddings...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, is_split_into_words=True)
embedding_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

#dataset
print("...Tokenizing data...")
label_to_id = get_label2id_list(d, 'tags')
tokenized_train = tokenize_and_align_labels(label_to_id, tokenizer, d['train'], 'words', 'tags')
tokenized_test = tokenize_and_align_labels(label_to_id, tokenizer, d['test'], 'words', 'tags')
examples = tokenized_train['input_ids']
examples_test = tokenized_test['input_ids']


### TRAINING ###
def train():
    print("TRAINING")
    train_losses = []
    train_counter = []
    train_acc = []
    avg_epoch_loss = []
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        for id, example in enumerate(tokenized_train['input_ids']):
            #model.initHidden()
            input = create_embeddings(embedding_model, tokenized_train['input_ids'][id]) #shape: (1, 86, 768)
            target = tokenized_train['labels'][id] #shape: (1, 86)
            optimizer.zero_grad()

            pred = model(input) #forward pass

            #reshape targets and predictions
            pred = pred.view(-1, pred.shape[-1])
            target = target.unsqueeze(0).view(-1)

            loss = criterion(pred, target)  # calculate loss
            acc = accuracy(pred, target)

            loss.backward() #backward pass
            optimizer.step() #update weights

            # print statistics
            running_loss += loss.item()
            running_acc += acc
            train_losses.append(loss.item())
            train_acc.append(acc)
            train_counter.append((epoch*len(examples)) + id)
            if id % 30 == 0 or id == len(examples):
                #print(f"Epoch: {epoch+1} | example: {id}/{len(examples)} | loss: {loss.item()}")
                print("Epoch: {:<12} | acc: {:<12} | loss: {:<12}".format(f"{epoch+1} ({id}/{len(examples)})", acc ,loss.item()))

        print(f"Loss after epoch {epoch+1}: {running_loss}")
        print(f"Avg acc after epoch {epoch+1}: {running_acc/len(examples)}")
        avg_epoch_loss.append(running_loss/len(examples))

    # Save model for inference
    torch.save(model.state_dict(), 'model/rnn.model')

    # plot error
    plt.plot(train_counter, train_losses, color='blue', zorder=1)
    plt.scatter(list([i * len(examples) for i in range(EPOCHS)]), avg_epoch_loss, color='red', zorder=2)
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Cross Entropy Loss')
    plt.show()

### EVALUATING ###
def test():
    print("EVALUATING THE MODEL")
    # TODO: We need to make the file path for loading the state dict an argument depending on which model we train?
    model.load_state_dict(torch.load('model/rnn.model'))
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for id, example in enumerate(tokenized_test['input_ids']):
            # model.initHidden()
            input = create_embeddings(embedding_model, tokenized_test['input_ids'][id])  # shape: (1, 86, 768)
            target = tokenized_test['labels'][id]

            pred = model(input)  # forward pass

            # reshape targets and predictions
            pred = pred.view(-1, pred.shape[-1])
            target = target.unsqueeze(0).view(-1)

            label_ids = torch.argmax(pred, dim=1)
            # print([id_to_label[i.item()] for i in label_ids])

            loss = criterion(pred, target)  # calculate loss
            acc = accuracy(pred, target)

            running_loss += loss.item()
            running_acc += acc

    print(f'Average Loss per example: {running_loss / len(examples_test)}')
    print(f'Average accuracy per example: {running_acc / len(examples_test)}')


train()

test()


def look_at_test_example(sentence_id):
    """
    print a testing example to the console
    :param sentence_id: id of the testing example that should be looked at
    """
    #  TODO: We need to make the file path for loading the state dict an argument depending on which model we train?
    model.load_state_dict(torch.load('model/rnn.model'))
    model.eval()

    input = create_embeddings(embedding_model, tokenized_test['input_ids'][sentence_id])  # shape: (1, 86, 768)

    pred = model(input)  # forward pass

    target = list(tokenized_test['labels'][sentence_id])
    label_ids = torch.argmax(pred, dim=1)

    # this should iterate over label_ids and check for each id what the corresponding key in the label_to_ids dict is
    predicted_labels = [list(label_to_id.keys())[list(label_to_id.values()).index(id)] for id in label_ids]

    to_be_ignored = [i for i, label_id in enumerate(target) if label_id == -100]
    target_cleaned = [label_id for i, label_id in enumerate(target) if i not in to_be_ignored]
    labels_cleaned = [label for i, label in enumerate(predicted_labels) if i not in to_be_ignored]

    sentence = d['test']['words'][sentence_id]  # list of shape: (1, sent_length)

    print(f"Test example {sentence_id}")
    print(f"Words: {sentence}")
    print(f"Predicted tags: {labels_cleaned}")
    print(f"Correct tags: {d['test']['tags'][sentence_id]}")