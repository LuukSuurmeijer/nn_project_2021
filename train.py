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
#plotting outputs/argparse
import matplotlib.pyplot as plt
import wandb
import pickle
import argparse
from functools import partial
import sys


def summarize(model, hyperparameters):
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
    for key, value in hyperparameters.items():
        print("{:<15}: {:<15}".format(key, str(value)))

#adapted from https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20for%20PoS%20Tagging.ipynb
def accuracy(preds, targets):
    # preds = preds * -1 (don't need this)
    max_preds = preds.argmax(dim = 1, keepdim = True).squeeze() # get the index of the max probability
    targets = targets
    correct = [i for i in range(len(targets)) if max_preds[i] == targets[i] and targets[i] != -100] #correct labels except the pads
    # non_pad_elements = (targets != -100).nonzero() (this was from the tutorial linked above, might need it again, because it does it with tensors?)

    # get the number of non-pad tokens in the target
    to_be_ignored = [i for i, label_id in enumerate(targets) if label_id == -100]
    target_cleaned = [label_id for i, label_id in enumerate(targets) if i not in to_be_ignored]
    num_target_non_pad_tokens = len(target_cleaned)

    # correct = max_preds[non_pad_elements].squeeze(1).eq(targets[non_pad_elements])
    num_correct = len(correct)
    return np.round(num_correct / num_target_non_pad_tokens, 6)
    # return correct.sum() / torch.FloatTensor([targets[non_pad_elements].shape[0]]).to(device)


parser = argparse.ArgumentParser(description='Train the neural network.')
parser.add_argument('--num_layers', type=int, default=1, help='number of recurrent layers')
parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
parser.add_argument('--hiddens', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--type', help="LSTM/RNN", required=True)
parser.add_argument('--batchsize', type=int, default=1, help="Batch size, must be 1 for CPU (i think)")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--loginterval', type=int, default=5, help="log interval")
parser.add_argument('--output', default=None, help="save model")
args = parser.parse_args()


HIDDEN_DIM = args.hiddens
EMBEDDING_DIM = 768
TAGSET_SIZE = 40
EPOCHS = args.epochs

wandb.init(project='nn_project_2021')
wandb.config.update(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#create model, define loss function and optimizer
model = RNNTagger(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, tagset_size=TAGSET_SIZE, n_layers=args.num_layers, type=args.type).to(device)

summarize(model, vars(args))
print(f"Using {args.type} on {device}")

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device) #ignore padding ?

#load the data
d = datasets.load_dataset('TagDataset.py', data_dir='train_test_split/')

#load the tokenizer and embeddings
print("...Generating embeddings...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, is_split_into_words=True)
embedding_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(device)
label_to_id = get_label2id_list(d, 'tags')

#preparing the datasets
print("...Tokenizing data...")

#train and test
train_dataset = d['train']
test_dataset = d['test']

#prepare tokenization function
tokenize_and_align_labels_p = partial(
    tokenize_and_align_labels,
    label_to_id=label_to_id,
    tokenizer=tokenizer,
    text_column_name='words',
    label_column_name='tags',
    )

#training dataloader
train_dataset = train_dataset.map(tokenize_and_align_labels_p, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize)

#testing dataloader
test_dataset = test_dataset.map(tokenize_and_align_labels_p, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize) #always batch size 1 for test set so we can plot


### TRAINING ###
def train():
    wandb.watch(model)
    print("TRAINING")
    train_losses = []
    train_counter = []
    train_acc = []
    avg_epoch_loss = []
    avg_epoch_acc = []
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        for id, example in enumerate(train_dataloader):
            if id == 50:
                break
            ex = example['input_ids'].to(device)
            input = create_embeddings(embedding_model, ex).to(device) #shape: (1, 86, 768)
            target = example['labels'].to(device)


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
            train_counter.append(((epoch*len(train_dataloader)) + id)*args.batchsize)

            wandb.log({"train_loss" : loss, "train_accuracy" : acc})

            print_acc = running_acc / len(train_dataloader)
            print_loss = running_loss / len(train_dataloader)

            if args.batchsize == 1 and id % args.loginterval == 0:
                print("Epoch: {:<12} | acc: {:<12} | loss: {:<12}".format(f"{epoch+1} ({id}/{len(train_dataloader)})",
                                                                          acc, loss.item()))

        print(f"Loss after epoch {epoch+1}: {running_loss}")
        print(f"Avg acc after epoch {epoch+1}: {running_acc/len(train_dataloader)}")
        avg_epoch_loss.append(running_loss/len(train_dataloader))
        avg_epoch_acc.append(running_acc/len(train_dataloader))

    # Save model for inference
    if args.output:
        torch.save(model.state_dict(), f'{args.output}.model') #save model
        with open(f'{args.output}_dict.pickle', 'wb') as handle: #save mapping from ids to labels
            pickle.dump(label_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plot error
    plt.plot(train_counter, train_losses, color='blue', zorder=1)
    plt.scatter(list([i * len(train_dataset) for i in range(EPOCHS)]), avg_epoch_loss, color='red', zorder=2)
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig(f'loss_{args.type}_{args.hiddens}_{args.num_layers}_{args.lr}.pdf')
    plt.clf()

    # plot accuracy
    plt.plot(train_counter, train_acc, color='blue', zorder=1)
    #plt.scatter(list([i * len(train_dataset) for i in range(EPOCHS)]), avg_epoch_acc, color='red', zorder=2)
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Accuracy')
    plt.savefig(f'acc_{args.type}_{args.hiddens}_{args.num_layers}_{args.lr}.pdf')
    plt.clf()


### EVALUATING ###
def test(testmodel=None, load=None):
    print("EVALUATING THE MODEL")
    # TODO: We need to make the file path for loading the state dict an argument depending on which model we train?
    if load:
        model.load_state_dict(torch.load(f'{load}'))
    elif testmodel:
        model = testmodel

    model.eval()


    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for id, example in enumerate(test_dataloader):
            # model.initHidden()
            ex = example['input_ids'].to(device)
            input = create_embeddings(embedding_model, ex).to(device)  # shape: (batch_size, 86, 768)
            target = example['labels'].to(device)

            pred = model(input)  # forward pass

            # reshape targets and predictions
            pred = pred.view(-1, pred.shape[-1])
            target = target.unsqueeze(0).view(-1)

            label_ids = torch.argmax(pred, dim=1)
            # print([id_to_label[i.item()] for i in label_ids])

            loss = criterion(pred, target)  # calculate loss
            acc = accuracy(pred, target)

            wandb.log({"test_loss" : loss, "test_accuracy" : acc})
            running_loss += loss.item()
            running_acc += acc

    print(f'Average Loss per example: {running_loss / len(test_dataloader)}')
    print(f'Average accuracy per example: {running_acc / len(test_dataloader)}')


def look_at_test_example(sentence_id):
    """
    print a testing example to the console
    :param sentence_id: id of the testing example that should be looked at
    """

    model.load_state_dict(torch.load('model/rnn.model'))
    model.eval()

    input = create_embeddings(embedding_model, test_dataset['input_ids'][sentence_id])  # shape: (1, 86, 768)

    pred = model(input)  # forward pass

    target = list(test_dataset['labels'][sentence_id])
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


train()

test(testmodel=model)
