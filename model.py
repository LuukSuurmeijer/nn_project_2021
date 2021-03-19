
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RNNTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(RNNTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # The RNN takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.RNNlayer = nn.RNN(embedding_dim, hidden_dim, batch_first=True) #dim: (embedding_dim, hidden_dim) tanh non-linearity
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size) #dim: (hidden_dim, tagset_size) linear

    def forward(self, sentence):
        #sentence is a torch tensor of (1, seq_len, training_size)
        hidden, _ = self.RNNlayer(sentence)
        output = self.hidden2tag(hidden)
        predictions = F.log_softmax(output, dim=1)
        return predictions

    def initHidden(self):
        return torch.zeros(1, self.hidden_dim)


        #768 * 100 = 76800

        #100 * 100 = 10000

        #100 * 39 = 3900