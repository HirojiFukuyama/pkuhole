import torch
from torch import dropout, nn
import numpy as np
import datetime as dt
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# 'vanilla' baseline model
class vanilla_LSTM(nn.Module):
    def __init__(self, words_num, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Embedding = nn.Embedding(num_embeddings=words_num, embedding_dim=embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.Linear = nn.Linear(hidden_size, words_num)


    def forward(self, data):
        data = self.Embedding(data)
        h0 = torch.zeros(self.num_layers, data.shape[0], self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, data.shape[0], self.hidden_size, device=device)
        data, (_, _) = self.LSTM(data, (h0, c0))
        out = self.Linear(data)
        return out


# the enhanced version is based on the vanilla one above (add something)
# but somehow it is worse than the vanilla one (WITH BATCHNORM)
class LSTM_enhanced(nn.Module):
    def __init__(self, words_num, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Embedding = nn.Embedding(num_embeddings=words_num, embedding_dim=embedding_dim)
        # add dropout to LSTM module
        self.LSTM = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.Linear = nn.Linear(hidden_size, words_num)


    def forward(self, data):
        data = self.Embedding(data)
        h0 = torch.zeros(self.num_layers, data.shape[0], self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, data.shape[0], self.hidden_size, device=device)
        data, (_, _) = self.LSTM(data, (h0, c0))
        out = self.Linear(data)
        return out
        

# 2022/2/28 reduce the power of the vanilla_LSTM model, hoping to reduce the overfitting
class vanilla_GRU(nn.Module):
    def __init__(self, words_num, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Embedding = nn.Embedding(num_embeddings=words_num, embedding_dim=embedding_dim)
        self.GRU = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.Linear = nn.Linear(hidden_size, words_num)


    def forward(self, data):
        data = self.Embedding(data)
        a0 = torch.zeros(self.num_layers, data.shape[0], self.hidden_size, device=device)
        data, _ = self.GRU(data, a0)
        out = self.Linear(data)
        return out
