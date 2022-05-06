import torch
from torch import dropout, nn
from torch.nn import functional as F
import numpy as np
import datetime as dt
from torch.autograd import Variable
import math
import copy
# from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, words_num, embedding_dim, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Embedding = nn.Embedding(num_embeddings=words_num, embedding_dim=embedding_dim)
        # add dropout to LSTM module
        self.LSTM = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.Linear = nn.Linear(hidden_size, words_num)


    def forward(self, data):
        data = self.Embedding(data)
        h0 = torch.zeros(self.num_layers, data.shape[0], self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, data.shape[0], self.hidden_size, device=device)
        data, (_, _) = self.LSTM(data, (h0, c0))
        out = self.Linear(data)
        return out
        

# 2022/2/28 reduce the power of the vanilla_LSTM model, hoping to reduce the overfitting
# too much dropout makes it hard to converge!!!
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


######
# Tune the dropout rate by yourself
class GRU_enhanced(nn.Module):
    def __init__(self, words_num, embedding_dim, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Embedding = nn.Embedding(num_embeddings=words_num, embedding_dim=embedding_dim)
        self.GRU = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.Linear = nn.Linear(hidden_size, words_num)


    def forward(self, data):
        data = self.Embedding(data)
        a0 = torch.zeros(self.num_layers, data.shape[0], self.hidden_size, device=device)
        data, _ = self.GRU(data, a0)
        out = self.Linear(data)
        return out


################ Transformer Decoder copied online ################
################ Haven't tried this part yet ######################
################ Reference: https://medium.com/towards-data-science/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, src):
        return self.embed(src)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).to(device)
        return x


def Attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
        
    if dropout is not None:
        scores = dropout(scores)
            
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = Attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

# layernorm
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).to(device)

    def forward(self, x, e_outputs, src_mask, trg_mask):
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
            src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
            return x

# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

## The final decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

########################## END ###############################

if __name__ == "__main__":
    NET = LSTM_enhanced(3061, 256, 256, 3, 0.1)
    input = torch.ones((32, 10), dtype=torch.long)
    output = NET(input)
    # writer = SummaryWriter("graph")
    # writer.add_graph(NET, input)
    # writer.close()