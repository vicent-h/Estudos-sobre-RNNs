import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            emb_size, 
            n_layers, 
            hidden_size,
            dropout):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(
            emb_size, 
            hidden_size, 
            n_layers, 
            batch_first=True,
            dropout=dropout
            )
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x: torch.Tensor, states=None):
        # device = x.device
        # lengths = (x != 0).sum(dim=1).to('cpu')

        # lengths_sorted, perm_idx = lengths.sort(0, descending=True)
        # x = x[perm_idx]

        x = self.emb(x)
        # x = pack_padded_sequence(x, lengths_sorted, batch_first=True)
        out, states = self.lstm(x, states)
        # out, _ = pad_packed_sequence(out, batch_first=True)

        out = self.out_layer(out)

        # desfaz a ordenação para manter a ordem original do batch
        # _, inv_perm_idx = perm_idx.sort(0)
        # out = out[inv_perm_idx]

        return out, states
    

class LSTMPad(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            emb_size, 
            n_layers, 
            hidden_size,
            dropout):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_size, 
            hidden_size, 
            n_layers, 
            batch_first=True,
            dropout=dropout
            )
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x: torch.Tensor, states=None):
        device = x.device
        lengths = (x != 0).sum(dim=1).to('cpu')

        lengths_sorted, perm_idx = lengths.sort(0, descending=True)
        x = x[perm_idx]

        if states is not None:
            h, c = states
            h = h[:, perm_idx]
            c = c[:, perm_idx]
            states = (h, c)

        x = self.emb(x)
        x = pack_padded_sequence(x, lengths_sorted, batch_first=True)
        out, states = self.lstm(x, states)
        out, _ = pad_packed_sequence(out, batch_first=True)

        out = self.out_layer(out)

        # desfaz a ordenação para manter a ordem original do batch
        _, inv_perm_idx = perm_idx.sort(0)
        out = out[inv_perm_idx]

        return out, states