import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, n_layers, vocab_size, embed_dim=256):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim * 2),
                nn.ReLU()
            ) for _ in range(n_layers)
        ])
        
        self.out_layer = nn.Linear(embed_dim * 2, vocab_size)

    def forward(self, x, mem_list=None):
        x = self.embedding(x)
        
        if mem_list is None:
            mem_list = [torch.zeros_like(x) for _ in range(self.n_layers)]
        
        new_memories = []
        
        for i in range(self.n_layers):
            current_mem = mem_list[i]
            
            out = self.layers[i](torch.cat([current_mem, x], dim=-1))
            
            new_h = out[..., :self.embed_dim].clone()
            x = out[..., self.embed_dim:].clone()
            
            new_memories.append(new_h)
            
        logits = self.out_layer(torch.cat([new_memories[-1], x], dim=-1))
        
        return logits, new_memories
    
class LSTMCell(nn.Module):
    def __init__(self, embed_dim=256, hidden_size=512):
        super(LSTMCell, self).__init__()
        self.embed_dim = embed_dim

        self.fg = nn.Sequential(nn.Linear(embed_dim+hidden_size, hidden_size), nn.Sigmoid())
        self.ig = nn.Sequential(nn.Linear(embed_dim+hidden_size, hidden_size), nn.Sigmoid())
        self.og = nn.Sequential(nn.Linear(embed_dim+hidden_size, hidden_size), nn.Sigmoid())
        self.cm = nn.Sequential(nn.Linear(embed_dim+hidden_size, hidden_size), nn.Tanh())

    def forward(self, x, h, c):
        combined = torch.cat([h, x], dim=-1)

        f = self.fg(combined)
        i = self.ig(combined)
        cm = self.cm(combined)
        og = self.og(combined)

        c_next = (c * f) + (i * cm)
        h_next = torch.tanh(c_next) * og
        
        return h_next, c_next
    
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, n_layers=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(LSTMCell(embed_dim, hidden_size))
        
        for _ in range(1, n_layers):
            self.layers.append(LSTMCell(hidden_size, hidden_size))
        
        self.out_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_list=None, c_list=None):
        x_emb = self.embedding(x) # [Batch, Seq_Len, Embed]
        B, L, E = x_emb.size()
        
        # Inicialização dos estados (se for a primeira janela)
        if h_list is None:
            h_list = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.n_layers)]
            c_list = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.n_layers)]
        else:
            # Garante que os estados iniciais não tenham dimensão de sequência (apenas Batch e Hidden)
            h_list = [state[:, -1, :] if state.dim() == 3 else state for state in h_list]
            c_list = [state[:, -1, :] if state.dim() == 3 else state for state in c_list]

        all_layer_outputs = []
        
        # O LOOP TEMPORAL OBRIGATÓRIO (A essência da RNN)
        for t in range(L):
            current_input = x_emb[:, t, :] # Pega o caractere atual [B, E]
            
            new_h_list = []
            new_c_list = []
            
            for n in range(self.n_layers):
                # Agora sim: h_list[n] é o h do passo t-1
                hn, cn = self.layers[n](current_input, h_list[n], c_list[n])
                
                new_h_list.append(hn)
                new_c_list.append(cn)
                
                # O input da próxima camada é o output da atual
                current_input = hn 
            
            # Atualiza os estados para o próximo timestep (t+1)
            h_list = new_h_list
            c_list = new_c_list
            all_layer_outputs.append(h_list[-1].unsqueeze(1)) # Guarda o h da última camada

        # Reconstrói a sequência para calcular a loss: [B, L, Vocab]
        full_sequence_h = torch.cat(all_layer_outputs, dim=1)
        logits = self.out_layer(full_sequence_h)
        
        return logits, h_list, c_list

class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # A nn.LSTM oficial processa a sequência inteira de uma vez de forma otimizada
        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hidden_size, 
            num_layers=n_layers, 
            batch_first=True # Garante o shape [Batch, Seq, Dim]
        )
        
        self.out_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, states=None):
        # x: [Batch, Seq] -> ids dos tokens
        x = self.embedding(x) # [Batch, Seq, embed_dim]
        
        # out: contém todos os h de todas as posições temporais
        # states: contém o último (h, c) necessário para a próxima janela
        out, states = self.lstm(x, states)
        
        logits = self.out_layer(out) # [Batch, Seq, vocab_size]
        
        return logits, states