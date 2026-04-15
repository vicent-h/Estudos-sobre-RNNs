import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        encoder_layers,
        encoder_hidden_size,
        encoder_dropout,
        decoder_layers,
        decoder_hidden_size,
        decoder_dropout,
        states_encoder_to_decoder = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.states_encoder_to_decoder = states_encoder_to_decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.embedding_dec = nn.Embedding(vocab_size, embed_dim)
        self.LSTMEncoder = nn.LSTM(
            embed_dim, 
            encoder_hidden_size, 
            encoder_layers, 
            batch_first=True,
            dropout=encoder_dropout,
        )
        self.LSTMDecoder = nn.LSTM(
            embed_dim, 
            decoder_hidden_size, 
            decoder_layers, 
            batch_first=True,
            dropout=decoder_dropout,
        )

        self.out_layer = nn.Linear(decoder_hidden_size, vocab_size)

    def forward(self, input, output, states=None):
        
        memory = self.encode(input, states)
        logits, (h, c) = self.decode(output, memory)
        
        return logits, (h, c)
    
    def encode(self, input, states=None):
        x_embedded = self.embedding(input)
        _, (h, c) = self.LSTMEncoder(x_embedded, states)
        return (h, c)
    
    def decode(self, output, memory):
        y_embedded = self.embedding(output)
        # y_embedded = self.embedding_dec(output)
        out, (h, c) = self.LSTMDecoder(y_embedded, memory)
        logits = self.out_layer(out)
        return logits, (h, c)
    

class LSTMBidirectional(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        encoder_layers,
        encoder_hidden_size,
        encoder_dropout,
        decoder_layers,
        decoder_hidden_size,
        decoder_dropout,
        states_encoder_to_decoder = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.states_encoder_to_decoder = states_encoder_to_decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layers = decoder_layers
        self.decoder_hidden_size = decoder_hidden_size
        # self.embedding_dec = nn.Embedding(vocab_size, embed_dim)
        self.LSTMEncoder = nn.LSTM(
            embed_dim, 
            encoder_hidden_size, 
            encoder_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=encoder_dropout,
        )
        self.LSTMDecoder = nn.LSTM(
            embed_dim, 
            decoder_hidden_size, 
            decoder_layers, 
            batch_first=True,
            dropout=decoder_dropout,
        )

        print(2 * encoder_hidden_size * encoder_layers, decoder_hidden_size * decoder_layers)
        self.proj_state_c = nn.Linear(2 * encoder_hidden_size * encoder_layers, decoder_hidden_size * decoder_layers)
        self.proj_state_h = nn.Linear(2 * encoder_hidden_size * encoder_layers, decoder_hidden_size * decoder_layers)
        self.out_layer = nn.Linear(decoder_hidden_size, vocab_size)

    def forward(self, input, output, states=None):
        
        memory = self.encode(input, states)
        logits, (h, c) = self.decode(output, memory)
        return logits, (h, c)
    
    def encode(self, input, states=None):
        x_embedded = self.embedding(input)
        _, (h, c) = self.LSTMEncoder(x_embedded, states) # states = [L_ENC*2, B, H_ENC]

        h = h.transpose(0,1) # [B, L_ENC*2, H_ENC]

        h = h.reshape(h.size(0), -1) # [B, L_ENC * 2 * H_ENC]

        c = c.transpose(0,1) # [B, L_ENC*2, H_ENC]
        c = c.reshape(c.size(0), -1) # [B, L_ENC * 2 * H_ENC]
        
        new_h = self.proj_state_h(h) # [B, L_DEC * H_DEC]
        new_c = self.proj_state_c(c) # [B, L_DEC * H_DEC]

        new_h = new_h.reshape(new_h.size(0), self.decoder_layers, self.decoder_hidden_size) # [B, L_DEC, H_DEC]
        new_c = new_c.reshape(new_c.size(0), self.decoder_layers, self.decoder_hidden_size) # [B, L_DEC, H_DEC]

        new_h = new_h.transpose(0, 1).contiguous() # [L_DEC, B, H_DEC]
        new_c = new_c.transpose(0, 1).contiguous() # [L_DEC, B, H_DEC]

        return (new_h, new_c)
    
    def decode(self, output, memory):
        y_embedded = self.embedding(output)
        # y_embedded = self.embedding_dec(output)
        out, (h, c) = self.LSTMDecoder(y_embedded, memory)
        logits = self.out_layer(out)
        return logits, (h, c)



class LSTMAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        encoder_layers,
        encoder_hidden_size,
        encoder_dropout,
        decoder_layers,
        decoder_hidden_size,
        decoder_dropout,
        states_encoder_to_decoder = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.states_encoder_to_decoder = states_encoder_to_decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layers = decoder_layers
        self.decoder_hidden_size = decoder_hidden_size
        # self.embedding_dec = nn.Embedding(vocab_size, embed_dim)
        self.LSTMEncoder = nn.LSTM(
            embed_dim, 
            encoder_hidden_size, 
            encoder_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=encoder_dropout,
        )
        self.LSTMDecoder = nn.LSTM(
            embed_dim+decoder_hidden_size, 
            decoder_hidden_size,
            decoder_layers, 
            batch_first=True,
            dropout=decoder_dropout,
        )

        self.encoded_proj = nn.Linear(2*encoder_hidden_size, decoder_hidden_size)
        self.proj_h = nn.Linear(2*encoder_hidden_size*encoder_layers, decoder_hidden_size*decoder_layers)
        self.proj_c = nn.Linear(2*encoder_hidden_size*encoder_layers, decoder_hidden_size*decoder_layers)
        self.out_layer = nn.Linear(decoder_hidden_size*2, vocab_size)

    def forward(self, input_encoder, input_decoder, states=None):
        
        encoded, memory = self.encode(input_encoder, states)
        logits, (h, c) = self.decode(input_decoder, encoded, memory)
        return logits, (h, c)
    
    def encode(self, input, states=None):
        x_embedded = self.embedding(input)
        encoded, (h, c) = self.LSTMEncoder(x_embedded, states) # states = [L_ENC*2, B, H_ENC]

        h = h.transpose(0,1) # [B, L_ENC*2, H_ENC]
        h = h.reshape(h.size(0), -1) # [B, L_ENC * 2 * H_ENC]

        c = c.transpose(0,1) # [B, L_ENC*2, H_ENC]
        c = c.reshape(c.size(0), -1) # [B, L_ENC * 2 * H_ENC]
        new_h = self.proj_h(h)
        new_c = self.proj_c(c)

        new_h = new_h.reshape(new_h.size(0), self.decoder_layers, self.decoder_hidden_size) # [B, L_DEC, H_DEC]
        new_c = new_c.reshape(new_c.size(0), self.decoder_layers, self.decoder_hidden_size) # [B, L_DEC, H_DEC]

        new_h = new_h.transpose(0, 1).contiguous() # [L_DEC, B, H_DEC]
        new_c = new_c.transpose(0, 1).contiguous() # [L_DEC, B, H_DEC]

        encoded = self.encoded_proj(encoded)
        return encoded, (new_h, new_c)
    
    def decode(self, input_decoder, encoded, memory):
        h, c = memory
        logits_total = []
        y_embedded = self.embedding(input_decoder) # [B, L, E]
        for i in range(y_embedded.size(1)):
            current_input = y_embedded[:, i:i+1, :]
            current_h = h[-1] # [B , H_ENC]
            current_h = current_h.unsqueeze(2) # [B, H_ENC, 1]

            score = torch.bmm(encoded, current_h) # [B, L, H_ENC] * [B, H_ENC, 1] = [B, L, 1]
            score_softmax = torch.softmax(score, dim=1) # [B, L, 1]
            score_softmax = score_softmax.squeeze(2) # [B, L, 1]
            
            context = torch.bmm(score_softmax.unsqueeze(1), encoded) # [B, 1, L] * [B, L, H_ENC]

            input_decoder_novo = torch.cat([context, current_input], dim=2) # [B, 1, E + H_DEC]

            output, (h, c) = self.LSTMDecoder(input_decoder_novo, (h, c))
            out = torch.concat([output, context], dim=2)
            logits = self.out_layer(out)
            # print(logits.size())
            logits_total.append(logits)
        
        logits_total = torch.concat(logits_total, dim=1)
        return logits_total, (h, c)
    

class LSTMAttentionCompl(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        encoder_layers,
        encoder_hidden_size,
        encoder_dropout,
        decoder_layers,
        decoder_hidden_size,
        decoder_dropout,
        states_encoder_to_decoder = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.states_encoder_to_decoder = states_encoder_to_decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layers = decoder_layers
        self.decoder_hidden_size = decoder_hidden_size
        # self.embedding_dec = nn.Embedding(vocab_size, embed_dim)
        self.LSTMEncoder = nn.LSTM(
            embed_dim, 
            encoder_hidden_size, 
            encoder_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=encoder_dropout,
        )
        self.LSTMDecoder = nn.LSTM(
            embed_dim+decoder_hidden_size, 
            decoder_hidden_size,
            decoder_layers, 
            batch_first=True,
            dropout=decoder_dropout,
        )

        self.encoded_proj = nn.Linear(2*encoder_hidden_size, decoder_hidden_size)
        self.proj_h = nn.Linear(2*encoder_hidden_size*encoder_layers, decoder_hidden_size*decoder_layers)
        self.proj_c = nn.Linear(2*encoder_hidden_size*encoder_layers, decoder_hidden_size*decoder_layers)
        self.out_layer = nn.Linear(decoder_hidden_size*2, vocab_size)

    def forward(self, input_encoder, input_decoder, states=None):
        
        encoded, memory = self.encode(input_encoder, states)
        logits, (h, c) = self.decode(input_decoder, encoded, memory)
        return logits, (h, c)
    
    def encode(self, input, states=None):
        x_embedded = self.embedding(input)
        encoded, (h, c) = self.LSTMEncoder(x_embedded, states) # states = [L_ENC*2, B, H_ENC]

        h = h.transpose(0,1) # [B, L_ENC*2, H_ENC]
        h = h.reshape(h.size(0), -1) # [B, L_ENC * 2 * H_ENC]

        c = c.transpose(0,1) # [B, L_ENC*2, H_ENC]
        c = c.reshape(c.size(0), -1) # [B, L_ENC * 2 * H_ENC]
        new_h = self.proj_h(h)
        new_c = self.proj_c(c)

        new_h = new_h.reshape(new_h.size(0), self.decoder_layers, self.decoder_hidden_size) # [B, L_DEC, H_DEC]
        new_c = new_c.reshape(new_c.size(0), self.decoder_layers, self.decoder_hidden_size) # [B, L_DEC, H_DEC]

        new_h = new_h.transpose(0, 1).contiguous() # [L_DEC, B, H_DEC]
        new_c = new_c.transpose(0, 1).contiguous() # [L_DEC, B, H_DEC]

        encoded = self.encoded_proj(encoded)
        return encoded, (new_h, new_c)
    
    def decode(self, input_decoder, encoded, memory):
        h, c = memory
        logits_total = []
        y_embedded = self.embedding(input_decoder) # [B, L, E]
        for i in range(y_embedded.size(1)):
            current_input = y_embedded[:, i:i+1, :]
            current_h = h[-1] # [B , H_ENC]
            current_h = current_h.unsqueeze(2) # [B, H_ENC, 1]

            score = torch.bmm(encoded, current_h) # [B, L, H_ENC] * [B, H_ENC, 1] = [B, L, 1]
            score_softmax = torch.softmax(score, dim=1) # [B, L, 1]
            score_softmax = score_softmax.squeeze(2) # [B, L]
            
            context = torch.bmm(score_softmax.unsqueeze(1), encoded) # [B, 1, L] * [B, L, H_ENC] = [B, 1, H_ENC]

            input_decoder_novo = torch.cat([context, current_input], dim=2) # [B, 1, E + H_DEC]

            output, (h, c) = self.LSTMDecoder(input_decoder_novo, (h, c))
            out = torch.concat([output, context], dim=2)
            logits = self.out_layer(out)
            # print(logits.size())
            logits_total.append(logits)

            #Parte complementar
            encoded = torch.concat([encoded, output], dim=1) # [B, L+1, H_DEC]
        
        logits_total = torch.concat(logits_total, dim=1)
        return logits_total, (h, c)

