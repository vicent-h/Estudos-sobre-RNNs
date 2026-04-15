import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMDecoderOnly(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        decoder_layers,
        decoder_hidden_size,
        decoder_dropout
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layers = decoder_layers
        self.decoder_hidden_size = decoder_hidden_size

        self.LSTMDecoder = nn.LSTM(
            embed_dim + decoder_hidden_size,
            decoder_hidden_size,
            decoder_layers, 
            batch_first=True,
            dropout=decoder_dropout
        )

        self.out_layer = nn.Linear(2*decoder_hidden_size, vocab_size)
        
    def forward(self, input: torch.Tensor, memory=None):
        B = input.size(0)

        if memory is None:
            device = input.device
            dtype = next(self.parameters()).dtype

            h = torch.zeros(self.decoder_layers, B, self.decoder_hidden_size, device=device, dtype=dtype)
            c = torch.zeros(self.decoder_layers, B, self.decoder_hidden_size, device=device, dtype=dtype)
            memory = (h, c)
        else:
            h, c = memory
            memory = (h.detach(), c.detach())

        logits, (h, c) = self.decode(input, memory)

        return logits, (h, c)
    
    def decode(self, input_decoder, memory):
        h, c = memory
        logits_total = []

        y_embedded = self.embedding(input_decoder)  # [B, L, E]

        encoded = None  # histórico (outputs anteriores)

        for i in range(y_embedded.size(1)):
            current_input = y_embedded[:, i:i+1, :]  # [B, 1, E]

            current_h = h[-1].unsqueeze(2)  # [B, H, 1]

            if encoded is None:
                context = torch.zeros(
                    current_input.size(0), 1, self.decoder_hidden_size,
                    device=current_input.device,
                    dtype=current_input.dtype
                )  # [B, 1, H]

            else:
                score = torch.bmm(encoded, current_h)  # [B, T, 1]
                attn_weights = torch.softmax(score, dim=1)  # [B, T, 1]

                context = torch.bmm(
                    attn_weights.transpose(1, 2),  # [B, 1, T]
                    encoded                          # [B, T, H]
                )  # [B, 1, H]

            input_decoder_novo = torch.cat([current_input, context], dim=2)  # [B, 1, E + H]

            output, (h, c) = self.LSTMDecoder(input_decoder_novo, (h, c))  # output: [B, 1, H]

            out = torch.cat([output, context], dim=2)  # [B, 1, H + H]
            logits = self.out_layer(out)  # [B, 1, vocab]

            logits_total.append(logits)

            if encoded is None:
                encoded = output  # [B, 1, H]
            else:
                encoded = torch.cat([encoded, output], dim=1)  # [B, T+1, H]

        logits_total = torch.cat(logits_total, dim=1)  # [B, L, vocab]

        return logits_total, (h, c)