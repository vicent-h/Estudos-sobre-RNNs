from typing import List

class Tokenizer:
    def __init__(self, vocab, special_tokens):

        self.vocab = vocab
        if(special_tokens):
            self.special_tokens = special_tokens
        else:
            self.special_tokens = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        self.pad_token_id = self.special_tokens['<pad>']
        self.bos_token_id = self.special_tokens['<bos>']
        self.eos_token_id = self.special_tokens['<eos>']
        self.unk_token_id = self.special_tokens['<unk>']
        self.tokenize_dict = self.special_tokens.copy()
        self.detokenize_dict = {v: k for k, v in self.special_tokens.items()}
        self.tokenize_dict.update({v: i+len(self.special_tokens) for i, v in enumerate(self.vocab)})
        self.detokenize_dict.update({i+len(self.special_tokens): v for i, v in enumerate(self.vocab)})
        self.vocab_size = len(self.vocab) + len(self.special_tokens)

    def tokenize_individual(self, x: str):
        if(x in self.vocab):
            return self.tokenize_dict[x]
        else:
            return self.unk_token_id
        
    def detokenize_individual(self, i: int):
        return self.detokenize_dict[i]
    
    def encode(self, x: str):
        tokens = []
        tokens.append(self.special_tokens['<bos>'])
        for i in x:
            tokens.append(self.tokenize_individual(i))
        tokens.append(self.special_tokens['<eos>'])
        return tokens
    
    def decode(self, i: List[int]):
        detokens = []
        for x in i:
            detokens.append(self.detokenize_individual(x))
        return ''.join(detokens)
    
    def to_dict(self):
        return {
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            vocab=data["vocab"],
            special_tokens=data["special_tokens"],
        )
