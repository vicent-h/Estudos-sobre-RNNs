import torch
import torch.nn as nn
from tokenizers import Tokenizer
from typing import List, Tuple
import pandas as pd
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, warmup_steps, optimizer, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if(step <= self.warmup_steps):
            # scalar de 0 até o warmup_steps
            scale = step / float(self.warmup_steps)
        else:
            #passando do warmup steps, deve decair até o o total_steps

            # o max serve para caso passe o warmup = total_steps, não dividir por 0
            #subtrair do warmup para resetar a porcentagem a partir do ponto do próprio warmup
            progress = (step - self.warmup_steps) / (
                max(1, self.total_steps - self.warmup_steps))

            #a escala deve ser o complementar do progresso, para começar do topo e ir decaindo
            scale = 1.0 - progress
            scale = max(scale, 0)

        return [base_lr * scale for base_lr in self.base_lrs]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokens: List[str], max_len: int):
        self.tokens = tokens
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, i):
        tokens = list(self.tokens[i])
        len_tokens = len(tokens)

        def padding(tokens, max_len):

            len_tokens = len(tokens)
            qtde_padding = max(0 , max_len - len_tokens)


            if (qtde_padding):
                tokens = tokens + [0] * qtde_padding
            tokens = tokens[:max_len]
            return tokens


        self.len_divided = len_tokens // 2

        x = padding(list(tokens[:self.len_divided]), self.max_len // 2)
        y = padding(list(tokens[self.len_divided:]), self.max_len // 2)
        
        return torch.tensor(x), torch.tensor(y)
    
import torch

def log_gradient_norms(
    model,
    writer,
    step: int,
    prefix: str = "grad_norm",
    norm_type: float = 2.0
):


    total_norm = 0.0
    param_norms = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_norm = param.grad.data.norm(norm_type)
        param_norms[name] = grad_norm.item()
        total_norm += grad_norm ** norm_type

    total_norm = total_norm ** (1.0 / norm_type)

    writer.add_scalar(f"{prefix}/total", total_norm.item(), step)

    for name, norm in param_norms.items():
        writer.add_scalar(f"{prefix}/per_param/{name}", norm, step)

    return total_norm.item(), param_norms

    
