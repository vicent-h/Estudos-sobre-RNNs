import torch
import torch.nn as nn
from tokenizers import Tokenizer
from typing import List, Tuple
import pandas as pd
from lstm import LSTM
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], tokenizer: Tokenizer, max_len: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, i):
        tokens = self.tokenizer.encode(self.texts[i]).ids
        len_tokens = len(tokens)
        qtde_padding = self.max_len - len_tokens

        tokens = tokens + [0] * qtde_padding

        x = tokens[:-1]
        y = tokens[1:]

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

    
def train_step(
    model: LSTM
    , optimizer: Adam
    , config: dict
    , step_info: dict
    , batch: Tuple[torch.Tensor, torch.Tensor]
    , eos_penalty: float = 1e-9
):
    model.train()
    logits: torch.Tensor
    input, out = batch
    input = input.to(config['device'])
    out = out.to(config['device']) # batch_size, seq_len

    states: Tuple[torch.Tensor, torch.Tensor] = None

    logits, states = model(input, states)
    #logits = batch_size, seq_len, vocab_size
    logits[:, :, 2] -= eos_penalty

    _, T_out, _ = logits.shape

    out = out[:, :T_out]

    loss: torch.Tensor = nn.functional.cross_entropy(
        logits.view(-1, config['vocab_size']), #batch_size*seq_len, vocab_size
        out.reshape(-1), #batch_size*seq_len
        ignore_index=0
    )

    loss = loss / config['accum_steps']

    loss.backward()

    nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=1.0)

    step_info['loss_sum'] += loss.item()
    step_info['accum_steps'] += 1

    if states is not None:
        states = (states[0].detach(), states[1].detach())

def eval_step(
    model: LSTM
    , config: dict
    , step_info: dict
    , dataloader: Tuple[torch.Tensor, torch.Tensor]
):
    model.eval()
    logits: torch.Tensor

    for batch in tqdm(dataloader, total=len(dataloader), desc='Evaluating'):
        input, out = batch
        input = input.to(config['device'])
        out = out.to(config['device']) # batch_size, seq_len

        states: Tuple[torch.Tensor, torch.Tensor] = None

        logits, states = model(input, states)

        #logits = batch_size, seq_len, vocab_size

        _, T_out, _ = logits.shape

        out = out[:, :T_out]

        loss: torch.Tensor = nn.functional.cross_entropy(
            logits.view(-1, config['vocab_size']), #batch_size*seq_len, vocab_size
            out.reshape(-1), #batch_size*seq_len
            ignore_index=0
        )

        step_info['eval_loss_sum'] += loss.item()
        step_info['eval_batches'] += 1

        states = (states[0].detach(), states[1].detach())


def train(
    model: LSTM
    , optimizer: Adam
    , config: dict
    , step_info: dict
    , dataloader_train: torch.utils.data.DataLoader
    , dataloader_eval: torch.utils.data.DataLoader
    , writer: SummaryWriter
):
    for epoch in range(config['epochs']):
        writer.add_scalar('epoch', epoch, step_info['global_steps'])
        for batch in tqdm(dataloader_train, total=len(dataloader_train), desc='Training'):
            train_step(model, optimizer, config, step_info, batch)

            if(step_info['accum_steps'] % config['accum_steps'] == 0):
                step_info['global_steps'] += 1
                
                log_gradient_norms(model, writer, step_info['global_steps'])

                optimizer.step()
                optimizer.zero_grad()
                

                avg_loss = step_info['loss_sum']
                writer.add_scalar('loss/train', avg_loss, step_info['global_steps'])

                step_info['loss_sum'] = 0
                step_info['accum_steps'] = 0

                if(step_info['global_steps'] % config['eval_steps'] == 0 and step_info['global_steps'] > 0):
                    eval_step(model, config, step_info, dataloader_eval)

                    avg_loss_eval = step_info['eval_loss_sum'] / step_info['eval_batches']
                    writer.add_scalar('loss/eval', avg_loss_eval, step_info['global_steps'])

                    step_info['eval_loss_sum'] = 0
                    step_info['eval_batches'] = 0

                    if(avg_loss_eval < step_info['best_eval_loss']):
                        step_info['best_eval_loss'] = avg_loss_eval

                        torch.save(model.state_dict(), f'models/{config['exp_name']}/model.pt')

