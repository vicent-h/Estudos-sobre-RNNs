import argparse
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from typing import List, Tuple
import pandas as pd
from lstm import LSTMAttention, LSTM
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from utils import Dataset, LinearWarmupScheduler, log_gradient_norms
import os
import json
from tqdm import tqdm


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

    decoder_input = out[:, :-1]
    target = out[:, 1:]

    logits, states = model(input, decoder_input, states) # logits shape = [B, L, VOCAB_SIZE]

    # print(logits.size(), target.size(), 'cross')
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, config['vocab_size']), # [B * L, VOCAB_SIZE]
        target.reshape(-1),
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

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc='Evaluating'):
            input, out = batch
            input = input.to(config['device'])
            out = out.to(config['device']) # batch_size, seq_len
            decoder_input = out[:, :-1]

            n_tokens = out.size(1)
            states: Tuple[torch.Tensor, torch.Tensor] = None

            all_logits = []

            encoded, states = model.encode(input, states)
            logits, states = model.decode(out[:, :1], encoded, states)
            all_logits.append(logits)

            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            for _ in range(1, n_tokens-1):
                logits, states = model.decode(next_token.detach(), encoded, states)
                all_logits.append(logits)

                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)

            
            logits_full = torch.cat(all_logits, dim=1)
            loss = nn.functional.cross_entropy(
                logits_full.reshape(-1, config['vocab_size']),
                out[:, 1:].reshape(-1),
                ignore_index=0
            )


            states = (states[0].detach(), states[1].detach())
            logits_tf, states = model(input, decoder_input, None)
            loss_tf = nn.functional.cross_entropy(
                logits_tf.reshape(-1, config['vocab_size']),
                out[:, 1:].reshape(-1),
                ignore_index=0
            )

            step_info['eval_loss_sum'] += loss.item()
            step_info['eval_loss_tf_sum'] += loss_tf.item()
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
    , scheduler: LinearWarmupScheduler = None
):
    for epoch in range(config['epochs']):
        writer.add_scalar('epoch', epoch, step_info['global_steps'])
        for batch in tqdm(dataloader_train, total=len(dataloader_train), desc='Training'):
            train_step(model, optimizer, config, step_info, batch)

            if(step_info['accum_steps'] % config['accum_steps'] == 0):
                step_info['global_steps'] += 1
                
                log_gradient_norms(model, writer, step_info['global_steps'])

                optimizer.step()
                if(scheduler is not None):
                    scheduler.step()

                optimizer.zero_grad()
                

                avg_loss = step_info['loss_sum']
                step_info['loss_sum'] = 0
                step_info['accum_steps'] = 0
                if(step_info['global_steps'] % config['log_steps'] == 0):
                    writer.add_scalar('loss/train', avg_loss, step_info['global_steps'])
                    if(scheduler is not None):
                        writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], step_info['global_steps'])


                if(step_info['global_steps'] % config['eval_steps'] == 0 and step_info['global_steps'] > 0):
                    eval_step(model, config, step_info, dataloader_eval)

                    avg_loss_eval = step_info['eval_loss_sum'] / step_info['eval_batches']
                    avg_loss_eval_tf = step_info['eval_loss_tf_sum'] / step_info['eval_batches']
                    writer.add_scalar('loss/eval', avg_loss_eval, step_info['global_steps'])
                    writer.add_scalar('loss/eval_tf', avg_loss_eval_tf, step_info['global_steps'])

                    step_info['eval_loss_sum'] = 0
                    step_info['eval_loss_tf_sum'] = 0
                    step_info['eval_batches'] = 0

                    if(avg_loss_eval < step_info['best_eval_loss']):
                        step_info['best_eval_loss'] = avg_loss_eval

                    torch.save(model.state_dict(), f'models/{config['exp_name']}/model.pt')



def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento de LSTM com BPE")

    # parâmetros do modelo
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--lstm_emb_size", type=int, default=256)
    parser.add_argument("--encoder_hidden_size", type=int, default=512)
    parser.add_argument("--encoder_num_layers", type=int, default=4)
    parser.add_argument("--encoder_dropout", type=float, default=0.0)
    parser.add_argument("--decoder_hidden_size", type=int, default=512)
    parser.add_argument("--decoder_num_layers", type=int, default=4)
    parser.add_argument("--decoder_dropout", type=float, default=0.0)

    # parâmetros de treino
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=float, default=2_000)
    parser.add_argument("--epochs", type=int, default=1000)

    # outros
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exp_name", type=str, default="v1")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config = {
        'vocab_size': args.vocab_size,
        'max_len': args.max_len,
        'emb_size': args.lstm_emb_size,
        'encoder_hidden_size': args.encoder_hidden_size,
        'encoder_num_layers': args.encoder_num_layers,
        'encoder_dropout': args.encoder_dropout,
        'decoder_hidden_size': args.decoder_hidden_size,
        'decoder_num_layers': args.decoder_num_layers,
        'decoder_dropout': args.decoder_dropout,
        'batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,
        'accum_steps': args.accum_steps,
        'log_steps': args.log_steps,
        'eval_steps': args.eval_steps,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'device': args.device,
        'warmup_steps': args.warmup_steps,
        'exp_name': args.exp_name
        
    }

    step_info = {
        'accum_steps': 0,
        'loss_sum': 0,
        'eval_loss_sum': 0,
        'eval_loss_tf_sum': 0,
        'eval_steps': 0,
        'global_steps': 0,
        'best_eval_loss': torch.inf,
        'eval_batches': 0
    }

    print(config)
    # print('Esperando 20 minutos')
    # time.sleep(20*60)

    print('Carregando tokenizador e textos')
    tokenizer = Tokenizer.from_file(f"artifacts/bpe_{config['vocab_size']}.json")
    df_train = pd.read_parquet('data/train_wiki_cleaned_cutted.pq')
    df_eval = pd.read_parquet('data/eval_wiki_cleaned_cutted.pq')

    print('Preparando dataset e dataloader')
    dataset_train = Dataset(df_train.tokens.values.tolist(), config['max_len'])
    dataset_eval = Dataset(df_eval.tokens.values.tolist(), config['max_len'])

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
    dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=config['eval_batch_size'], shuffle=True)

    print('Declarando modelo, otimizador e writer')
    model = LSTMAttention(
        config['emb_size'],
        config['vocab_size'],
        config['encoder_num_layers'],
        config['encoder_hidden_size'],
        config['encoder_dropout'],
        config['decoder_num_layers'],
        config['decoder_hidden_size'],
        config['decoder_dropout']
    ).to(config['device'])

    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    writer = SummaryWriter(log_dir=f'runs/{config['exp_name']}')
    scheduler = LinearWarmupScheduler(config['warmup_steps'], optimizer, total_steps=500_000)

    os.makedirs(f'models/{config['exp_name']}', exist_ok=True)

    with open(f"models/{config['exp_name']}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print('Começando o treinamento')
    train(
        model,
        optimizer,
        config,
        step_info,
        dataloader_train,
        dataloader_eval,
        writer,
        scheduler
    )



