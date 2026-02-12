import argparse
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from typing import List, Tuple
import pandas as pd
from lstm import LSTM, LSTMPad
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from utils import Dataset, train
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento de LSTM com BPE")

    # parâmetros do modelo
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--lstm_emb_size", type=int, default=256)
    parser.add_argument("--lstm_hidden_size", type=int, default=512)
    parser.add_argument("--lstm_num_layers", type=int, default=4)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # parâmetros de treino
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
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
        'lstm_emb_size': args.lstm_emb_size,
        'lstm_hidden_size': args.lstm_hidden_size,
        'lstm_num_layers': args.lstm_num_layers,
        'lstm_dropout': args.lstm_dropout,
        'batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,
        'accum_steps': args.accum_steps,
        'log_steps': args.log_steps,
        'eval_steps': args.eval_steps,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'device': args.device,
        'exp_name': args.exp_name
    }

    step_info = {
        'accum_steps': 0,
        'loss_sum': 0,
        'eval_loss_sum': 0,
        'eval_steps': 0,
        'global_steps': 0,
        'best_eval_loss': torch.inf,
        'eval_batches': 0
    }

    print(config)

    print('Carregando tokenizador e textos')
    tokenizer = Tokenizer.from_file(f"artifacts/bpe_{config['vocab_size']}.json")
    df_train = pd.read_parquet('../data/train_wiki_cleaned_cutted.pq')
    df_eval = pd.read_parquet('../data/eval_wiki_cleaned_cutted.pq')

    print('Preparando dataset e dataloader')
    dataset_train = Dataset(df_train.text_cut.values.tolist(), tokenizer, config['max_len'])
    dataset_eval = Dataset(df_eval.text_cut.values.tolist(), tokenizer, config['max_len'])

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
    dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=config['eval_batch_size'], shuffle=True)

    print('Declarando modelo, otimizador e writer')
    model = LSTMPad(
        config['vocab_size'],
        config['lstm_emb_size'],
        config['lstm_num_layers'],
        config['lstm_hidden_size'],
        config['lstm_dropout']
    ).to(config['device'])

    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    writer = SummaryWriter(log_dir=f'runs/{config['exp_name']}')

    n_parameters = sum(p.numel() for p in model.parameters())

    config['parameters'] = n_parameters

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
        writer
    )