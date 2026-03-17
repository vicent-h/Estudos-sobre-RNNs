import torch
import torch.nn as nn
from lstm import LSTM
import json
from rouge_score import rouge_scorer
from tqdm import tqdm
from tokenizers import Tokenizer
from typing import List
import random
import numpy as np
import pandas as pd
from utils import Dataset

scorer = rouge_scorer.RougeScorer(
    ['rouge2'],
    use_stemmer=True
)

def _read_model(model_name):
    # print('Lendo o modelo')
    with open(f"models/{model_name}/config.json", "r", encoding="utf-8") as f:
        config: dict = json.load(f)
    model = LSTM(
        config['vocab_size'],
        config['lstm_emb_size'],
        config['lstm_num_layers'],
        config['lstm_hidden_size'],
        config['lstm_dropout']
    ).to(config['device'])
    try:
        model.load_state_dict(torch.load(f"models/{model_name}/model.pt", map_location=config['device']))
    except Exception as e:
        print(model_name, e)
    return model, config

def _read_tokenizer(config):
    # print('Lendo tokenizador')
    return Tokenizer.from_file(f"artifacts/bpe_{config['vocab_size']}.json")

def _metric(frase_correta, frase_gerada):
    global scorer
    scores = scorer.score(frase_correta, frase_gerada)
    
    return scores['rouge2'].fmeasure

def sampling(logits, option):

    logits = logits / option['temperature']

    probs = torch.softmax(logits, dim=-1)

    topk_probs, topk_indices = torch.topk(probs, option['k'], dim=-1)

    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    sampled = torch.multinomial(topk_probs, 1)

    next_token = torch.gather(topk_indices, 1, sampled)

    return next_token

@torch.no_grad()
def generate(
    model: nn.Module,
    config: dict,
    option: dict,
    tokenizer: Tokenizer,
    dataloader: torch.utils.data.DataLoader,
    len_initial_text=10
):

    model.eval()

    total_metric = 0.0
    num_batches = 0

    for batch in dataloader:

        input_tokens, target_tokens = batch

        input_tokens: torch.Tensor = input_tokens.to(config['device'])
        target_tokens: torch.Tensor = target_tokens.to(config['device'])

        states: List[: torch.Tensor] = None

        # prefixo inicial
        prefix = input_tokens[:, :len_initial_text]

        logits, states = model(prefix, states)

        generated_tokens = []

        for _ in range(target_tokens.shape[1]):

            logits = logits[:, -1, :] # [B, L, vocab_size] -> [B, 1, vocab_size]

            next_token = sampling(logits, option)

            generated_tokens.append(next_token)

            logits, states = model(next_token, states)

        pred_tokens = torch.cat(generated_tokens, dim=-1)

        # Decode batch predictions and targets
        pred_texts = tokenizer.decode_batch(pred_tokens.tolist())
        target_texts = tokenizer.decode_batch(target_tokens.tolist())

        # Compute metric for this batch
        batch_metric = 0.0
        for pred, target in zip(pred_texts, target_texts):
            batch_metric += _metric(target, pred)
        batch_metric /= len(pred_texts)

        total_metric += batch_metric
        num_batches += 1

    average_metric = total_metric / num_batches if num_batches > 0 else 0.0
    return average_metric

def _get_best_option(options):
    # print('Selecionando a melhor opção')
    best_metric = 0
    best_option = {}
    for option in options:
        if(option['metric'] > best_metric):
            best_metric = option['metric']
            best_option = option
    return best_option

def _get_dataloader(tokenizer, config_test):
    # print('Lendo dataloader')
    df_test = pd.read_parquet('/media/alvarinho/dados/Estudos/data/test_wiki_cleaned_cutted.pq')

    dataset_test = Dataset(
        df_test.text_cut.values.tolist(),
        tokenizer,
        max_len=200
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test, 
        batch_size=config_test['batch_size'],
        shuffle=True
    )
    return dataloader_test


def optimize(
        model_name, 
        num_options: int = 20,
        len_initial_texts = 256,
        batch_size = 128
    ):
    options: List[dict] = list()
    config_test = {
        'batch_size': batch_size
    }

    model, config = _read_model(model_name)
    tokenizer = _read_tokenizer(config)
    dataloader = _get_dataloader(tokenizer, config_test)


    for _ in range(num_options):
        option = {'temperature': float(random.choice(np.arange(0.1, 1.6, 0.1))),
             'k': int(random.choice(np.arange(1, 10)))}
        
        metric = generate(
            model, config,
            option, tokenizer, 
            dataloader, len_initial_texts
        )

        option['metric'] = metric

        options.append(option)


    best_option = _get_best_option(options)

    return options, best_option

def _read_models_result():
    models_result = {}
    try:
        with open('artifacts/tunning_sampling_results.json', 'r', encoding='utf-8') as f:
            models_result = json.load(f)
        print('Arquivo de resultados lido com sucesso')
    except Exception as e:
        print(e)
        pass
    return models_result

def optimize_multiple_models(num_options=20):
    models_list = ['v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10']
    models_result = _read_models_result()

    for model in tqdm(models_list, desc='Optimizing'):
        if(model in list(models_result.keys())):
            print(f'A {model} já possui resultados')
            continue
        options, best_option = optimize(model, num_options)
        print(f'Best option for {model=}: {best_option}')
        models_result[model] = {
            'options': options,
            'best_option': best_option
        }

        with open('artifacts/tunning_sampling_results.json', 'w', encoding='utf-8') as f:
            json.dump(models_result, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    optimize_multiple_models()
