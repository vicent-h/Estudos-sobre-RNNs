import torch
import torch.nn as nn
from lstm import LSTM, LSTMBidirectional, LSTMAttention
import json
from rouge_score import rouge_scorer
from tqdm import tqdm
from tokenizers import Tokenizer
from typing import List
import random
import numpy as np
import pandas as pd
from utils import Dataset
import traceback

scorer = rouge_scorer.RougeScorer(
    ['rouge2'],
    use_stemmer=True
)

models_retry = [LSTM, LSTMBidirectional, LSTMAttention]
def _read_model(model_name):
    # print('Lendo o modelo')
    with open(f"models/{model_name}/config.json", "r", encoding="utf-8") as f:
        config: dict = json.load(f)
    for M in models_retry:
        try:
            model = M(
                config['emb_size'],
                config['vocab_size'],
                config['encoder_num_layers'],
                config['encoder_hidden_size'],
                config['encoder_dropout'],
                config['decoder_num_layers'],
                config['decoder_hidden_size'],
                config['decoder_dropout']
            ).to(config['device'])
            model.load_state_dict(torch.load(f"models/{model_name}/model.pt", map_location=config['device']))
            break
        except Exception as e:
            continue
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
    model_name: str,
    model: nn.Module,
    config: dict,
    option: dict,
    tokenizer: Tokenizer,
    dataloader: torch.utils.data.DataLoader,
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

        if 'attention' in model_name:
            encoded, states = model.encode(input_tokens, states)
            logits, states = model.decode(target_tokens[:, :1], encoded, states)
        else:
            states = model.encode(input_tokens, states)
            logits, states = model.decode(target_tokens[:, :1], states)


        generated_tokens = []

        for i in range(target_tokens.shape[1]):

            logits = logits[:, -1, :] # [B, L, vocab_size] -> [B, 1, vocab_size]

            next_token = sampling(logits, option)

            generated_tokens.append(next_token)

            if 'attention' in model_name:
                logits, states = model.decode(target_tokens[:, i:i+1], encoded, states)
            else:
                logits, states = model.decode(target_tokens[:, i:i+1], states)


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
    df_test = pd.read_parquet('data/eval_wiki_cleaned_cutted.pq')

    dataset_test = Dataset(
        df_test.tokens.values.tolist(),
        max_len=512
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
        batch_size = 256
    ):
    options: List[dict] = list()
    batch_sizes = [64, 32, 16, 8]
    for batch_size in batch_sizes:
        try:
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
                    model_name, model, config,
                    option, tokenizer, 
                    dataloader
                )

                option['metric'] = metric

                options.append(option)


            best_option = _get_best_option(options)

            return options, best_option
        except Exception as e:
            traceback.print_exc()
            print(e)
            pass
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

def optimize_multiple_models(num_options=100):
    models_list = [
        'v1', 'v2', 'v3_bidirectional', 'v4_bidirectional_warmup', 'v5_attention', 
        'v6_attention_decoder_too', 'v7_bidirectional_warmup'
    ]
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
