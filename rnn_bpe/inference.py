import pandas as pd
import numpy as np
from itertools import chain
from collections import Counter
from rouge_score import rouge_scorer
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import Tokenizer

def evaluate_models(df):

    
    df = df.copy()
    
    # =========================
    # 🔹 Funções auxiliares
    # =========================
    
    def repetition_rate(tokens):
        if len(tokens) == 0:
            return 0
        unique_tokens = len(set(tokens))
        return 1 - (unique_tokens / len(tokens))
    
    def ngram_repetition(tokens, n=2):
        if len(tokens) < n:
            return 0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        return 1 - (unique / total)
    
    def token_overlap(row):
        orig = set(row['tokens_originais'])
        pred = set(row['tokens_preditados'])
        if len(pred) == 0:
            return 0
        return len(orig & pred) / len(pred)
    
    def vocab_diversity(group):
        all_tokens = list(chain.from_iterable(group['tokens_preditados']))
        if len(all_tokens) == 0:
            return 0
        return len(set(all_tokens)) / len(all_tokens)
    
    def compute_rouge_scores(df):
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        rouge1, rouge2, rougeL = [], [], []
        
        for orig, pred in zip(df['frase_original'], df['frase_preditada']):
            scores = scorer.score(orig, pred)
            rouge1.append(scores['rouge1'].fmeasure)
            rouge2.append(scores['rouge2'].fmeasure)
            rougeL.append(scores['rougeL'].fmeasure)
        
        df['rouge1'] = rouge1
        df['rouge2'] = rouge2
        df['rougeL'] = rougeL
        
        return df
    
    # =========================
    # 🔹 Métricas básicas
    # =========================
    
    df['len_original_words'] = df['frase_original'].str.split().apply(len)
    df['len_pred_words'] = df['frase_preditada'].str.split().apply(len)
    
    df['len_original_tokens'] = df['tokens_originais'].apply(len)
    df['len_pred_tokens'] = df['tokens_preditados'].apply(len)
    
    df['length_ratio_words'] = df['len_pred_words'] / df['len_original_words'].replace(0, 1)
    df['length_ratio_tokens'] = df['len_pred_tokens'] / df['len_original_tokens'].replace(0, 1)
    
    # =========================
    # 🔹 Repetições
    # =========================
    
    df['token_repetition_rate'] = df['tokens_preditados'].apply(repetition_rate)
    df['bigram_repetition'] = df['tokens_preditados'].apply(lambda x: ngram_repetition(x, 2))
    df['trigram_repetition'] = df['tokens_preditados'].apply(lambda x: ngram_repetition(x, 3))
    
    # =========================
    # 🔹 Overlap
    # =========================
    
    df['token_overlap'] = df.apply(token_overlap, axis=1)
    
    # =========================
    # 🔹 ROUGE
    # =========================
    
    df = compute_rouge_scores(df)
    
    # =========================
    # 🔹 Agregações por modelo
    # =========================
    
    metrics_mean = df.groupby('modelo').agg({
        'loss': 'mean',
        'len_original_words': 'mean',
        'len_pred_words': 'mean',
        'len_original_tokens': 'mean',
        'len_pred_tokens': 'mean',
        'length_ratio_words': 'mean',
        'length_ratio_tokens': 'mean',
        'token_repetition_rate': 'mean',
        'bigram_repetition': 'mean',
        'trigram_repetition': 'mean',
        'rouge1': 'mean',
        'rouge2': 'mean',
        'rougeL': 'mean',
        'token_overlap': 'mean'
    }).reset_index()
    
    # =========================
    # 🔹 Diversidade global
    # =========================
    
    diversity = df.groupby('modelo').apply(vocab_diversity).reset_index(name='vocab_diversity')
    
    # =========================
    # 🔹 Correlação loss vs repetição
    # =========================
    
    correlation = (
        df.groupby('modelo')
        .apply(lambda g: g[['loss', 'token_repetition_rate']].corr().iloc[0,1])
        .reset_index(name='corr_loss_repetition')
    )
    
    # =========================
    # 🔹 Consolidação final
    # =========================
    
    final_metrics = (
        metrics_mean
        .merge(diversity, on='modelo')
        .merge(correlation, on='modelo')
    )
    
    return final_metrics, df


def test_step(config, step_info, model, dataloader, vocab_size, tok: Tokenizer):
    model.eval()
    phrases_pred, tokens_pred_total = [], []
    phrases_input, tokens_input_total = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc='Testing'):
            
            input_tokens, output_tokens = batch
            input_tokens: torch.Tensor = input_tokens.to(config['device'])
            output_tokens = output_tokens.to(config['device'])

            states = None
            predict_tokens, states = model(input_tokens, states) #Batch, Seq Len, Vocab Size
            pred_tokens = predict_tokens

            predict_tokens = predict_tokens.view(-1, vocab_size) # Batch*Seq Len, Vocab Size
            output_tokens = output_tokens.view(-1)

            loss = nn.functional.cross_entropy(predict_tokens, output_tokens, ignore_index=0)

            step_info['test_accum_steps'] += 1
            step_info['test_loss_sum'] += loss.item()

            if(states):
                states = (states[0].detach(), states[1].detach())

            pred_tokens = torch.argmax(pred_tokens, dim=-1)
            for i in range(len(pred_tokens)):
                p_tokens = pred_tokens[i].tolist()
                p2_tokens = input_tokens[i].tolist()
                p = tok.decode(p_tokens)
                p2 = tok.decode(p2_tokens)
                phrases_pred.append(p)
                tokens_pred_total.append(p_tokens)
                phrases_input.append(p2)
                tokens_input_total.append(p2_tokens)


        avg_loss = step_info['test_loss_sum'] / step_info['test_accum_steps']
        
        step_info['test_loss_sum'] = 0
        step_info['test_accum_steps'] = 0

        return avg_loss, phrases_pred, phrases_input, tokens_pred_total, tokens_input_total
    

def predicting(
    initial_text: str,
    model,
    max_len,
    tok,
    eos_penalty_base: float,
    temperature: float,
    k=1,
    device="cpu"
):
    model.eval()

    eos_id = tok.token_to_id("[EOS]")

    enc = tok.encode(initial_text).ids
    enc = torch.tensor(enc, dtype=torch.long, device=device)

    enc = enc.unsqueeze(0)  # (1, seq_len)

    states = None

    with torch.no_grad():

        # primeira passada com o prompt inteiro
        logits, states = model(enc, states)

        for i in range(max_len):

            logits_step = logits[:, -1, :]  # último token
            logits_step = logits_step.squeeze(0)

            # penalização mais estável
            # penalty = eos_penalty_base * (1 - i / max_len)
            # logits_step[eos_id] -= penalty

            logits_step = logits_step / temperature

            probs = torch.softmax(logits_step, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, k=k)

            topk_probs = topk_probs / topk_probs.sum()

            next_token = topk_indices[torch.multinomial(topk_probs, 1)]

            if next_token.item() == eos_id:
                break

            # prepara próximo passo (somente último token!)
            next_token_input = next_token.unsqueeze(0)

            logits, states = model(next_token_input, states)

            enc = torch.cat([enc, next_token.unsqueeze(0)], dim=1)

    text = tok.decode(enc.squeeze(0).tolist())

    text = text.replace(" ##", "").replace("##", "")

    return text



