#!/usr/bin/env python3
"""
Load the saved model, build summaries by selecting top-k scored sentences for each article,
and compute average ROUGE-1/2/L (F1) over the labeled test set.

Usage:
  python src/evaluate_and_infer.py --model_dir models/extractive_saved_model --data_dir data/cnn_dm --vocab_path data/cnn_dm/vocab.npy --seq_len 32 --top_k 3
"""
import argparse
import json
import os
import numpy as np
import tensorflow as tf
from rouge_score import rouge_scorer
from tqdm import tqdm

def load_jsonl(path):
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for l in f:
            out.append(json.loads(l))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True)
    p.add_argument('--data_dir', default='data/cnn_dm')
    p.add_argument('--vocab_path', default='data/cnn_dm/vocab.npy')
    p.add_argument('--seq_len', type=int, default=32)
    p.add_argument('--top_k', type=int, default=3)
    args = p.parse_args()

    model = tf.keras.models.load_model(args.model_dir)
    # build TextVectorization with saved vocab to tokenize sentences at eval time
    vect = tf.keras.layers.TextVectorization(output_mode='int', output_sequence_length=args.seq_len)
    vocab = np.load(args.vocab_path, allow_pickle=True).tolist()
    vect.set_vocabulary(vocab)

    test_path = os.path.join(args.data_dir, 'test_labeled.jsonl')
    if not os.path.exists(test_path):
        raise SystemExit(f"Missing {test_path} - make sure Step 2 produced test_labeled.jsonl")

    test = load_jsonl(test_path)
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

    scores = []
    for item in tqdm(test, desc='Evaluating'):
        sents = item['sentences']
        if len(sents)==0:
            continue
        tokenized = vect(np.array(sents)).numpy()
        # predict per-sentence scores
        preds = model.predict(tokenized, verbose=0).flatten()
        # select top_k indices by score
        k = min(args.top_k, len(preds))
        top_idx = np.argsort(preds)[-k:][::-1]
        # keep original order for readability
        selected = sorted(top_idx)
        summary = ' '.join([sents[i] for i in selected])
        ref = item.get('summary','')
        r = scorer.score(ref, summary)
        scores.append(r)

    if len(scores)==0:
        print('No examples scored.')
        return
    import statistics
    print('Avg ROUGE-1 F:', statistics.mean([s['rouge1'].fmeasure for s in scores]))
    print('Avg ROUGE-2 F:', statistics.mean([s['rouge2'].fmeasure for s in scores]))
    print('Avg ROUGE-L F:', statistics.mean([s['rougeL'].fmeasure for s in scores]))

if __name__=='__main__':
    main()