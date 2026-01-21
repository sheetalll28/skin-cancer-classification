#!/usr/bin/env python3
"""
Flatten labeled article->sentence data, build a TextVectorization on train sentences,
tokenize each sentence, and save:

Outputs (under data/cnn_dm):
 - vocab.npy                (vocabulary list from TextVectorization)
 - train_sentences.npz      (x: token ids array, y: labels array)
 - validation_sentences.npz
 - test_sentences.npz
 - train_sentences.jsonl    (human-readable flattened sentences)
 - validation_sentences.jsonl
 - test_sentences.jsonl

Usage:
  python src/build_vectorizer_and_flatten.py --data_dir data/cnn_dm --max_tokens 4000 --seq_len 32
"""
import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def load_labeled_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def flatten_split(items):
    sentences = []
    labels = []
    metas = []
    for item in items:
        aid = item.get('id', '')
        sents = item.get('sentences', [])
        labs = item.get('labels', [])
        # Ensure lengths match
        for i, s in enumerate(sents):
            lbl = labs[i] if i < len(labs) else 0
            sentences.append(s)
            labels.append(int(lbl))
            metas.append({'article_id': aid, 'sent_index': i})
    return sentences, np.array(labels, dtype=np.int8), metas

def save_flat_jsonl(sentences, labels, metas, out_path):
    with open(out_path, 'w', encoding='utf-8') as fout:
        for i, s in enumerate(sentences):
            rec = {'id': f'{i}', 'sentence': s, 'label': int(labels[i]), 'meta': metas[i]}
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/cnn_dm', help='labeled JSONL dir')
    parser.add_argument('--max_tokens', type=int, default=4000, help='vocab size for TextVectorization')
    parser.add_argument('--seq_len', type=int, default=32, help='token sequence length per sentence')
    parser.add_argument('--out_dir', default='data/cnn_dm', help='where to save outputs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Input labeled files produced in Step 2
    in_train = os.path.join(args.data_dir, 'train_labeled.jsonl')
    in_val = os.path.join(args.data_dir, 'validation_labeled.jsonl')
    in_test = os.path.join(args.data_dir, 'test_labeled.jsonl')

    # Load
    train_items = load_labeled_jsonl(in_train) if os.path.exists(in_train) else []
    val_items = load_labeled_jsonl(in_val) if os.path.exists(in_val) else []
    test_items = load_labeled_jsonl(in_test) if os.path.exists(in_test) else []

    if len(train_items) == 0:
        print('No train_labeled.jsonl found or file empty. Make sure Step 2 produced it.')
        return

    # Flatten into sentence lists
    train_sents, train_labels, train_metas = flatten_split(train_items)
    val_sents, val_labels, val_metas = flatten_split(val_items) if len(val_items) else ([], np.array([]), [])
    test_sents, test_labels, test_metas = flatten_split(test_items) if len(test_items) else ([], np.array([]), [])

    print(f'Flattened: train={len(train_sents)} val={len(val_sents)} test={len(test_sents)} sentences')

    # Save human-readable flattened files (optional, helpful for inspection)
    save_flat_jsonl(train_sents, train_labels, train_metas, os.path.join(args.out_dir, 'train_sentences.jsonl'))
    if len(val_sents):
        save_flat_jsonl(val_sents, val_labels, val_metas, os.path.join(args.out_dir, 'validation_sentences.jsonl'))
    if len(test_sents):
        save_flat_jsonl(test_sents, test_labels, test_metas, os.path.join(args.out_dir, 'test_sentences.jsonl'))

    # Build TextVectorization on training sentences
    vectorizer = layers.TextVectorization(max_tokens=args.max_tokens,
                                          output_mode='int',
                                          output_sequence_length=args.seq_len)
    # adapt expects a tf.data or numpy array of strings
    print('Adapting TextVectorization to training sentences (this may take a moment)...')
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_sents).batch(1024))
    vocab = vectorizer.get_vocabulary()
    np.save(os.path.join(args.out_dir, 'vocab.npy'), np.array(vocab))
    print(f'Saved vocabulary (size {len(vocab)}) to {os.path.join(args.out_dir, "vocab.npy")}')

    # Tokenize (transform) and save token arrays as compressed .npz
    def tokenize_and_save(sentences, labels, out_npz_path, vect):
        if len(sentences) == 0:
            return
        arr = vect(np.array(sentences)).numpy().astype(np.int32)
        # Save compressed
        np.savez_compressed(out_npz_path, x=arr, y=labels)
        print(f'Saved tokenized data to {out_npz_path}; x.shape={arr.shape}, y.shape={labels.shape}')

    # Save train/val/test tokenized arrays
    tokenize_and_save(train_sents, train_labels, os.path.join(args.out_dir, 'train_sentences.npz'), vectorizer)
    if len(val_sents):
        tokenize_and_save(val_sents, val_labels, os.path.join(args.out_dir, 'validation_sentences.npz'), vectorizer)
    if len(test_sents):
        tokenize_and_save(test_sents, test_labels, os.path.join(args.out_dir, 'test_sentences.npz'), vectorizer)

    # For convenience, save a small stats file
    stats = {
        'train_sentences': len(train_sents),
        'validation_sentences': len(val_sents),
        'test_sentences': len(test_sents),
        'vocab_size': len(vocab),
        'seq_len': args.seq_len
    }
    with open(os.path.join(args.out_dir, 'flatten_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print('Wrote flatten_stats.json')

if __name__ == '__main__':
    main()