#!/usr/bin/env python3
"""
Convert SavedModel to int8 TFLite using a representative dataset of tokenized sentences.

Usage:
  python src/convert_tflite.py --saved_model models/extractive_saved_model --vocab data/cnn_dm/vocab.npy --seq_len 32 --out models/extractive_int8.tflite
"""
import argparse
import json
import os
import random
import numpy as np
import tensorflow as tf

def representative_gen(vocab_path, seq_len, rep_file, num_samples=500):
    vocab = np.load(vocab_path, allow_pickle=True).tolist()
    vect = tf.keras.layers.TextVectorization(max_tokens=len(vocab), output_mode='int', output_sequence_length=seq_len)
    vect.set_vocabulary(vocab)
    # read sentences from train_sentences.jsonl if available (human-readable flattened)
    samples = []
    if os.path.exists(rep_file):
        with open(rep_file,'r',encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    samples.append(rec.get('sentence',''))
                except:
                    continue
    random.shuffle(samples)
    for s in samples[:num_samples]:
        arr = vect(np.array([s])).numpy().astype(np.int32)
        # converter expects a generator of lists (one element per input tensor)
        yield [arr]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--saved_model', required=True)
    p.add_argument('--vocab', default='data/cnn_dm/vocab.npy')
    p.add_argument('--seq_len', type=int, default=32)
    p.add_argument('--rep_file', default='data/cnn_dm/train_sentences.jsonl')
    p.add_argument('--out', default='models/extractive_int8.tflite')
    p.add_argument('--rep_samples', type=int, default=500)
    args = p.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    rep = lambda: representative_gen(args.vocab, args.seq_len, args.rep_file, num_samples=args.rep_samples)
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Use int8 for both input and output to get full integer quantization
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(args.out, 'wb') as f:
        f.write(tflite_model)
    print('Wrote', args.out, 'size bytes=', os.path.getsize(args.out))

if __name__=='__main__':
    main()