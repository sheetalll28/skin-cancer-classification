#!/usr/bin/env python3
"""
Robust TFLite simulator: inspects model input dtype/quantization and converts
the tokenized input to the expected dtype before calling set_tensor.

Usage:
  python src/simulate_tflite.py --tflite models/extractive_int8.tflite --vocab data/cnn_dm/vocab.npy --seq_len 32
"""
import argparse
import numpy as np
import time
import tensorflow as tf
import os
import json

def quantize_input_if_needed(interpreter, input_index, inp):
    """Adapt `inp` (numpy array) to the interpreter's expected dtype and quantization.
    Returns the array ready to pass to interpreter.set_tensor(...).
    """
    input_details = interpreter.get_input_details()[input_index]
    expected_dtype = input_details['dtype']
    quant = input_details.get('quantization', (0.0, 0))
    scale, zero_point = quant if len(quant) == 2 else (0.0, 0)

    # If interpreter expects int8 or uint8, apply quantization formula
    if expected_dtype == np.int8 or expected_dtype == np.uint8:
        if scale == 0:
            raise ValueError('Input tensor is quantized but scale == 0. Aborting.')
        # Convert input to float32 first (inp may already be int token ids)
        f = inp.astype(np.float32)
        q = np.round(f / scale + zero_point)
        # clip to dtype range
        if expected_dtype == np.int8:
            q = np.clip(q, -128, 127).astype(np.int8)
        else:
            q = np.clip(q, 0, 255).astype(np.uint8)
        return q
    else:
        # Most common case: model expects int32 (token IDs) or float32
        return inp.astype(expected_dtype)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tflite', required=True)
    p.add_argument('--vocab', default='data/cnn_dm/vocab.npy')
    p.add_argument('--seq_len', type=int, default=32)
    p.add_argument('--sentence', default="The economy is looking stronger according to today's report.")
    args = p.parse_args()

    # Build TextVectorization with saved vocab
    vocab = np.load(args.vocab, allow_pickle=True).tolist()
    vect = tf.keras.layers.TextVectorization(output_mode='int', output_sequence_length=args.seq_len)
    vect.set_vocabulary(vocab)

    inp = vect(np.array([args.sentence])).numpy()  # typically int32 token ids
    # Try to use tflite_runtime if available
    try:
        import tflite_runtime.interpreter as tflite_rt
        Interpreter = tflite_rt.Interpreter
    except Exception:
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('Input details:', input_details)
    print('Output details:', output_details)
    # Prepare input buffer with correct dtype/quantization
    prepared = quantize_input_if_needed(interpreter, 0, inp)

    # Warmup
    interpreter.set_tensor(input_details[0]['index'], prepared)
    for _ in range(5):
        interpreter.invoke()

    # timed runs
    times=[]
    for _ in range(50):
        t0=time.time()
        interpreter.set_tensor(input_details[0]['index'], prepared)
        interpreter.invoke()
        t1=time.time()
        times.append((t1-t0)*1000.0)
    out = interpreter.get_tensor(output_details[0]['index'])
    print('Output (sample):', out)
    print('Mean ms:', np.mean(times), 'Std ms:', np.std(times))
    print('Model file size bytes:', os.path.getsize(args.tflite))

if __name__=='__main__':
    main()