#!/usr/bin/env python3
"""
End-to-end TFLite extractive summarizer demo (file IO).

- Reads article from --article_file (required)
- Tokenizes sentences using saved TextVectorization vocabulary (vocab.npy)
- Runs the TFLite model to score each sentence
- Selects top-k sentences (keeps original order) and writes the summary to --out_file (required)
- Also writes JSON metadata (selected indices, per-sentence scores, elapsed time)

Usage:
  python src/tflite_infer_demo.py \
    --tflite models/extractive_int8.tflite \
    --vocab data/cnn_dm/vocab.npy \
    --article_file examples/article.txt \
    --out_file out/summary.json \
    --top_k 3
"""
import argparse
import os
import json
import numpy as np
import time
import nltk

# Prefer tflite_runtime for smaller installs, fallback to TF
try:
    import tflite_runtime.interpreter as tflite_rt  # type: ignore
    Interpreter = tflite_rt.Interpreter  # type: ignore
except Exception:
    import tensorflow as tf  # type: ignore
    Interpreter = tf.lite.Interpreter  # type: ignore
    from tensorflow.keras.layers import TextVectorization  # type: ignore

# If TF is not present, TextVectorization will be None and we'll error later if needed
try:
    from tensorflow.keras.layers import TextVectorization  # type: ignore
except Exception:
    TextVectorization = None  # type: ignore

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize  # noqa


def load_article_from_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Article file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def build_vectorizer_from_vocab(vocab_path, seq_len):
    vocab = np.load(vocab_path, allow_pickle=True).tolist()
    if TextVectorization is None:
        raise RuntimeError("TextVectorization not available (TensorFlow not installed). Install tensorflow.")
    vect = TextVectorization(output_mode='int', output_sequence_length=seq_len)
    vect.set_vocabulary(vocab)
    return vect


def prepare_input_for_interpreter(interpreter, token_array):
    input_details = interpreter.get_input_details()[0]
    expected_dtype = input_details['dtype']
    quant = input_details.get('quantization', (0.0, 0))
    scale, zero_point = quant if len(quant) == 2 else (0.0, 0)

    if expected_dtype == np.int8 or expected_dtype == np.uint8:
        if scale == 0:
            raise ValueError("Interpreter expects quantized inputs but scale==0")
        f = token_array.astype(np.float32)
        q = np.round(f / scale + zero_point)
        if expected_dtype == np.int8:
            q = np.clip(q, -128, 127).astype(np.int8)
        else:
            q = np.clip(q, 0, 255).astype(np.uint8)
        return q
    else:
        return token_array.astype(expected_dtype)


def dequantize_output(output_array, output_details):
    od = output_details[0]
    out_dtype = od['dtype']
    quant = od.get('quantization', (0.0, 0))
    scale, zero_point = quant if len(quant) == 2 else (0.0, 0)
    if out_dtype in (np.int8, np.uint8):
        return (output_array.astype(np.float32) - zero_point) * scale
    else:
        return output_array.astype(np.float32)


def score_sentences_with_tflite(interpreter, tokenized_sentences):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    prepared = prepare_input_for_interpreter(interpreter, tokenized_sentences)

    # Try batch inference first
    try:
        interpreter.set_tensor(input_details[0]['index'], prepared)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        scores = dequantize_output(out, output_details).reshape(-1)
        return scores
    except Exception:
        # Fall back to per-sentence inference
        scores = []
        for i in range(prepared.shape[0]):
            single = prepared[i : i + 1]
            interpreter.set_tensor(input_details[0]['index'], single)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])
            s = dequantize_output(out, output_details).reshape(-1)[0]
            scores.append(float(s))
        return np.array(scores, dtype=np.float32)


def write_summary_out(out_path, summary, selected_indices, selected_sentences, scores, elapsed):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    data = {
        "summary": summary,
        "selected_indices": selected_indices,
        "selected_sentences": selected_sentences,
        "scores": [float(s) for s in scores],
        "inference_elapsed_seconds": float(elapsed),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite', required=True, help='Path to .tflite model')
    parser.add_argument('--vocab', required=True, help='Path to vocab.npy (TextVectorization vocabulary)')
    parser.add_argument('--article_file', required=True, help='Path to article text file (input)')
    parser.add_argument('--out_file', required=True, help='Path to output JSON file (summary + metadata)')
    parser.add_argument('--seq_len', type=int, default=32, help='Token sequence length expected by model')
    parser.add_argument('--top_k', type=int, default=3, help='Number of sentences to select for summary')
    parser.add_argument('--max_sentences', type=int, default=50, help='Truncate article to this many sentences for speed')
    args = parser.parse_args()

    article = load_article_from_file(args.article_file)
    if not article:
        raise SystemExit("Article file is empty or unreadable.")

    sentences = sent_tokenize(article)
    if len(sentences) == 0:
        raise SystemExit("No sentences found in article.")
    if len(sentences) > args.max_sentences:
        sentences = sentences[: args.max_sentences]

    vect = build_vectorizer_from_vocab(args.vocab, args.seq_len)
    tokenized = vect(np.array(sentences)).numpy().astype(np.int32)

    interpreter = Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()

    start = time.time()
    scores = score_sentences_with_tflite(interpreter, tokenized)
    elapsed = time.time() - start

    k = min(args.top_k, len(sentences))
    top_idx = np.argsort(scores)[-k:][::-1]  # highest first
    selected_indices = sorted(top_idx)
    selected_sentences = [sentences[i] for i in selected_indices]
    summary = " ".join(selected_sentences)

    # Console output (brief)
    print("\n=== Extractive summary (top_k=%d) ===\n" % (k))
    for i in selected_indices:
        print(f"[{i}] score={scores[i]:.4f}  {sentences[i]}")
    print("\nSummary written to:", args.out_file)
    print(f"(Inference time for scoring: {elapsed:.4f}s for {len(sentences)} sentences)\n")

    # Write JSON out file
    def write_summary_out(out_path, summary, selected_indices, selected_sentences, scores, elapsed):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # Convert to native Python types for JSON serialization
        sel_idx_py = [int(i) for i in (selected_indices if isinstance(selected_indices, (list, tuple, np.ndarray)) else [selected_indices])]
        scores_py = []
        # handle numpy arrays, lists, or single values
        if isinstance(scores, np.ndarray):
            scores_py = [float(x) for x in scores.tolist()]
        elif isinstance(scores, (list, tuple)):
            scores_py = [float(x) for x in scores]
        else:
            # single numeric value
            scores_py = [float(scores)]

        data = {
            "summary": summary,
            "selected_indices": sel_idx_py,
            "selected_sentences": selected_sentences,
            "scores": scores_py,
            "inference_elapsed_seconds": float(elapsed),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    write_summary_out(args.out_file, summary, selected_indices, selected_sentences, scores.tolist(), elapsed)


if __name__ == "__main__":
    main()