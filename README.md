# Tiny Extractive Summarizer — TinyML Demo

A compact extractive summarization pipeline and TinyML demonstration.  
This repository shows how to go from labeled article/sentence data to a tiny sentence-scoring model, how to quantize it to a small TFLite binary, and how to run an end-to-end TFLite inference demo that produces extractive summaries.

Goals
- Prepare and flatten article→sentence labeled data.
- Build a TextVectorization vocabulary and tokenize sentences.
- Train a small sentence-scoring model (Embedding → GAP → Dense → Sigmoid).
- Convert the trained SavedModel to a compact, integer-quantized TFLite model.
- Provide a TFLite-based end-to-end demo that reads an article file, scores sentences with the TFLite model, selects top-k sentences, and writes the summary + metadata to a JSON file.
- Focus on TinyML-friendly design: small model size, integer quantization, and runtime that can run on resource-constrained devices.

Contents
- src/prep_and_label.py
  - (Your Step 1/2 script) Prepares and labels articles, produces:
    - data/cnn_dm/train_labeled.jsonl
    - data/cnn_dm/validation_labeled.jsonl
    - data/cnn_dm/test_labeled.jsonl
- src/build_vectorizer_and_flatten.py
  - Step 3: flatten article-level JSONL into sentence-level JSONL, build TextVectorization, save:
    - data/cnn_dm/vocab.npy
    - data/cnn_dm/train_sentences.npz (x,y)
    - data/cnn_dm/validation_sentences.npz
    - data/cnn_dm/test_sentences.npz
    - data/cnn_dm/train_sentences.jsonl (human-readable)
    - data/cnn_dm/flatten_stats.json
- src/train_extractive.py
  - Step 4: train the tiny extractive model and save SavedModel to models/extractive_saved_model
- src/test_loaded_model.py
  - Small script to load SavedModel and run a quick prediction to verify training outputs
- src/evaluate_and_infer.py
  - Evaluate extractive model by building top-k summaries and computing average ROUGE scores
- src/convert_tflite.py
  - Convert SavedModel → TFLite (integer quantization) using a representative dataset
- src/simulate_tflite.py
  - Load .tflite model locally and measure inference latency; robust to input dtype/quantization details
- src/tflite_infer_demo.py
  - Final demo: read article file, split into sentences, tokenize, run TFLite scoring, select top-k, write JSON output
  - Two versions may exist depending on whether TensorFlow's TextVectorization is available:
    - TF-backed vectorizer (requires TensorFlow installed)
    - TF-free vectorizer (pure-Python tokenizer + vocab mapping) — preferable if you want a lightweight runtime

Quick start (recommended)
1. Create and activate a Python virtual environment (use a TensorFlow-supported Python version, recommended: 3.10 or 3.11)

PowerShell (recommended):
```powershell
# remove old venv if needed
Remove-Item -Recurse -Force .\venv

# create with py launcher (if Python 3.10 installed)
py -3.10 -m venv venv

# activate (PowerShell)
. .\venv\Scripts\Activate.ps1
```

2. Install Python packages (inside venv)
```bash
pip install --upgrade pip
pip install tensorflow numpy tqdm rouge-score nltk
# optional for tiny runtime tests:
pip install tflite-runtime==2.11.0  # or skip and use tf.lite.Interpreter
```

If you cannot install TensorFlow (very large), use the TF-free demo variant and only install tflite-runtime + numpy + nltk.

3. Step 1/2 — Prepare labeled files
(If you already have prepared files skip this)
```bash
python src/prep_and_label.py --limit 2000
# or run your repository's script for production with no --limit
```
After successful run you should see:
- data/cnn_dm/train_labeled.jsonl
- data/cnn_dm/validation_labeled.jsonl
- data/cnn_dm/test_labeled.jsonl

4. Step 3 — Build vectorizer, flatten, and tokenize
```bash
python src/build_vectorizer_and_flatten.py --data_dir data/cnn_dm --max_tokens 4000 --seq_len 32
```
Outputs (data/cnn_dm):
- vocab.npy
- train_sentences.npz (x: shape (N, seq_len), y: shape (N,))
- validation_sentences.npz (optional)
- train_sentences.jsonl (human-readable)
- flatten_stats.json

5. Step 4 — Train the tiny extractive model
```bash
python src/train_extractive.py --data_dir data/cnn_dm --models_dir models --epochs 10 --batch_size 256
```
This produces:
- models/extractive_saved_model (SavedModel directory)
- models/best.keras (optional best checkpoint)

Quick verify:
```bash
python src/test_loaded_model.py
# prints model loaded, sample predictions, etc.
```

6. Step 5 — Evaluate (ROUGE) and convert to TFLite
Evaluate with the SavedModel (ROUGE):
```bash
python src/evaluate_and_infer.py --model_dir models/extractive_saved_model --data_dir data/cnn_dm --vocab_path data/cnn_dm/vocab.npy --seq_len 32 --top_k 3
```

Convert to int8 TFLite (uses train_sentences.jsonl as representative file):
```bash
python src/convert_tflite.py --saved_model models/extractive_saved_model --vocab data/cnn_dm/vocab.npy --seq_len 32 --rep_samples 500 --out models/extractive_int8.tflite
```

Test the TFLite model (simulate runtime + latency):
```bash
python src/simulate_tflite.py --tflite models/extractive_int8.tflite --vocab data/cnn_dm/vocab.npy --seq_len 32
```
Notes:
- The simulate script adapts to the TFLite model's input dtype. If the TFLite model expects int32 (common when inputs are token IDs), simulated input will be cast to int32. If model expects int8 inputs, simulate quantizes token ids to int8 using the interpreter's quantization params (scale, zero_point).
- The output of a quantized model may be int8; dequantize using:
  float_value = (q_value - zero_point) * scale

7. End-to-end TFLite demo (read article file → write summary JSON)
Single-line command (from repo root, with venv activated):
```bash
python src/tflite_infer_demo.py --tflite models/extractive_int8.tflite --vocab data/cnn_dm/vocab.npy --article_file examples/article.txt --out_file out/summary.json --top_k 3
```
- The demo writes JSON to out/summary.json with structure:
```json
{
  "summary": "...",
  "selected_indices": [2, 5, 7],
  "selected_sentences": ["...", "...", "..."],
  "scores": [0.413, 0.378, 0.362],
  "inference_elapsed_seconds": 0.0123,
  "generated_at": "2026-01-18T00:48:53"
}
```

Troubleshooting & common pitfalls
- PowerShell vs Bash one-liners
  - Do not paste Bash here-docs into PowerShell. PowerShell will parse Python code as shell code and throw parser errors. Use python script files or single-line python -c "..." commands with careful quoting.
- Python version / TensorFlow compatibility
  - TensorFlow provides wheels for specific Python versions (commonly 3.8–3.11 for TF 2.x). Using very new Python (e.g., 3.14) will cause "No matching distribution found for tensorflow".
  - Check installed Pythons:
    - py -0p (Windows py launcher)
    - where.exe python
  - Recreate venv pointing to a supported interpreter:
    - py -3.10 -m venv venv
- Virtualenv uses the interpreter used to create it
  - If your venv shows Python 3.14 inside VS Code but your global python is 3.7, that venv was created with a different interpreter. Delete and recreate the venv with the desired interpreter.
- TFLite dtype mismatch error
  - ValueError: Cannot set tensor: Got value of type INT8 but expected type INT32...
  - Inspect interpreter.get_input_details() and convert your input to the expected dtype. The simulate/demo scripts do this for you automatically.
- json.dump TypeError: numpy int/float is not JSON serializable
  - Convert numpy types to native Python types before dumping: e.g., scores_py = [float(x) for x in scores]

TinyML-specific concepts and guidance (detailed)
TinyML is about running ML on resource constrained devices (microcontrollers, tiny CPUs). Key considerations:

1) Model size
- The TFLite int8 model you produce is tiny (example output ~68 KB). Small models are required to fit in flash memory on microcontrollers.
- Strategies to reduce size:
  - Use small embedding dims (8–32), small dense layers (16–128 units).
  - Reduce vocab size (use hashing or smaller vocab) if possible.
  - Prune weights or use weight clustering (TensorFlow Model Optimization Toolkit).
  - Convert to int8 TFLite (post-training quantization) — reduces size and improves CPU performance.

2) Quantization
- Post-training integer quantization converts weights and activations to int8 and optionally inputs/outputs to int8.
- Converter configuration:
  - converter.optimizations = [tf.lite.Optimize.DEFAULT]
  - converter.representative_dataset = <generator> — REQUIRED for full integer quantization (activations)
  - converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  - converter.inference_input_type and converter.inference_output_type control model I/O dtypes (int8 vs int32/float32)
- At runtime, if model inputs/outputs are quantized, you must quantize inputs and dequantize outputs using the interpreter's quantization params (scale, zero_point).
  - Dequantization: float = (q - zero_point) * scale

3) Input representation
- For text models, inputs are token IDs (integers). Two common choices for TinyML:
  - Keep inputs as int32 token IDs (easier — no input quantization). Internals can be quantized — this is a common compromise.
  - Quantize inputs to int8 if you want a fully integer pipeline (saves memory/ops), but you must ensure runtime code performs the correct quantization of token IDs (less common and trickier).
- In this project the TFLite model produced int32 inputs and int8 outputs (internals quantized). The demo handles either case.

4) Runtime
- On-device runtime options:
  - TensorFlow Lite (full Python TF.SE) — heavy for edge devices; good for desktops and servers.
  - tflite-runtime (a smaller, standalone interpreter) ��� better for constrained Linux devices (Raspberry Pi).
  - Microcontroller (C) TFLite Micro — for MCUs. For this, you will need a C implementation and likely convert model to a C array with xxd or using the TFLite Micro tools.
- Delegate support (XNNPACK on CPU, NNAPI, GPU) can accelerate inference on larger devices.

5) Measuring performance
- Use stable timings: run many iterations (>50) after a warm-up and report mean and std in ms.
- For microcontrollers, measure wall-clock with device timers or host-side serial timestamps.

6) Evaluation and quantization-aware validation
- Evaluate model performance (ROUGE) using the full SavedModel before conversion.
- After quantization, compare SavedModel vs TFLite predictions on a sample set:
  - Compute MSE and Pearson correlation between predicted scores.
  - Build summaries with both and compute ROUGE to detect metric drift.
- If quantization hurts metrics too much, consider:
  - Quantization-aware training (QAT)
  - Smaller learning rate / fine-tuning after quantization

Design rationale for this project
- Small embedding and simple pooling (GAP) make the model parameter-sparse and efficient.
- Sentence-level scoring is a cheap extractive approach that works reasonably well for many datasets and is trivial to run in a streaming or memory-limited setting.
- Representing samples as token sequences of fixed length ensures a constant small input size.
- Full-integer quantization (internals quantized) gives large size and speed wins with small accuracy loss.

Files and purpose (summary)
- src/prep_and_label.py — produce labeled article JSONL (Step 1/2)
- src/build_vectorizer_and_flatten.py — flatten and build TextVectorization, produce .npz tokenized arrays (Step 3)
- src/train_extractive.py — train model and save SavedModel (Step 4)
- src/test_loaded_model.py — quick verification of SavedModel
- src/evaluate_and_infer.py — ROUGE evaluation and summary generation (SavedModel)
- src/convert_tflite.py — TFLite conversion + quantization (Step 5)
- src/simulate_tflite.py — local runtime simulation and timing
- src/tflite_infer_demo.py — final demo (read file → write out JSON summary). Two variants are provided — TF-backed and TF-free (pure Python token mapping). Use TF-free to avoid heavy TF dependency if you only need to run the tiny TFLite model.

Best practices and tips
- Always pin your TensorFlow version in requirements if reproducibility matters (e.g., tensorflow==2.11.0).
- Keep an eye on your Python version; use a supported version for TF. On Windows use the py launcher to create venvs tied to a specific installed interpreter (py -3.10 -m venv venv).
- For TinyML deployments, prefer tflite-runtime on small Linux systems, and TFLite Micro for microcontrollers.
- Use representative datasets that match the runtime input distribution for best quantization results. For token IDs, treat them consistently (if inputs are int32 token IDs in runtime, do not force int8 input quantization unless you implement quantization at runtime).

Examples
- Run end-to-end (after completing Steps 1–5):
```bash
# build vectorizer and tokenize
python src/build_vectorizer_and_flatten.py --data_dir data/cnn_dm --max_tokens 4000 --seq_len 32

# train
python src/train_extractive.py --data_dir data/cnn_dm --models_dir models --epochs 10

# convert to tflite (int8)
python src/convert_tflite.py --saved_model models/extractive_saved_model --vocab data/cnn_dm/vocab.npy --seq_len 32 --rep_samples 500 --out models/extractive_int8.tflite

# simulate tflite
python src/simulate_tflite.py --tflite models/extractive_int8.tflite --vocab data/cnn_dm/vocab.npy --seq_len 32

# run demo on an article file and write summary.json
python src/tflite_infer_demo.py --tflite models/extractive_int8.tflite --vocab data/cnn_dm/vocab.npy --article_file examples/article.txt --out_file out/summary.json --top_k 3
```

Notes on reproducibility
- Random seeds: add reproducible seeds in training if you want deterministic runs:
```python
import random, numpy as np, tensorflow as tf
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)
```
- Save the vocabulary file (vocab.npy) and flatten_stats.json with each experiment.

Next steps / enhancements
- Replace TextVectorization with a deterministic tokenizer shared with the target runtime (e.g., a simple wordpiece or hashing tokenizer) to avoid needing TF on-device.
- Try Quantization-Aware Training (QAT) to reduce quantization-induced accuracy drops.
- Implement redundancy reduction (MMR or similarity thresholding) when selecting top-k sentences to avoid near-duplicate sentences in the summary.
- Integrate with TFLite Micro for MCU deployments:
  - Convert .tflite to a C array and build with TFLite Micro examples.
- Explore knowledge distillation from a larger extractive or abstractive model.

References
- TensorFlow Lite conversion guide: https://www.tensorflow.org/lite/convert
- TFLite int8 quantization: https://www.tensorflow.org/lite/performance/post_training_quantization
- TinyML community: https://www.tinyml.org/
- Rouge score: https://pypi.org/project/rouge-score/

License
- (Add your preferred license here or a note about repository usage.)

Contact / Help
- If you hit issues, include:
  - The command you ran
  - Full traceback
  - python --version and pip list (inside venv)
  - Small sample input (article) causing trouble
I can help debug common issues like Python/TF mismatches, TFLite dtype mismatch errors, and json serialization problems.

Enjoy building tiny extractive summarizers and deploying them to tiny devices!