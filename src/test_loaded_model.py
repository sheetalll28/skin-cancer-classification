import os
import sys
import numpy as np
import tensorflow as tf

MODEL_DIR = 'models/extractive_saved_model'
NPZ_PATH = 'data/cnn_dm/train_sentences.npz'

if not os.path.exists(MODEL_DIR):
    print(f"Missing model directory: {MODEL_DIR}")
    sys.exit(1)
if not os.path.exists(NPZ_PATH):
    print(f"Missing tokenized data file: {NPZ_PATH}")
    sys.exit(1)

try:
    model = tf.keras.models.load_model(MODEL_DIR)
    print("Loaded model successfully.")
except Exception as e:
    print("Error loading model:", str(e))
    raise

try:
    arr = np.load(NPZ_PATH)
    x = arr['x']
    print("Loaded tokenized array x.shape =", x.shape)
except Exception as e:
    print("Error loading npz:", str(e))
    raise

# Run a small batch through the model
try:
    sample = x[:5]
    preds = model.predict(sample)
    print("Sample predictions:\n", preds)
except Exception as e:
    print("Error during prediction:", str(e))
    raise