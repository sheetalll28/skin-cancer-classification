#!/usr/bin/env python3
"""
Train a tiny extractive sentence scorer using the tokenized sentence arrays
created in Step 3 (train_sentences.npz, validation_sentences.npz).

Saves:
  models/extractive_saved_model  (SavedModel)
Usage:
  python src/train_extractive.py --data_dir data/cnn_dm --models_dir models --epochs 10
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

def load_npz(path):
    d = np.load(path)
    return d['x'], d['y']

def build_model(vocab_size, seq_len, emb_dim=32, dense_units=64, dropout=0.2):
    inp = keras.Input(shape=(seq_len,), dtype='int32', name='input_ids')
    x = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=seq_len, name='embed')(inp)
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(dense_units, activation='relu', name='dense1')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    out = layers.Dense(1, activation='sigmoid', name='score')(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prepare_dataset(x, y, batch_size=256, shuffle=True, buffer_size=10000):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/cnn_dm', help='where tokenized .npz files are')
    parser.add_argument('--models_dir', default='models', help='where to save model')
    parser.add_argument('--vocab_path', default='data/cnn_dm/vocab.npy', help='vocab.npy path')
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--dense_units', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--limit_train', type=int, default=None, help='optional limit on number of training sentences (for quick runs)')
    args = parser.parse_args()

    train_npz = os.path.join(args.data_dir, 'train_sentences.npz')
    val_npz = os.path.join(args.data_dir, 'validation_sentences.npz')
    if not os.path.exists(train_npz):
        raise SystemExit(f'ERROR: missing {train_npz}. Make sure Step 3 produced it.')

    print('Loading tokenized arrays...')
    x_train, y_train = load_npz(train_npz)
    if args.limit_train:
        x_train = x_train[:args.limit_train]
        y_train = y_train[:args.limit_train]
    if os.path.exists(val_npz):
        x_val, y_val = load_npz(val_npz)
    else:
        x_val, y_val = None, None

    print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
    if x_val is not None:
        print('x_val shape:', x_val.shape, 'y_val shape:', y_val.shape)

    # load vocab to determine vocab_size for Embedding
    if os.path.exists(args.vocab_path):
        vocab = np.load(args.vocab_path, allow_pickle=True)
        vocab_size = int(len(vocab))
    else:
        # Fallback: infer max id + 1
        vocab_size = int(x_train.max()) + 1
    print('Using vocab_size:', vocab_size)

    # compute class weights because positives are rare
    classes = np.unique(y_train)
    try:
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    except Exception:
        class_weight = None
    print('Class weights:', class_weight)

    model = build_model(vocab_size=vocab_size, seq_len=args.seq_len,
                        emb_dim=args.emb_dim, dense_units=args.dense_units)
    model.summary()

    train_ds = prepare_dataset(x_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_ds = prepare_dataset(x_val, y_val, batch_size=args.batch_size, shuffle=False) if x_val is not None else None

    os.makedirs(args.models_dir, exist_ok=True)
    ckpt_path = os.path.join(args.models_dir, 'best.keras')
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=False),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]

    print('Starting training...')
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              callbacks=callbacks, class_weight=class_weight)

    # Save final SavedModel
    saved_model_dir = os.path.join(args.models_dir, 'extractive_saved_model')
    print('Saving SavedModel to', saved_model_dir)
    model.save(saved_model_dir, include_optimizer=False)
    print('Saved model.')

if __name__ == '__main__':
    main()