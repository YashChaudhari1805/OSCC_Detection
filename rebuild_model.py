"""
rebuild_model.py
----------------
Rebuilds the OSCC model from the original .h5 weights file.

Uses native keras 3 (import keras) — NOT tensorflow.keras or tf_keras.
Both of those namespaces break on Windows with tensorflow-intel 2.16.1.

Usage:
    python rebuild_model.py

Requires:
    my_final_oscc_model.h5  (your original trained model — place in project root)

Produces:
    oscc_model_rebuilt.keras  (used by the Flask app)
"""

import os
import sys

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.optimizers import Adamax
from keras.layers import Dense, Dropout, BatchNormalization
from keras import regularizers
from keras.applications import EfficientNetB3

print("=" * 60)
print("  OSCC Model Rebuilder")
print("=" * 60)
print(f"  TensorFlow : {tf.__version__}")
print(f"  Keras      : {keras.__version__}")
print()

OLD_MODEL_PATH = "my_final_oscc_model.h5"
NEW_MODEL_PATH = "oscc_model_rebuilt.keras"

IMG_SHAPE   = (224, 224, 3)
CLASS_COUNT = 2

try:
    print("[1/4] Building model architecture...")
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SHAPE,
        pooling="max",
    )

    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(
            256,
            kernel_regularizer=regularizers.L2(l2=0.016),
            activity_regularizer=regularizers.L1(l1=0.006),
            bias_regularizer=regularizers.L1(l1=0.006),
            activation="relu",
        ),
        Dropout(rate=0.45),
        Dense(CLASS_COUNT, activation="softmax"),
    ])
    print("    OK Architecture created.")

    print("\n[2/4] Checking for original weights file...")
    if not os.path.exists(OLD_MODEL_PATH):
        print(f"    ERROR: '{OLD_MODEL_PATH}' not found.")
        print("    Place your trained .h5 file in the project root and re-run.")
        sys.exit(1)
    print(f"    OK Found '{OLD_MODEL_PATH}'.")

    print("\n[3/4] Loading weights...")
    model.load_weights(OLD_MODEL_PATH)
    print("    OK Weights loaded.")

    print("\n[4/4] Compiling and saving...")
    model.compile(
        Adamax(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.save(NEW_MODEL_PATH)
    print(f"    OK Saved to '{NEW_MODEL_PATH}'.")

    print()
    print("=" * 60)
    print(f"  SUCCESS - use '{NEW_MODEL_PATH}' to run the app.")
    print("=" * 60)

except Exception as e:
    print(f"\n  REBUILD FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
