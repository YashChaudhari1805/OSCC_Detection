import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.applications import EfficientNetB3
import os

print("=" * 60)
print("OSCC Model Rebuilder")
print("=" * 60)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

# --- 1. Define the EXACT architecture from your notebook ---
# Image size from Cell 5
img_shape = (224, 224, 3) 
# Class count from Cell 7 (derived from train_gen)
class_count = 2 

OLD_MODEL_PATH = 'my_final_oscc_model.h5'
NEW_MODEL_PATH = 'oscc_model_rebuilt.keras'

try:
    print(f"\nBuilding model architecture based on oscc (1).ipynb...")
    
    # Base model from Cell 7
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        pooling='max'  # <-- This was the critical difference!
    )
    
    # Sequential model from Cell 7
    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(256, kernel_regularizer=regularizers.l2(l2=0.016),
              activity_regularizer=regularizers.l1(l1=0.006),
              bias_regularizer=regularizers.l1(l1=0.006),
              activation='relu'),
        Dropout(rate=0.45, seed=123),
        Dense(class_count, activation='softmax')
    ])
    
    print("✓ Model architecture created.")
    model.summary()

    # --- 2. Load the weights from the old .h5 file ---
    if not os.path.exists(OLD_MODEL_PATH):
        print(f"\n✗ ERROR: Cannot find {OLD_MODEL_PATH}")
        print("Please make sure your original model file is in this folder.")
    else:
        print(f"\nAttempting to load weights from {OLD_MODEL_PATH}...")
        # We load *only* the weights, not the broken structure
        model.load_weights(OLD_MODEL_PATH)
        print("✓ Weights loaded successfully!")

        # --- 3. Compile and save in the new .keras format ---
        print("\nRecompiling model...")
        model.compile(Adamax(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("✓ Model compiled.")

        print(f"\nSaving new, fixed model to {NEW_MODEL_PATH}...")
        model.save(NEW_MODEL_PATH)
        print("✓ Model saved successfully!")

        print("\n" + "=" * 60)
        print(f"SUCCESS! Use '{NEW_MODEL_PATH}' in your Flask app.")
        print("=" * 60)

except Exception as e:
    print(f"\n✗ ✗ ✗ REBUILD FAILED ✗ ✗ ✗")
    print(f"An error occurred: {e}")
    print("Please check that your 'requirements.txt' libraries are installed.")