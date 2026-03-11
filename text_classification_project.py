import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import warnings

# --- Environment & logging ---
# Suppressing oneDNN info and Python deprecation warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import logging

# Resolving tensorflow logging error
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
try:
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

import tensorflow_hub as hub
import tensorflow_datasets as tfds

# 1. Loading Dataset (IMDB Reviews)

(train_data, val_data, test_data), info = tfds.load(
    name="imdb_reviews",
    split=["train[:70%]", "train[70%:]", "test"],
    as_supervised=True,
    with_info=True,
    shuffle_files=True,
)
 
print("Dataset loaded successfully")
print("Train size:", info.splits["train"].num_examples)
print("Test size:", info.splits["test"].num_examples)
 
# 2. Preparing data pipeline (batched, cached, prefetch)

batch_size = 128
AUTOTUNE = tf.data.AUTOTUNE

train_data = (
    train_data.shuffle(10000)
    .batch(batch_size)
    .cache()
    .prefetch(AUTOTUNE)
)

val_data = val_data.batch(batch_size).cache().prefetch(AUTOTUNE)

test_data = test_data.batch(batch_size).cache().prefetch(AUTOTUNE)


# 3. Building the Model with keras
embedding_url = "https://tfhub.dev/google/nnlm-en-dim50/2"

model = tf.keras.Sequential([
    hub.KerasLayer(
        embedding_url,
        trainable=True,
        input_shape=[],
        dtype=tf.string,
        name="embedding_layer"
    ),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
], name="text_classification_model")

# 4. Compiling the Model

# Use explicit loss object and metrics to avoid deprecation issues
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]

model.compile(optimizer="adam", loss=loss, metrics=metrics)
 
# 5. Model Summary (OUTPUT 1)

print("\nMODEL SUMMARY\n")
model.summary()
 
# 6. Training the Model

print("\nSTARTING TRAINING...\n")
 
try:
    history = model.fit(
        train_data,
        epochs=5,
        validation_data=val_data,
        verbose=1,
    )
    print("\nTRAINING COMPLETED\n")
except Exception as e:
    print("Training failed:", str(e))
    raise
 
# 7. Evaluating the Model (OUTPUT 2)

results = model.evaluate(test_data, verbose=2)

# 8. Final Test Results (OUTPUT 3)

print("\nFINAL TEST RESULTS\n")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")
