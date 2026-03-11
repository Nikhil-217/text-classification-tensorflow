import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Load dataset
train_data, validate_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

# Build model
model = tf.keras.Sequential([
    hub.KerasLayer(
        "https://tfhub.dev/google/nnlm-en-dim50/2",
        trainable=True,
        input_shape=[]
    ),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Show model structure (OUTPUT 1)
model.summary()

print("\nSTARTING TRAINING...\n")

# ✅ FIX IS HERE
history = model.fit(
    train_data.shuffle(10000).batch(100),
    epochs=5,
    validation_data=validate_data.batch(100),  # <-- CORRECT
    verbose=1
)

print("\nTRAINING DONE\n")

# Evaluation (OUTPUT 2)
results = model.evaluate(test_data.batch(100), verbose=2)

# Final metrics (OUTPUT 3)
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.3f}")
