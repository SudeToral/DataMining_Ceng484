import os
import numpy as np

# Suppress TensorFlow info/warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NAME = "ANN"
USES_SCALED = True   # gradient-based → needs StandardScaler

# Reproducibility
tf.random.set_seed(42)


def build(input_dim: int = 10):
    """
    3-layer network matching the paper's description:
      input → Dense(128, ReLU) → Dropout(0.3)
             → Dense(64, ReLU)  → Dropout(0.2)
             → Dense(1, sigmoid)

    Binary cross-entropy loss, Adam optimiser.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fit_predict(X_train, y_train, X_test):
    """
    Train the Keras model and return predictions (0/1 array).
    class_weight corrects for the 500/268 imbalance.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    total = n_neg + n_pos
    # inversely proportional weights
    class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}
    print(f"  [class_weight] 0 (healthy)={class_weight[0]:.3f}  "
          f"1 (diabetic)={class_weight[1]:.3f}")

    model = build(input_dim=X_train.shape[1])

    model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        class_weight=class_weight,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True
            )
        ],
        verbose=0,
    )

    # run_eagerly avoids repeated tf.function retracing across scenarios
    proba = model.predict(X_test, verbose=0).flatten()
    return (proba >= 0.5).astype(int)


# Suppress TF retracing warnings when called from a loop
tf.get_logger().setLevel("ERROR")
