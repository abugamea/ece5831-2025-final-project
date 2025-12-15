import os
import tensorflow as tf
from tensorflow import keras

# -----------------------
# Config
# -----------------------
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
SEED = 123

TRAIN_DIR = "project/data/train"
TEST_DIR  = "project/data/test"
MODEL_DIR = "project/models"
MODEL_PATH = os.path.join(MODEL_DIR, "ok_nok_model.h5")


# -----------------------
# Data
# -----------------------
def load_datasets(img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED):
    """
    Loads train and test datasets from directory structure:
      data/train/OK, data/train/NOK
      data/test/OK,  data/test/NOK

    Returns:
      train_ds, val_ds, test_ds, class_names
    """
    train_raw = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
        seed=seed,
    )

    test_raw = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,  # IMPORTANT for correct evaluation alignment
    )

    class_names = train_raw.class_names  # capture BEFORE prefetch
    print("Class names:", class_names)

    # Split train into train/val (e.g., 80/20)
    total_batches = tf.data.experimental.cardinality(train_raw).numpy()
    train_batches = int(0.8 * total_batches)

    train_ds = train_raw.take(train_batches)
    val_ds = train_raw.skip(train_batches)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_raw.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# -----------------------
# Model
# -----------------------
def build_model(img_size=IMG_SIZE):
    """
    Lightweight CNN for OK vs NOK classification.
    Note: Includes Rescaling(1/255) so prediction code must NOT divide by 255 again.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(img_size, img_size, 3)),
            keras.layers.Rescaling(1.0 / 255),

            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),

            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),

            keras.layers.Conv2D(128, 3, activation="relu"),
            keras.layers.MaxPooling2D(),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets()

    print("Building model...")
    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print(f"Saving model to: {MODEL_PATH}")
    model.save(MODEL_PATH)

    # Optional: quick sanity eval on held-out test set
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
