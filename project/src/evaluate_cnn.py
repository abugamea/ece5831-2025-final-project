import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Try to import sklearn metrics with a helpful message
try:
    from sklearn.metrics import confusion_matrix, classification_report
except Exception as e:
    raise ImportError(
        "scikit-learn is required for confusion matrix and classification report.\n"
        "Install it with: pip install scikit-learn\n"
        "Or conda install scikit-learn\n"
    ) from e


IMG_SIZE = 224
BATCH_SIZE = 16

TEST_DIR = "project/data/test"
MODEL_PATH = "project/models/ok_nok_model.h5"


def load_test_dataset(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    test_raw = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,  # IMPORTANT
    )
    class_names = test_raw.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_raw.prefetch(AUTOTUNE)
    return test_ds, class_names


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return fig


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train first using train_cnn.py."
        )

    print(f"Loading model from {MODEL_PATH} ...")
    model = keras.models.load_model(MODEL_PATH)

    print("Loading test dataset...")
    test_ds, class_names = load_test_dataset()
    print("Class names:", class_names)

    # Collect predictions and ground truth
    y_true = []
    y_prob = []

    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0).reshape(-1)
        y_prob.extend(probs.tolist())
        y_true.extend(y_batch.numpy().reshape(-1).tolist())

    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)

    # labels 0 and 1 correspond to class_names[0], class_names[1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:\n")
    print(
        classification_report(
            y_true, y_pred,
            labels=[0, 1],
            target_names=class_names,
            digits=2
        )
    )

    # Plot confusion matrix
    fig = plot_confusion_matrix(cm, class_names)
    plt.show()


if __name__ == "__main__":
    main()

