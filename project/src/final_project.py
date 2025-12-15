import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224
MODEL_PATH = "project/models/ok_nok_model.h5"
TEST_DIR = "project/data/test"


def predict_single_image(img_path, model, class_names, img_size=IMG_SIZE, threshold=0.5):
    """
    Dogs/Cats-style single-image prediction.
    IMPORTANT: Do NOT divide by 255 here because the model contains Rescaling(1/255).
    """
    img = image.load_img(img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img)          # 0..255
    x = np.expand_dims(x, axis=0)        # (1, H, W, 3)

    prob_class1 = model.predict(x, verbose=0)[0][0]  # probability for label=1
    pred_idx = 1 if prob_class1 >= threshold else 0
    pred_label = class_names[pred_idx]

    plt.imshow(image.load_img(img_path))
    plt.axis("off")
    plt.title(f"Pred: {pred_label} (P(class1)={prob_class1:.3f})")
    plt.show()

    return pred_label, prob_class1


def get_random_mixed_test_images(test_dir=TEST_DIR, n=10):
    """
    Returns a random mixed list of image paths from:
      data/test/OK and data/test/NOK
    """
    image_paths = []
    for cls in ["OK", "NOK"]:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                image_paths.append(os.path.join(cls_dir, fname))

    random.shuffle(image_paths)
    return image_paths[:n]


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train first using train_cnn.py."
        )

    model = keras.models.load_model(MODEL_PATH)

    # Determine class_names from folder order in training convention:
    # Keras assigns label 0 to the first folder name alphabetically.
    # If your folders are OK and NOK, alphabetical order is: ['NOK', 'OK']
    # We'll read from test directory to match:
    tmp = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        label_mode="binary",
        shuffle=False
    )
    class_names = tmp.class_names
    print("Class names:", class_names)

    # Random mixed test demo
    paths = get_random_mixed_test_images(n=10)
    for p in paths:
        print(f"\nTesting: {p}")
        predict_single_image(p, model, class_names)


if __name__ == "__main__":
    main()

