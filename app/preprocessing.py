
# preprocessing.py — SKELETON — fill in from your training pipeline
import numpy as np
import cv2

TARGET_SIZE = (128, 128)   # must match training input shape

def preprocess_for_inference(raw_array: np.ndarray) -> np.ndarray:
    img = cv2.resize(raw_array, TARGET_SIZE)

    # TODO — pick the normalization YOU trained with:
    #   /255.0                                    (your custom CNN)
    #   vgg16.preprocess_input(img * 255.0)        (VGG-based models)
    #   resnet50.preprocess_input(img * 255.0)     (ResNet-based models)

    # TODO — if your chosen model uses 3-channel input, stack to RGB:
    #   img = np.stack([img] * 3, axis=-1)

    img = img.astype(np.float32)
    return np.expand_dims(img, axis=0)   # shape: (1, H, W, C)
