import numpy as np
import cv2

def preprocess_image(image, target_size=64):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.mean() > 127:
        image = 255 - image

    image = image.astype("float32") / 255.0

    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return image