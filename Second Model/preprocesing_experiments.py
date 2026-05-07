import numpy as np
import cv2


#here we try various preprocesing techniques to see if we can improve the performance of our model
def preprocess_image(image, target_size=64):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype("float32") / 255.0

    binary = (image > 0.5).astype("float32")

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    coords = cv2.findNonZero((binary * 255).astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = binary[y:y+h, x:x+w]

        max_dim = max(w, h)
        pad_x = (max_dim - w) // 2
        pad_y = (max_dim - h) // 2
        centered = np.zeros((max_dim, max_dim), dtype="float32")
        centered[pad_y:pad_y+h, pad_x:pad_x+w] = cropped
    else:
        centered = binary

    image = cv2.resize(centered, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return image