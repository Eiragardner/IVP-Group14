import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from preprocesing import preprocess_image

TARGET_SIZE = 64

def build_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(TARGET_SIZE, TARGET_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_train_data(train_dir="train"):
    images, labels = [], []
    for digit in range(10):
        folder = os.path.join(train_dir, str(digit))
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
            processed = preprocess_image(img, target_size=TARGET_SIZE)
            images.append(processed)
            labels.append(digit)
    images = np.array(images).reshape(-1, TARGET_SIZE, TARGET_SIZE, 1)
    labels = to_categorical(np.array(labels), num_classes=10)
    return images, labels

def train_cnn(train_dir="train", epochs=20, batch_size=32):
    X_train, y_train = load_train_data(train_dir)
    model = build_cnn()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    model.save("digit_cnn.h5")
    print("Model saved to digit_cnn.h5")
    return model

if __name__ == "__main__":
    train_cnn()