import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from preprocesing import preprocess_image

TARGET_SIZE = 64

def load_test_data(test_dir="test"):
    images, ids = [], []
    for filename in os.listdir(test_dir):
        if not filename.lower().endswith(".png"):
            continue
        file_id = os.path.splitext(filename)[0]
        filepath = os.path.join(test_dir, filename)
        img = cv2.imread(filepath)
        if img is None:
            continue
        processed = preprocess_image(img, target_size=TARGET_SIZE)
        images.append(processed)
        ids.append(file_id)
    images = np.array(images).reshape(-1, TARGET_SIZE, TARGET_SIZE, 1)
    return images, ids

def predict_and_save(model, test_dir="test", output_file="predictions.csv"):
    images, ids = load_test_data(test_dir)
    predictions = model.predict(images)
    pred_labels = np.argmax(predictions, axis=1)

    with open(output_file, "w") as f:
        f.write("Id,Category\n")
        for file_id, label in zip(ids, pred_labels):
            f.write(f"{file_id},{label}\n")
    print(f"Predictions saved to {output_file}")
    return images, ids, predictions, pred_labels

def visualize_results(model, train_dir="train", test_dir="test"):
    from train import load_train_data

    X_train, y_train = load_train_data(train_dir)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)

    images, ids, predictions, pred_labels = predict_and_save(model, test_dir)

    train_preds = np.argmax(model.predict(X_train), axis=1)
    train_true = np.argmax(y_train, axis=1)

    cm = confusion_matrix(train_true, train_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Train Data)')
    plt.savefig("confusion_matrix.png")
    plt.show()

    per_digit_acc = []
    for d in range(10):
        mask = train_true == d
        if mask.sum() > 0:
            acc = (train_preds[mask] == d).mean()
        else:
            acc = 0
        per_digit_acc.append(acc)

    plt.figure(figsize=(10, 6))
    plt.bar(range(10), per_digit_acc, color='steelblue')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Digit (Train Data)')
    plt.xticks(range(10))
    plt.ylim(0, 1.05)
    plt.savefig("accuracy_per_digit.png")
    plt.show()

    # --- Training vs Test accuracy ---
    # Re-train briefly to get history, or just show bar comparison
    # For simplicity, show train accuracy as bar
    print(f"Train Accuracy: {train_acc:.4f}")
    plt.figure(figsize=(6, 5))
    plt.bar(['Train'], [train_acc], color=['steelblue'])
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.ylim(0, 1.05)
    plt.savefig("train_accuracy.png")
    plt.show()

    # --- ROC Curves (one-vs-rest on train data) ---
    train_probs = model.predict(X_train)
    plt.figure(figsize=(10, 8))
    for d in range(10):
        fpr, tpr, _ = roc_curve((train_true == d).astype(int), train_probs[:, d])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Digit {d} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest, Train Data)')
    plt.legend(loc='lower right')
    plt.savefig("roc_curves.png")
    plt.show()

if __name__ == "__main__":
    model = load_model("digit_cnn.h5")
    visualize_results(model)