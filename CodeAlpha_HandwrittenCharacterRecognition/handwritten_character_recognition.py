"""
Handwritten Character Recognition using CNN
CodeAlpha Machine Learning Internship - Task 3

Objective:
Identify handwritten digits using image processing and deep learning.

Dataset:
MNIST handwritten digit dataset loaded from TensorFlow/Keras.

Model:
Convolutional Neural Network (CNN)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_data():
    """Load MNIST dataset and preprocess images."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize pixel values from 0-255 to 0-1
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Reshape for CNN: samples, height, width, channels
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # One-hot encode target labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat


def build_cnn_model():
    """Build CNN architecture."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_sample_digits(X_train, y_train):
    """Save sample digit images."""
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(8, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
        plt.title(f"Label: {y_train[i]}")
        plt.axis("off")

    plt.suptitle("Sample Handwritten Digits")
    plt.tight_layout()
    plt.savefig("images/sample_digits.png", dpi=200)
    plt.close()


def save_training_accuracy(history):
    """Save training and validation accuracy graph."""
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], marker="o", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/training_accuracy.png", dpi=200)
    plt.close()


def save_confusion_matrix(y_test, y_pred):
    """Save confusion matrix image."""
    os.makedirs("images", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(10), np.arange(10))
    plt.yticks(np.arange(10), np.arange(10))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=200)
    plt.close()


def save_prediction_samples(X_test, y_test, y_pred):
    """Save sample prediction images."""
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
        plt.title(f"A:{y_test[i]} P:{y_pred[i]}")
        plt.axis("off")

    plt.suptitle("Sample Predictions")
    plt.tight_layout()
    plt.savefig("images/prediction_samples.png", dpi=200)
    plt.close()


def main():
    print("Loading and preprocessing MNIST dataset...")
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = load_and_preprocess_data()

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    save_sample_digits(X_train, y_train)

    print("Building CNN model...")
    model = build_cnn_model()
    model.summary()

    print("Training CNN model...")
    history = model.fit(
        X_train,
        y_train_cat,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print("Test Accuracy:", test_accuracy)

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    save_training_accuracy(history)
    save_confusion_matrix(y_test, y_pred)
    save_prediction_samples(X_test, y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    model.save("models/handwritten_cnn_model.h5")

    print("\nFiles saved successfully.")
    print("Model: models/handwritten_cnn_model.h5")
    print("Images saved in images/ folder.")


if __name__ == "__main__":
    main()
