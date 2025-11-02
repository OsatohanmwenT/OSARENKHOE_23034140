"""
Train emotion detection model using the dataset
"""

import os
from datetime import datetime

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Emotion labels (matching your dataset folders)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTIONS_DISPLAY = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Image size for processing
IMG_SIZE = (48, 48)


def load_dataset(dataset_path, max_images_per_class=1000):
    """
    Load images from dataset folder

    Args:
        dataset_path: Path to train or test folder
        max_images_per_class: Maximum images to load per emotion (for faster training)

    Returns:
        X: Array of images
        y: Array of labels
    """
    X = []
    y = []

    print(f"\n📂 Loading dataset from: {dataset_path}")
    print("=" * 60)

    for emotion_idx, emotion in enumerate(EMOTIONS):
        emotion_path = os.path.join(dataset_path, emotion)

        if not os.path.exists(emotion_path):
            print(f"⚠ Warning: {emotion} folder not found")
            continue

        image_files = [
            f
            for f in os.listdir(emotion_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Limit images per class for faster training
        image_files = image_files[:max_images_per_class]

        print(f"Loading {emotion:10s}: {len(image_files):5d} images", end=" ")

        loaded = 0
        for img_file in image_files:
            img_path = os.path.join(emotion_path, img_file)

            try:
                # Read image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                # Resize to standard size
                img = cv2.resize(img, IMG_SIZE)

                # Normalize pixel values
                img = img.astype("float32") / 255.0

                # Flatten for sklearn
                img_flat = img.flatten()

                X.append(img_flat)
                y.append(emotion_idx)
                loaded += 1

            except Exception:
                continue

        print(f"✓ ({loaded} loaded)")

    print("=" * 60)

    return np.array(X), np.array(y)


def train_model(X_train, y_train, X_test, y_test):
    """
    Train Random Forest classifier

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Trained model
    """
    print("\n🤖 Training Random Forest Model...")
    print("=" * 60)

    # Create model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=20,  # Maximum depth of trees
        min_samples_split=5,  # Minimum samples to split node
        min_samples_leaf=2,  # Minimum samples in leaf
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=1,
    )

    # Train model
    print("\nTraining in progress...")
    model.fit(X_train, y_train)

    print("\n✓ Training completed!")

    # Evaluate on training set
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"\n📊 Training Accuracy: {train_accuracy * 100:.2f}%")

    # Evaluate on test set
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"📊 Test Accuracy: {test_accuracy * 100:.2f}%")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, test_pred, target_names=EMOTIONS_DISPLAY))

    return model


def plot_confusion_matrix(y_test, y_pred, save_path="models/confusion_matrix.png"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTIONS_DISPLAY,
        yticklabels=EMOTIONS_DISPLAY,
    )
    plt.title("Confusion Matrix - Emotion Detection")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("🎭 EMOTION DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if datasets exist
    train_path = "datasets/train"
    test_path = "datasets/test"

    if not os.path.exists(train_path):
        print(f"\n❌ Error: Training dataset not found at {train_path}")
        print("Please ensure your dataset is in the 'datasets/train' folder")
        return

    # Load training data
    print("\n1️⃣ LOADING TRAINING DATA")
    X_train, y_train = load_dataset(train_path, max_images_per_class=1000)

    if len(X_train) == 0:
        print("❌ Error: No training images loaded")
        return

    print(f"\n✓ Loaded {len(X_train)} training images")
    print(f"✓ Image shape: {IMG_SIZE}")
    print(f"✓ Feature vector size: {X_train.shape[1]}")

    # Load test data if available
    X_test, y_test = None, None
    if os.path.exists(test_path):
        print("\n2️⃣ LOADING TEST DATA")
        X_test, y_test = load_dataset(test_path, max_images_per_class=200)
        print(f"\n✓ Loaded {len(X_test)} test images")
    else:
        print("\n2️⃣ SPLITTING DATA (80/20)")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        print(f"✓ Training set: {len(X_train)} images")
        print(f"✓ Test set: {len(X_test)} images")

    # Train model
    print("\n3️⃣ TRAINING MODEL")
    model = train_model(X_train, y_train, X_test, y_test)

    # Save model
    print("\n4️⃣ SAVING MODEL")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)
    model_path = "models/emotion_model.pkl"
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")

    # Save confusion matrix
    print("\n5️⃣ GENERATING VISUALIZATIONS")
    print("=" * 60)
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)

    # Display sample predictions
    print("\n6️⃣ SAMPLE PREDICTIONS")
    print("=" * 60)
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)

    for idx in sample_indices:
        true_label = EMOTIONS_DISPLAY[y_test[idx]]
        pred_proba = model.predict_proba([X_test[idx]])[0]
        pred_label = EMOTIONS_DISPLAY[model.predict([X_test[idx]])[0]]
        confidence = pred_proba.max() * 100

        print(f"\nSample {idx}:")
        print(f"  True: {true_label}")
        print(f"  Predicted: {pred_label} ({confidence:.1f}% confidence)")
        print("  Top 3 predictions:")
        top_3 = np.argsort(pred_proba)[-3:][::-1]
        for i in top_3:
            print(f"    - {EMOTIONS_DISPLAY[i]}: {pred_proba[i]*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Model saved to: {model_path}")
    print(f"🎯 Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\n🚀 You can now use the model in your application!")
    print("   Run: python run.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
