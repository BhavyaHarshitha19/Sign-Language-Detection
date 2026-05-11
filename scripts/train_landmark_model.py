"""
Train ASL Model on MediaPipe Landmark Features
===============================================
Trains on all extracted landmark features with data augmentation.

Usage:
    python scripts/train_landmark_model.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def augment_landmarks(X: np.ndarray) -> np.ndarray:
    """
    Augment landmark data by adding small random noise and slight rotations.
    This helps the model generalize to different hand sizes and angles.
    """
    augmented = []
    for sample in X:
        # Original
        augmented.append(sample)

        # Add small Gaussian noise (simulates slight hand tremor)
        for _ in range(2):
            noisy = sample + np.random.normal(0, 0.01, sample.shape)
            augmented.append(noisy.astype(np.float32))

        # Mirror horizontally (flip x coordinates of all landmarks)
        # Landmarks are [lm0_x, lm0_y, lm0_z, lm1_x, lm1_y, lm1_z, ...]
        mirrored = sample.copy()
        for i in range(0, len(sample), 3):
            mirrored[i] = -mirrored[i]  # Flip x
        augmented.append(mirrored.astype(np.float32))

    return np.array(augmented, dtype=np.float32)


def main():
    dataset_path = DATA_DIR / "asl_landmark_features.csv"

    if not dataset_path.exists():
        print("[ERROR] Landmark features not found.")
        print("        Run: python scripts/extract_landmark_features.py first")
        return

    print("[INFO] Loading landmark feature dataset...")
    df = pd.read_csv(dataset_path)
    print(f"       Total samples: {len(df)}")
    print(f"       Classes: {df['label'].nunique()}")

    # Show class distribution
    counts = df['label'].value_counts().sort_index()
    print("\n       Class distribution:")
    for cls, cnt in counts.items():
        print(f"         {cls}: {cnt}")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    print(f"\n       Feature dimensions: {X.shape[1]}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split BEFORE augmentation to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print(f"       Train: {len(X_train)}  Test: {len(X_test)}")

    # Augment training data only
    print(f"\n[INFO] Augmenting training data...")
    X_train_aug = augment_landmarks(X_train)
    y_train_aug = np.repeat(y_train, 4)  # 1 original + 2 noisy + 1 mirrored = 4x

    print(f"       Augmented train size: {len(X_train_aug)}")

    # Train Random Forest
    print(f"\n[INFO] Training Random Forest (n=500, all cores)...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train_aug, y_train_aug)

    # Evaluate on original (non-augmented) test set
    print(f"\n[INFO] Evaluating on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Show which classes are struggling
    print("\nPer-class accuracy:")
    for i, cls in enumerate(le.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            cls_acc = accuracy_score(y_test[mask], y_pred[mask])
            status = "✅" if cls_acc >= 0.8 else "⚠️ " if cls_acc >= 0.5 else "❌"
            print(f"  {status} {cls}: {cls_acc*100:.1f}%")

    # Save model
    model_path = MODEL_DIR / "asl_classifier.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    meta = {
        "feature_type": "landmarks",
        "n_features": X.shape[1],
        "n_classes": len(le.classes_),
        "classes": le.classes_.tolist(),
        "accuracy": float(acc),
    }
    with open(MODEL_DIR / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SAVED] Model -> {model_path}")
    print(f"[SAVED] Encoder -> {encoder_path}")
    print(f"\nNext: uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
