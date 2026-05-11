"""
ISL Gesture Classifier — Training Script
=========================================
Reads data/isl_dataset.csv, trains a Random Forest classifier,
evaluates it, and saves:
  models/isl_classifier.pkl   – trained model
  models/label_encoder.pkl    – LabelEncoder for class names
  logs/training_report.txt    – classification report
  logs/confusion_matrix.csv   – confusion matrix

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --test-size 0.25 --n-estimators 200
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

DATA_FILE  = Path("data/isl_dataset.csv")
MODEL_DIR  = Path("models")
LOG_DIR    = Path("logs")
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size",     type=float, default=0.2)
    parser.add_argument("--n-estimators",  type=int,   default=150)
    parser.add_argument("--random-state",  type=int,   default=42)
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found.\n"
            "Run scripts/collect_data.py first to build your dataset."
        )

    df = pd.read_csv(DATA_FILE)
    print(f"[INFO] Dataset: {len(df)} samples, {df['label'].nunique()} classes")
    print(f"       Classes: {sorted(df['label'].unique())}\n")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values

    # ── Encode labels ─────────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_enc,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"[INFO] Training RandomForest (n_estimators={args.n_estimators})…")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred    = clf.predict(X_test)
    acc       = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")

    print(f"\n[RESULTS]")
    print(f"  Test accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  CV accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=4,
    )
    print(f"\n{report}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    # ── Save artefacts ────────────────────────────────────────────────────
    with open(MODEL_DIR / "isl_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    report_path = LOG_DIR / "training_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Test accuracy : {acc:.4f}\n")
        f.write(f"CV accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        f.write(report)

    cm_path = LOG_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    # Save class list for the recognizer
    with open(MODEL_DIR / "classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f, indent=2)

    print(f"\n[SAVED]")
    print(f"  Model          → {MODEL_DIR}/isl_classifier.pkl")
    print(f"  Label encoder  → {MODEL_DIR}/label_encoder.pkl")
    print(f"  Class list     → {MODEL_DIR}/classes.json")
    print(f"  Report         → {report_path}")
    print(f"  Confusion mat  → {cm_path}")


if __name__ == "__main__":
    main()
