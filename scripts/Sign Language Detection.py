"""
ASL Model Training Script
=========================
Trains a Random Forest classifier on real ASL image dataset features.
Uses MediaPipe landmark features extracted from Kaggle ASL Alphabet Dataset.

Usage:
    python scripts/train_asl_model.py
"""

import pickle
from pathlib import Path
     
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def main():
    # Load processed dataset
    dataset_path = DATA_DIR / "asl_features.csv"
    if not dataset_path.exists():
        print("Processed dataset not found. Run scripts/download_asl_dataset.py first.")
        return
    
    print("[INFO] Loading ASL feature dataset...")
    df = pd.read_csv(dataset_path)
    print(f"       Total samples: {len(df)}")
    print(f"       Classes: {df['label'].nunique()}")
    print(f"       Class distribution:")
    
    # Show class distribution
    class_counts = df['label'].value_counts().sort_index()
    for class_name, count in class_counts.items():
        print(f"         {class_name}: {count} samples")
    
    # Prepare features and labels - Updated to work with simple features
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    if not feature_cols:
        # Fallback: try landmark columns (for MediaPipe features)
        feature_cols = [col for col in df.columns if col.startswith('landmark_')]
    
    if not feature_cols:
        print("[ERROR] No feature columns found in dataset!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values
    
    print(f"       Feature vector size: {X.shape[1]} dimensions")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    print(f"       Training samples: {len(X_train)}")
    print(f"       Test samples: {len(X_test)}")
    
    # Train Random Forest model
    print("\n[INFO] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\n[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n" + "="*60)
    print("ASL MODEL TRAINING RESULTS")
    print("="*60)
    
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    report = classification_report(
        y_test, y_pred, 
        target_names=le.classes_, 
        digits=4
    )
    print(f"\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix Summary:")
    print(f"  Shape: {cm.shape}")
    print(f"  Diagonal sum (correct predictions): {np.trace(cm)}")
    print(f"  Total predictions: {np.sum(cm)}")
    
    # Show sample of confusion matrix
    print(f"\nSample Confusion Matrix (first 10x10):")
    print("Rows=True Labels, Cols=Predicted Labels")
    sample_size = min(10, len(le.classes_))
    sample_cm = cm[:sample_size, :sample_size]
    sample_classes = le.classes_[:sample_size]
    
    # Print header
    print(f"{'':>4}", end="")
    for cls in sample_classes:
        print(f"{cls:>4}", end="")
    print()
    
    # Print matrix rows
    for i, true_cls in enumerate(sample_classes):
        print(f"{true_cls:>4}", end="")
        for j in range(len(sample_classes)):
            print(f"{sample_cm[i,j]:>4}", end="")
        print()
    
    # Feature importance
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    
    print(f"\nTop 10 Most Important Features:")
    for i, feat_idx in enumerate(top_features):
        if len(feature_cols) > feat_idx:
            feature_name = feature_cols[feat_idx]
            importance = feature_importance[feat_idx]
            print(f"  {i+1:2d}. {feature_name}: {importance:.4f}")
    
    # Save model and encoder
    model_path = MODEL_DIR / "asl_classifier.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)
    
    # Save training metadata
    metadata = {
        "accuracy": float(accuracy),
        "n_samples": len(df),
        "n_features": X.shape[1],
        "n_classes": len(le.classes_),
        "classes": le.classes_.tolist(),
        "model_params": model.get_params()
    }
    
    with open(MODEL_DIR / "training_metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2)
    
    # Save evaluation results
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    with open(logs_dir / "training_results.txt", "w") as f:
        f.write(f"ASL Model Training Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(logs_dir / "confusion_matrix.csv")
    
    print(f"\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Model: {model_path}")
    print(f"  Label Encoder: {encoder_path}")
    print(f"  Metadata: {MODEL_DIR}/training_metadata.json")
    print(f"  Results: logs/training_results.txt")
    print(f"  Confusion Matrix: logs/confusion_matrix.csv")


if __name__ == "__main__":
    main()