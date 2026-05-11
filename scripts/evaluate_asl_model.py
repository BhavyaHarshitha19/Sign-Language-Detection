"""
ASL Model Evaluation Script
===========================
Loads the trained ASL model and evaluates it on test data.
Prints comprehensive metrics to console only (not displayed in UI).

Usage:
    python scripts/evaluate_asl_model.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")


def print_detailed_confusion_matrix(cm, class_names, top_n=15):
    """Print a detailed confusion matrix with class names."""
    print(f"\nDetailed Confusion Matrix (showing top {top_n}x{top_n}):")
    print("Rows = True Labels, Columns = Predicted Labels")
    print("-" * 80)
    
    # Limit to top_n classes for readability
    n_show = min(top_n, len(class_names))
    
    # Print header
    print(f"{'True\\Pred':>8}", end="")
    for i in range(n_show):
        print(f"{class_names[i]:>6}", end="")
    print()
    
    # Print separator
    print("-" * (8 + 6 * n_show))
    
    # Print matrix rows
    for i in range(n_show):
        print(f"{class_names[i]:>8}", end="")
        for j in range(n_show):
            print(f"{cm[i,j]:>6}", end="")
        print()


def analyze_misclassifications(cm, class_names, top_n=5):
    """Analyze and print top misclassifications."""
    print(f"\nTop {top_n} Misclassification Pairs:")
    print("-" * 50)
    
    # Find off-diagonal elements (misclassifications)
    misclass = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i,j] > 0:
                misclass.append((cm[i,j], class_names[i], class_names[j]))
    
    # Sort by count and show top N
    misclass.sort(reverse=True)
    for count, true_label, pred_label in misclass[:top_n]:
        print(f"  {true_label} → {pred_label}: {count} times")


def main():
    # Load model and encoder
    model_path = MODEL_DIR / "asl_classifier.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    
    if not model_path.exists() or not encoder_path.exists():
        print("Trained model not found. Run scripts/train_asl_model.py first.")
        return
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    
    # Load test data
    dataset_path = DATA_DIR / "asl_features.csv"
    if not dataset_path.exists():
        print("Dataset not found. Run scripts/download_asl_dataset.py first.")
        return
    
    df = pd.read_csv(dataset_path)
    
    # Prepare features and labels
    feature_cols = [col for col in df.columns if col.startswith('landmark_')]
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values
    y_encoded = label_encoder.transform(y)
    
    # Use same split as training for consistent evaluation
    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(label_encoder.classes_))
    )
    
    # Print comprehensive evaluation results
    print("=" * 80)
    print("ASL MODEL EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\nDataset Information:")
    print(f"  Total samples in dataset: {len(df)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Number of classes: {len(label_encoder.classes_)}")
    print(f"  Feature vector size: {X.shape[1]} dimensions")
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    print(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 60)
    
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name:<8} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<8}")
    
    # Macro and weighted averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print("-" * 60)
    print(f"{'Macro Avg':<8} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f}")
    print(f"{'Weighted':<8} {weighted_precision:<10.4f} {weighted_recall:<10.4f} {weighted_f1:<10.4f}")
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix Analysis:")
    print(f"  Matrix shape: {cm.shape}")
    print(f"  Correct predictions (diagonal): {np.trace(cm)}")
    print(f"  Total predictions: {np.sum(cm)}")
    print(f"  Misclassifications: {np.sum(cm) - np.trace(cm)}")
    
    # Show detailed confusion matrix
    print_detailed_confusion_matrix(cm, label_encoder.classes_)
    
    # Analyze common misclassifications
    analyze_misclassifications(cm, label_encoder.classes_)
    
    # Class-wise accuracy
    class_accuracies = []
    print(f"\nPer-Class Accuracy:")
    print("-" * 30)
    for i, class_name in enumerate(label_encoder.classes_):
        if support[i] > 0:
            class_acc = cm[i, i] / support[i]
            class_accuracies.append(class_acc)
            print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.1f}%)")
    
    # Summary statistics
    print(f"\nAccuracy Statistics:")
    print(f"  Best performing class: {class_accuracies[np.argmax(class_accuracies)]:.4f}")
    print(f"  Worst performing class: {class_accuracies[np.argmin(class_accuracies)]:.4f}")
    print(f"  Mean class accuracy: {np.mean(class_accuracies):.4f}")
    print(f"  Std class accuracy: {np.std(class_accuracies):.4f}")
    
    # Model information
    print(f"\nModel Information:")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Number of estimators: {model.n_estimators}")
    print(f"  Max depth: {model.max_depth}")
    
    # Feature importance (top 10)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]
        
        print(f"\nTop 10 Most Important Features:")
        for i, feat_idx in enumerate(top_features):
            landmark_idx = feat_idx // 3
            coord = ['x', 'y', 'z'][feat_idx % 3]
            importance = feature_importance[feat_idx]
            print(f"  {i+1:2d}. Landmark {landmark_idx:2d} ({coord}): {importance:.4f}")
    
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    # Save detailed results
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    with open(logs_dir / "detailed_evaluation.txt", "w") as f:
        f.write("ASL Model Detailed Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4))
        f.write(f"\n\nConfusion Matrix:\n{cm}\n")
    
    print(f"\nDetailed results saved to: logs/detailed_evaluation.txt")


if __name__ == "__main__":
    main()