"""
Simple ASL Dataset Processor (No MediaPipe)
===========================================
Creates a simple feature extractor without MediaPipe to avoid compatibility issues.
Uses basic image features for demonstration.

Usage:
    python scripts/simple_dataset_processor.py
"""

import os
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Dataset paths - Use absolute path to avoid confusion
import os
DATA_DIR = Path("data")
# The dataset is one level up from sign-language-api directory
DATASET_DIR = Path("..") / "archive (3)" / "asl_alphabet_train" / "asl_alphabet_train"

# ASL alphabet classes (A-Z)
ASL_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


def extract_simple_features(image_path):
    """
    Extract simple image features without MediaPipe.
    This is a fallback method for demonstration.
    """
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Resize to standard size
        image = cv2.resize(image, (64, 64))
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract simple features (histogram + moments)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        moments = cv2.moments(gray)
        
        # Combine features
        features = []
        features.extend(hist.flatten())  # 16 histogram features
        
        # Add 7 Hu moments (invariant to scale, rotation, translation)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(hu_moments)
        
        # Add basic statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray)
        ])
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def check_dataset_structure():
    """Check if the dataset is properly structured."""
    print("[INFO] Checking dataset structure...")
    print(f"[DEBUG] Looking for dataset at: {DATASET_DIR.absolute()}")
    
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        print(f"[DEBUG] Absolute path: {DATASET_DIR.absolute()}")
        print(f"[DEBUG] Current working directory: {Path.cwd()}")
        
        # Let's try to find the dataset
        possible_paths = [
            Path("..") / "archive (3)" / "asl_alphabet_train" / "asl_alphabet_train",
            Path("../..") / "archive (3)" / "asl_alphabet_train" / "asl_alphabet_train",
            Path("archive (3)") / "asl_alphabet_train" / "asl_alphabet_train",
        ]
        
        print("[DEBUG] Trying alternative paths:")
        for path in possible_paths:
            print(f"  {path.absolute()}: {'EXISTS' if path.exists() else 'NOT FOUND'}")
        
        return False
    
    missing_classes = []
    total_images = 0
    
    for class_name in ASL_CLASSES:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            missing_classes.append(class_name)
        else:
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            total_images += len(image_files)
            print(f"  Class {class_name}: {len(image_files)} images")
    
    if missing_classes:
        print(f"[ERROR] Missing class folders: {missing_classes}")
        return False
    
    print(f"[SUCCESS] Dataset structure is correct!")
    print(f"          Total images: {total_images}")
    print(f"          Classes: {len(ASL_CLASSES)} (A-Z)")
    return True


def process_dataset():
    """Process all images and extract features."""
    print("[INFO] Processing images and extracting simple features...")
    
    features = []
    labels = []
    processed_count = 0
    skipped_count = 0
    
    for class_name in ASL_CLASSES:
        class_dir = DATASET_DIR / class_name
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        print(f"[INFO] Processing class '{class_name}': {len(image_files)} images")
        
        class_features = []
        
        # Process subset of images for speed (first 100 per class)
        sample_files = image_files[:100] if len(image_files) > 100 else image_files
        
        for image_path in tqdm(sample_files, desc=f"Class {class_name}"):
            feature_vec = extract_simple_features(image_path)
            
            if feature_vec is not None:
                class_features.append(feature_vec)
                processed_count += 1
            else:
                skipped_count += 1
        
        features.extend(class_features)
        labels.extend([class_name] * len(class_features))
        
        print(f"[INFO] Class '{class_name}': {len(class_features)} valid samples")
    
    print(f"\n[SUMMARY]")
    print(f"  Total processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Final dataset size: {len(features)} samples")
    print(f"  Feature vector size: {len(features[0]) if features else 0}")
    
    return features, labels


def save_dataset(features, labels):
    """Save processed features and labels."""
    if not features:
        print("[ERROR] No features to save")
        return
    
    DATA_DIR.mkdir(exist_ok=True)
    
    # Save features to CSV
    csv_path = DATA_DIR / "asl_features.csv"
    
    # Create column names
    n_features = len(features[0])
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    header = feature_cols + ["label"]
    
    print(f"[INFO] Saving features to {csv_path}")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for feature_vec, label in zip(features, labels):
            row = feature_vec.tolist() + [label]
            writer.writerow(row)
    
    # Save class labels
    classes_path = DATA_DIR / "asl_classes.json"
    unique_classes = sorted(list(set(labels)))
    
    with open(classes_path, "w") as f:
        json.dump(unique_classes, f, indent=2)
    
    print(f"[INFO] Saved {len(features)} samples to {csv_path}")
    print(f"[INFO] Saved {len(unique_classes)} classes to {classes_path}")


def main():
    """Main processing pipeline."""
    print("Simple ASL Dataset Processor (No MediaPipe)")
    print("=" * 60)
    print("NOTE: Using basic image features instead of hand landmarks")
    print("=" * 60)
    
    if not check_dataset_structure():
        return
    
    features, labels = process_dataset()
    
    if not features:
        print("[ERROR] No valid features extracted.")
        return
    
    save_dataset(features, labels)
    
    print("\n" + "=" * 60)
    print("DATASET PROCESSING COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("  1. python scripts/train_asl_model.py")
    print("  2. python scripts/evaluate_asl_model.py")
    print("  3. uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()