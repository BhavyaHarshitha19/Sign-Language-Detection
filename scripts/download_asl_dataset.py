"""
ASL Alphabet Dataset Processor (Manual Download Version)
=======================================================
Processes manually downloaded ASL Alphabet Dataset using MediaPipe
to extract hand landmark features for training.

Dataset Location: archive (3)/asl_alphabet_train/asl_alphabet_train/
Expected Structure:
archive (3)/asl_alphabet_train/asl_alphabet_train/A/    ← Images of A gestures
archive (3)/asl_alphabet_train/asl_alphabet_train/B/    ← Images of B gestures
...
archive (3)/asl_alphabet_train/asl_alphabet_train/Z/    ← Images of Z gestures

Usage:
    python scripts/download_asl_dataset.py

Output:
    - data/asl_features.csv: Processed landmark features
    - data/asl_classes.json: Class labels
"""

import os
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Dataset paths - Updated to correct relative path
DATA_DIR = Path("data")
DATASET_DIR = Path("..") / "archive (3)" / "asl_alphabet_train" / "asl_alphabet_train"

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ASL alphabet classes (A-Z) - excluding non-letter classes
ASL_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


def check_dataset_structure():
    """Check if the dataset is properly structured."""
    print("[INFO] Checking dataset structure...")
    
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("\nExpected location: archive (3)/asl_alphabet_train/asl_alphabet_train/")
        print("Make sure you've extracted the dataset to the correct location.")
        return False
    
    missing_classes = []
    total_images = 0
    
    for class_name in ASL_CLASSES:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            missing_classes.append(class_name)
        else:
            # Count images in this class
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            total_images += len(image_files)
            print(f"  Class {class_name}: {len(image_files)} images")
    
    if missing_classes:
        print(f"[ERROR] Missing class folders: {missing_classes}")
        return False
    
    # Check for extra folders (del, nothing, space)
    extra_folders = []
    for item in DATASET_DIR.iterdir():
        if item.is_dir() and item.name not in ASL_CLASSES:
            extra_folders.append(item.name)
    
    if extra_folders:
        print(f"[INFO] Found extra folders (will be ignored): {extra_folders}")
    
    print(f"[SUCCESS] Dataset structure is correct!")
    print(f"          Total images: {total_images}")
    print(f"          Classes: {len(ASL_CLASSES)} (A-Z)")
    return True


def extract_landmarks_from_image(image_path):
    """
    Extract MediaPipe hand landmarks from an image.
    
    Returns:
        np.ndarray: 63-dim feature vector or None if no hand detected
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(rgb_image)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Extract landmarks from first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Convert to numpy array
    landmarks = np.array([
        [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
    ], dtype=np.float32)
    
    # Normalize landmarks
    # 1. Translate wrist (landmark 0) to origin
    landmarks -= landmarks[0]
    
    # 2. Scale by hand size (distance from wrist to middle finger MCP)
    hand_size = np.linalg.norm(landmarks[9] - landmarks[0]) + 1e-6
    landmarks /= hand_size
    
    # Flatten to 63-dimensional vector
    return landmarks.flatten()


def process_dataset():
    """
    Process all images in the dataset and extract landmark features.
    
    Returns:
        tuple: (features, labels) where features is list of 63-dim vectors
    """
    print("[INFO] Processing images and extracting landmarks...")
    
    features = []
    labels = []
    processed_count = 0
    skipped_count = 0
    
    # Process each class folder
    for class_name in ASL_CLASSES:
        class_dir = DATASET_DIR / class_name
        
        # Get all image files
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        print(f"[INFO] Processing class '{class_name}': {len(image_files)} images")
        
        class_features = []
        
        # Process each image with progress bar
        for image_path in tqdm(image_files, desc=f"Class {class_name}"):
            landmarks = extract_landmarks_from_image(image_path)
            
            if landmarks is not None:
                class_features.append(landmarks)
                processed_count += 1
            else:
                skipped_count += 1
        
        # Add to main dataset
        features.extend(class_features)
        labels.extend([class_name] * len(class_features))
        
        print(f"[INFO] Class '{class_name}': {len(class_features)} valid samples")
    
    print(f"\n[SUMMARY]")
    print(f"  Total processed: {processed_count}")
    print(f"  Skipped (no hand): {skipped_count}")
    print(f"  Final dataset size: {len(features)} samples")
    print(f"  Classes: {len(set(labels))}")
    
    return features, labels


def save_dataset(features, labels):
    """Save processed features and labels to CSV and JSON files."""
    if not features:
        print("[ERROR] No features to save")
        return
    
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    
    # Save features to CSV
    csv_path = DATA_DIR / "asl_features.csv"
    
    # Create column names for 63 features
    feature_cols = [f"landmark_{i}" for i in range(63)]
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
    print("ASL Alphabet Dataset Processor")
    print("=" * 60)
    print(f"Dataset location: {DATASET_DIR}")
    print("=" * 60)
    
    # Step 1: Check dataset structure
    if not check_dataset_structure():
        return
    
    # Step 2: Process images and extract features
    features, labels = process_dataset()
    
    if not features:
        print("[ERROR] No valid features extracted. Check dataset structure and images.")
        return
    
    # Step 3: Save processed dataset
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