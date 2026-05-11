"""
Extract MediaPipe Hand Landmark Features from ASL Dataset
=========================================================
Uses MediaPipe to extract 21 hand landmarks (63 features) from training images.
These features are background-independent and work well for real-time recognition.

Usage:
    python scripts/extract_landmark_features.py
"""

import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Paths
DATA_DIR = Path("data")
DATASET_DIR = Path("..") / "archive (3)" / "asl_alphabet_train" / "asl_alphabet_train"
ASL_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# MediaPipe setup
mp_hands = mp.solutions.hands


def extract_landmarks(image_path, hands_detector):
    """Extract normalized hand landmarks from an image."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    # Convert to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    # Get first hand landmarks
    hand_lm = results.multi_hand_landmarks[0]

    # Extract x, y, z for all 21 landmarks = 63 values
    coords = []
    for lm in hand_lm.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    coords = np.array(coords, dtype=np.float32).reshape(21, 3)

    # Normalize: move wrist to origin, scale by hand size
    wrist = coords[0].copy()
    coords -= wrist
    hand_size = np.linalg.norm(coords[9]) + 1e-6
    coords /= hand_size

    return coords.flatten()  # 63 features


def main():
    print("MediaPipe Landmark Feature Extractor")
    print("=" * 50)

    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset not found at: {DATASET_DIR.absolute()}")
        sys.exit(1)

    print(f"[INFO] Dataset: {DATASET_DIR.absolute()}")

    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / "asl_landmark_features.csv"

    # Feature column names: lm0_x, lm0_y, lm0_z, lm1_x, ...
    feature_cols = []
    for i in range(21):
        feature_cols += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]

    total_saved = 0
    total_skipped = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.1
    ) as hands:

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(feature_cols + ["label"])

            for class_name in ASL_CLASSES:
                class_dir = DATASET_DIR / class_name
                images = sorted(class_dir.glob("*.jpg"))  # ALL images (3000 per class)

                class_saved = 0
                for img_path in tqdm(images, desc=f"Class {class_name}"):
                    features = extract_landmarks(img_path, hands)
                    if features is not None:
                        writer.writerow(features.tolist() + [class_name])
                        class_saved += 1
                    else:
                        total_skipped += 1

                total_saved += class_saved
                print(f"  {class_name}: {class_saved} samples saved")

    print(f"\n[DONE] Saved {total_saved} samples to {output_path}")
    print(f"       Skipped {total_skipped} images (no hand detected)")
    print(f"\nNext step: python scripts/train_landmark_model.py")


if __name__ == "__main__":
    main()
