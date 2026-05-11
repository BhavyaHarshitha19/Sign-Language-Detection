"""
ISL Dataset Collection Tool
============================
Opens your webcam and records MediaPipe hand landmark vectors
for each gesture class. Saves to data/isl_dataset.csv.

Usage:
    python scripts/collect_data.py --label college --samples 200
    python scripts/collect_data.py --label go      --samples 200
    ... repeat for every class in your vocabulary

Controls (while window is open):
    SPACE  – toggle recording on/off
    Q      – quit and save
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

DATA_DIR  = Path("data")
DATA_FILE = DATA_DIR / "isl_dataset.csv"
DATA_DIR.mkdir(exist_ok=True)

LANDMARK_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
HEADER        = LANDMARK_COLS + ["label"]


def normalize_landmarks(hand_landmarks) -> np.ndarray:
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    pts -= pts[0]                                        # wrist to origin
    scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6      # scale by palm size
    pts  /= scale
    return pts.flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",   required=True, help="Gesture class label, e.g. 'college'")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to collect")
    args = parser.parse_args()

    # Write CSV header only if file is new
    write_header = not DATA_FILE.exists()
    csv_file = open(DATA_FILE, "a", newline="")
    writer   = csv.writer(csv_file)
    if write_header:
        writer.writerow(HEADER)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap       = cv2.VideoCapture(0)
    recording = False
    collected = 0

    print(f"\n[INFO] Collecting '{args.label}' — target: {args.samples} samples")
    print("       Press SPACE to start/stop recording, Q to quit.\n")

    while cap.isOpened() and collected < args.samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
            if recording:
                vec = normalize_landmarks(res.multi_hand_landmarks[0])
                writer.writerow(vec.tolist() + [args.label])
                collected += 1

        # HUD
        color = (0, 200, 80) if recording else (60, 60, 200)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"{'● REC' if recording else '  PAUSED'}  [{collected}/{args.samples}]  label={args.label}",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("ISL Data Collection — Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
            print(f"[{'REC' if recording else 'PAUSE'}]")
        elif key == ord("q"):
            break

    cap.release()
    hands.close()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Saved {collected} samples for '{args.label}' → {DATA_FILE}")


if __name__ == "__main__":
    main()
