"""
Real ASL Alphabet Recognizer - Landmark-Based
==============================================
Uses MediaPipe 21-point hand landmarks (63 features) for recognition.
Landmarks are background-independent, so they work in any environment.
"""

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

MODEL_DIR = Path("models")


class RealASLRecognizer:
    """ASL recognizer using MediaPipe hand landmarks."""

    def __init__(self, min_detection_confidence: float = 0.7):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.6,
        )

        self.model = None
        self.label_encoder = None
        self.model_loaded = False
        self.feature_type = "landmarks"  # default
        self._load_model()

        # Stability: require N consistent frames before emitting a letter
        self._recent: list = []
        self._stability = 3

    # ── Model Loading ─────────────────────────────────────────────────────

    def _load_model(self):
        model_path = MODEL_DIR / "asl_classifier.pkl"
        encoder_path = MODEL_DIR / "label_encoder.pkl"
        meta_path = MODEL_DIR / "training_metadata.json"

        if not model_path.exists() or not encoder_path.exists():
            print("[WARNING] Trained model not found.")
            print("          Run: python scripts/extract_landmark_features.py")
            print("          Then: python scripts/train_landmark_model.py")
            return

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)

            # Read metadata to know which feature type was used
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self.feature_type = meta.get("feature_type", "landmarks")
                n_feat = meta.get("n_features", "?")
                print(f"[INFO] Model loaded | type={self.feature_type} | features={n_feat} | classes={meta.get('n_classes')}")
            else:
                print("[INFO] Model loaded (no metadata found)")

            self.model_loaded = True

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")

    # ── Feature Extraction ────────────────────────────────────────────────

    def _extract_landmark_features(self, hand_landmarks) -> Optional[np.ndarray]:
        """Extract normalized 63-dim landmark vector from MediaPipe result."""
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])

        coords = np.array(coords, dtype=np.float32).reshape(21, 3)

        # Normalize: wrist to origin, scale by middle-finger-mcp distance
        wrist = coords[0].copy()
        coords -= wrist
        hand_size = np.linalg.norm(coords[9]) + 1e-6
        coords /= hand_size

        return coords.flatten()  # 63 values

    def _extract_image_features(self, frame, hand_landmarks) -> Optional[np.ndarray]:
        """Fallback: extract 27-dim image features (for old image-trained model)."""
        try:
            h, w, _ = frame.shape
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]

            pad = 40
            x1 = max(0, int(min(xs)) - pad)
            y1 = max(0, int(min(ys)) - pad)
            x2 = min(w, int(max(xs)) + pad)
            y2 = min(h, int(max(ys)) + pad)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            roi = cv2.resize(roi, (64, 64))
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
            moments = cv2.moments(gray)
            hu = cv2.HuMoments(moments).flatten()
            stats = np.array([np.mean(gray), np.std(gray), float(np.min(gray)), float(np.max(gray))])

            features = np.concatenate([hist, hu, stats]).astype(np.float32)
            return features if len(features) == 27 else None

        except Exception as e:
            print(f"[ERROR] Image feature extraction: {e}")
            return None

    # ── Classification ────────────────────────────────────────────────────

    def _classify(self, features: np.ndarray) -> Optional[str]:
        if not self.model_loaded or features is None:
            return None

        try:
            proba = self.model.predict_proba([features])[0]
            confidence = float(np.max(proba))
            pred_idx = int(np.argmax(proba))
            label = self.label_encoder.inverse_transform([pred_idx])[0]

            # Require a meaningful confidence gap over second-best
            sorted_p = np.sort(proba)[::-1]
            gap = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else sorted_p[0]

            print(f"[DEBUG] {label}  conf={confidence:.3f}  gap={gap:.3f}")

            if confidence < 0.15 or gap < 0.05:
                return None

            return label

        except Exception as e:
            print(f"[ERROR] Classification: {e}")
            return None

    def _stable(self, prediction: Optional[str]) -> Optional[str]:
        """Return prediction only when it's consistent for N frames."""
        self._recent.append(prediction)
        if len(self._recent) > self._stability:
            self._recent.pop(0)

        if (
            len(self._recent) == self._stability
            and prediction is not None
            and all(p == prediction for p in self._recent)
        ):
            return prediction
        return None

    # ── Public API ────────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], np.ndarray, bool]:
        """
        Process one BGR frame.
        Returns (letter_or_None, annotated_frame, hand_detected).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        annotated = frame_bgr.copy()

        if not results.multi_hand_landmarks:
            self._recent.clear()
            return None, annotated, False

        # Draw landmarks
        for hl in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated, hl,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

        hand_lm = results.multi_hand_landmarks[0]

        # Choose feature extraction based on model type
        if self.feature_type == "landmarks":
            features = self._extract_landmark_features(hand_lm)
        else:
            features = self._extract_image_features(frame_bgr, hand_lm)

        raw = self._classify(features)
        letter = self._stable(raw)

        # Overlay
        if letter:
            cv2.rectangle(annotated, (10, 50), (280, 95), (0, 200, 0), -1)
            cv2.putText(annotated, f"Letter: {letter}", (15, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        elif raw:
            cv2.rectangle(annotated, (10, 50), (280, 95), (0, 200, 200), -1)
            cv2.putText(annotated, f"Detecting: {raw}", (15, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return letter, annotated, True

    def get_model_info(self) -> dict:
        if not self.model_loaded:
            return {"loaded": False}
        return {
            "loaded": True,
            "feature_type": self.feature_type,
            "n_classes": len(self.label_encoder.classes_),
            "classes": self.label_encoder.classes_.tolist(),
            "model_type": type(self.model).__name__,
        }

    def close(self):
        if self.hands:
            self.hands.close()
