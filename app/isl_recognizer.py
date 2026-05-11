"""
ISL (Indian Sign Language) Gesture Recognizer
----------------------------------------------
Uses MediaPipe Hands to extract 21-landmark feature vectors per frame,
then classifies them with a pluggable model.

ISL vocabulary stub — replace _classify_landmarks() with your trained model.
The ISL_LABEL_MAP covers common ISL words; extend as needed.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ── ISL token vocabulary ──────────────────────────────────────────────────
# Maps classifier output label → ISL token word
ISL_LABEL_MAP: dict[str, str] = {
    # Greetings
    "namaste":      "HELLO",
    "bye":          "BYE",
    "thank_you":    "THANK YOU",
    # Pronouns
    "i_me":         "I",
    "you":          "YOU",
    "he_she":       "HE",
    "we":           "WE",
    "they":         "THEY",
    # Common verbs
    "go":           "GO",
    "come":         "COME",
    "eat":          "EAT",
    "drink":        "DRINK",
    "see":          "SEE",
    "want":         "WANT",
    "like":         "LIKE",
    "help":         "HELP",
    "study":        "STUDY",
    "work":         "WORK",
    # Places
    "home":         "HOME",
    "school":       "SCHOOL",
    "college":      "COLLEGE",
    "hospital":     "HOSPITAL",
    "market":       "MARKET",
    # Time
    "today":        "TODAY",
    "yesterday":    "YESTERDAY",
    "tomorrow":     "TOMORROW",
    "morning":      "MORNING",
    "night":        "NIGHT",
    # Adjectives / misc
    "good":         "GOOD",
    "bad":          "BAD",
    "happy":        "HAPPY",
    "sad":          "SAD",
    "yes":          "YES",
    "no":           "NO",
    "please":       "PLEASE",
    "sorry":        "SORRY",
    "name":         "NAME",
    "what":         "WHAT",
    "where":        "WHERE",
    "when":         "WHEN",
    "how":          "HOW",
    "number_1":     "ONE",
    "number_2":     "TWO",
    "number_3":     "THREE",
}


def _landmark_array(hand_landmarks) -> np.ndarray:
    """Flatten 21 hand landmarks (x, y, z) → 63-dim feature vector."""
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    # Normalize: translate wrist to origin, scale by hand size
    wrist = pts[0].copy()
    pts  -= wrist
    scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6   # middle-finger MCP
    pts  /= scale
    return pts.flatten()


def _classify_landmarks(landmarks: np.ndarray) -> Optional[str]:
    """
    Plug your trained ISL classifier here.

    Example (sklearn):
        return model.predict([landmarks])[0]

    Example (ONNX):
        outputs = session.run(None, {"input": landmarks.reshape(1,-1)})
        return label_encoder.inverse_transform(outputs[0])[0]

    Returns a key from ISL_LABEL_MAP, or None.
    """
    # ── STUB: returns None until a real model is loaded ──────────────────
    return None


class ISLRecognizer:
    """
    Stateless per-frame ISL gesture recognizer.
    Call process_frame() with a BGR numpy array; get back a token or None.
    Also returns annotated frame with hand skeleton drawn.
    """

    def __init__(self, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.6):
        self.hands = mp_hands.Hands(
            static_image_mode=False,       # video mode for tracking
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(
        self, frame_bgr: np.ndarray
    ) -> tuple[Optional[str], np.ndarray, bool]:
        """
        Args:
            frame_bgr: BGR image from webcam.

        Returns:
            (token, annotated_frame, hand_detected)
            token           – ISL word token or None
            annotated_frame – BGR frame with hand skeleton overlay
            hand_detected   – True if any hand landmarks found
        """
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        annotated = frame_bgr.copy()

        if not results.multi_hand_landmarks:
            return None, annotated, False

        for hand_lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

        # Classify primary hand
        landmarks = _landmark_array(results.multi_hand_landmarks[0])
        label     = _classify_landmarks(landmarks)
        token     = ISL_LABEL_MAP.get(label) if label else None
        return token, annotated, True

    def close(self):
        self.hands.close()
