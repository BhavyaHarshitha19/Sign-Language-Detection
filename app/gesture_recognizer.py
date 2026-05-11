"""
Gesture recognizer using MediaPipe Hands.
Extracts hand landmarks from frames and maps them to sign tokens.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ---------------------------------------------------------------------------
# Landmark-to-token mapping (extend this dict with your trained labels)
# ---------------------------------------------------------------------------
# Each entry maps a gesture label (from your classifier) to an ASL token.
# For demo purposes we use a tiny static-gesture vocabulary.
GESTURE_LABEL_MAP: dict[str, str] = {
    "thumbs_up":   "GOOD",
    "open_hand":   "HELLO",
    "fist":        "STOP",
    "point_up":    "ONE",
    "victory":     "TWO",
    # Add more as your classifier grows
}


def _landmark_array(hand_landmarks) -> np.ndarray:
    """Flatten 21 hand landmarks (x, y, z) into a 1-D feature vector."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    ).flatten()


def _classify_landmarks(landmarks: np.ndarray) -> Optional[str]:
    """
    Placeholder classifier — replace with a trained model (e.g. sklearn, ONNX).
    Returns a gesture label string or None.
    """
    # TODO: load your model and call model.predict([landmarks])
    # For now we return None (no gesture detected)
    return None


class GestureRecognizer:
    """
    Processes a single image frame and returns a sign token (or None).
    """

    def __init__(self, min_detection_confidence: float = 0.7):
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[str]:
        """
        Args:
            frame_bgr: OpenCV BGR image as a numpy array.

        Returns:
            A sign token string (e.g. "HELLO") or None if nothing detected.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        # Use the first detected hand
        landmarks = _landmark_array(results.multi_hand_landmarks[0])
        label = _classify_landmarks(landmarks)
        return GESTURE_LABEL_MAP.get(label) if label else None

    def process_video_bytes(self, video_bytes: bytes) -> list[str]:
        """
        Decode a video file from raw bytes, sample frames, and return
        a deduplicated list of detected sign tokens in order.
        """
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        tokens: list[str] = []
        last_token: Optional[str] = None

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(fps / 2))  # sample 2 frames per second
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                token = self.process_frame(frame)
                if token and token != last_token:
                    tokens.append(token)
                    last_token = token
            frame_idx += 1

        cap.release()
        os.unlink(tmp_path)
        return tokens

    def close(self):
        self.hands.close()
