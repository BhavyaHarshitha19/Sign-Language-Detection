"""
ASL Gesture Recognizer with Real Trained Model
===============================================
Real-time ASL gesture recognition using MediaPipe + trained Random Forest model.
Uses the model trained on real Kaggle ASL Alphabet Dataset images.
NO placeholder logic - all predictions come from the trained model.
"""

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


class ASLRecognizer:
    """Real-time ASL gesture recognizer using trained model on real dataset."""
    
    def __init__(self, min_detection_confidence: float = 0.7):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # ASL alphabet typically uses one hand
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.6
        )
        
        # Load trained model and label encoder
        self.model = None
        self.label_encoder = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the trained ASL classifier and label encoder."""
        model_path = MODEL_DIR / "asl_classifier.pkl"
        encoder_path = MODEL_DIR / "label_encoder.pkl"
        
        if not model_path.exists() or not encoder_path.exists():
            print("[WARNING] Trained model not found.")
            print("          Run the following commands to train the model:")
            print("          1. python scripts/download_asl_dataset.py")
            print("          2. python scripts/train_asl_model.py")
            print("          Model will return None for all predictions until trained.")
            return
        
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            
            self.model_loaded = True
            print(f"[INFO] Loaded ASL model with {len(self.label_encoder.classes_)} classes")
            print(f"       Classes: {', '.join(self.label_encoder.classes_)}")
            
            # Load metadata if available
            metadata_path = MODEL_DIR / "training_metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                print(f"       Model accuracy: {metadata.get('accuracy', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = None
            self.label_encoder = None
            self.model_loaded = False
    
    def _normalize_landmarks(self, hand_landmarks) -> np.ndarray:
        """
        Convert MediaPipe landmarks to normalized 63-dim feature vector.
        Uses the same normalization as training data.
        """
        # Extract (x, y, z) coordinates for all 21 landmarks
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
        ], dtype=np.float32)
        
        # Normalize: wrist (landmark 0) to origin
        landmarks -= landmarks[0]
        
        # Scale by hand size (distance from wrist to middle finger MCP - landmark 9)
        hand_size = np.linalg.norm(landmarks[9] - landmarks[0]) + 1e-6
        landmarks /= hand_size
        
        # Flatten to 63-dimensional vector (21 landmarks × 3 coordinates)
        return landmarks.flatten()
    
    def _classify_gesture(self, landmarks: np.ndarray) -> Optional[str]:
        """
        Classify gesture using trained model.
        Returns ASL class label or None if model not loaded or prediction fails.
        """
        if not self.model_loaded or self.model is None or self.label_encoder is None:
            return None
        
        try:
            # Reshape for sklearn (expects 2D array)
            features = landmarks.reshape(1, -1)
            
            # Predict using trained Random Forest model
            prediction = self.model.predict(features)[0]
            
            # Get prediction confidence (optional)
            probabilities = self.model.predict_proba(features)[0]
            confidence = np.max(probabilities)
            
            # Only return prediction if confidence is above threshold
            if confidence < 0.3:  # Adjust threshold as needed
                return None
            
            # Convert encoded label back to class name
            class_label = self.label_encoder.inverse_transform([prediction])[0]
            
            return class_label
            
        except Exception as e:
            print(f"[ERROR] Classification failed: {e}")
            return None
    
    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], np.ndarray, bool]:
        """
        Process a single frame and return ASL prediction.
        
        Args:
            frame_bgr: BGR image from webcam
            
        Returns:
            (token, annotated_frame, hand_detected)
            - token: ASL class prediction or None
            - annotated_frame: Frame with hand landmarks drawn
            - hand_detected: True if hand landmarks were found
        """
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        # Copy frame for annotation
        annotated = frame_bgr.copy()
        
        if not results.multi_hand_landmarks:
            return None, annotated, False
        
        # Draw hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
        
        # Use first detected hand for classification
        landmarks = self._normalize_landmarks(results.multi_hand_landmarks[0])
        token = self._classify_gesture(landmarks)
        
        return token, annotated, True
    
    def get_model_info(self) -> dict:
        """Return information about the loaded model."""
        if not self.model_loaded:
            return {"loaded": False, "error": "Model not loaded"}
        
        info = {
            "loaded": True,
            "n_classes": len(self.label_encoder.classes_),
            "classes": self.label_encoder.classes_.tolist(),
            "model_type": type(self.model).__name__
        }
        
        if hasattr(self.model, 'n_estimators'):
            info["n_estimators"] = self.model.n_estimators
        
        return info
    
    def close(self):
        """Clean up resources."""
        if self.hands:
            self.hands.close()


# Test function for debugging
def test_recognizer():
    """Test the recognizer with webcam."""
    recognizer = ASLRecognizer()
    
    if not recognizer.model_loaded:
        print("Model not loaded. Cannot test.")
        return
    
    cap = cv2.VideoCapture(0)
    
    print("ASL Recognizer Test")
    print("Press 'q' to quit, 'space' to print current prediction")
    print("Press 'i' to show model info")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        token, annotated, hand_detected = recognizer.process_frame(frame)
        
        # Display prediction on frame
        status = f"Hand: {'Yes' if hand_detected else 'No'}"
        prediction = f"Prediction: {token if token else 'None'}"
        model_status = f"Model: {'Loaded' if recognizer.model_loaded else 'Not Loaded'}"
        
        cv2.putText(annotated, model_status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, prediction, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("ASL Recognizer Test", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            print(f"Current prediction: {token}")
        elif key == ord('i'):
            info = recognizer.get_model_info()
            print(f"Model info: {info}")
    
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    test_recognizer()