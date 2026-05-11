"""
Simple ASL Recognizer (Compatible with Simple Features)
=======================================================
Uses the same simple feature extraction as the training data.
Works with the model trained on basic image features.
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

MODEL_DIR = Path("models")


class SimpleASLRecognizer:
    """ASL recognizer using simple image features (compatible with training)."""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and label encoder."""
        model_path = MODEL_DIR / "asl_classifier.pkl"
        encoder_path = MODEL_DIR / "label_encoder.pkl"
        
        if not model_path.exists() or not encoder_path.exists():
            print("[WARNING] Trained model not found.")
            print("          Run: python scripts/train_asl_model.py")
            return
        
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            
            self.model_loaded = True
            print(f"[INFO] Loaded simple ASL model with {len(self.label_encoder.classes_)} classes")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
    
    def _extract_simple_features(self, frame):
        """Extract the same simple features used in training."""
        try:
            # Resize to standard size (same as training)
            frame_resized = cv2.resize(frame, (64, 64))
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            # Extract simple features (same as training)
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            moments = cv2.moments(gray)
            
            # Combine features
            features = []
            features.extend(hist.flatten())  # 16 histogram features
            
            # Add 7 Hu moments
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
            print(f"[ERROR] Feature extraction failed: {e}")
            return None
    
    def _classify_gesture(self, features):
        """Classify gesture using trained model."""
        if not self.model_loaded or self.model is None:
            return None
        
        try:
            # Predict using trained model
            prediction = self.model.predict([features])[0]
            
            # Get prediction confidence
            probabilities = self.model.predict_proba([features])[0]
            confidence = np.max(probabilities)
            
            # Only return prediction if confidence is reasonable
            if confidence < 0.4:  # Adjust threshold as needed
                return None
            
            # Convert to class name
            class_label = self.label_encoder.inverse_transform([prediction])[0]
            return class_label
            
        except Exception as e:
            print(f"[ERROR] Classification failed: {e}")
            return None
    
    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], np.ndarray, bool]:
        """
        Process frame and return ASL prediction.
        
        Returns:
            (token, annotated_frame, hand_detected)
        """
        # For simple features, we don't need hand detection
        # We'll just process the whole frame
        annotated = frame_bgr.copy()
        
        # Extract features from the frame
        features = self._extract_simple_features(frame_bgr)
        
        if features is None:
            return None, annotated, False
        
        # Classify the gesture
        token = self._classify_gesture(features)
        
        # Draw a simple indicator if we detected something
        hand_detected = token is not None
        if hand_detected:
            # Draw a green rectangle to show detection
            cv2.rectangle(annotated, (10, 10), (100, 50), (0, 255, 0), 2)
            cv2.putText(annotated, f"Detected: {token}", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return token, annotated, hand_detected
    
    def get_model_info(self) -> dict:
        """Return model information."""
        if not self.model_loaded:
            return {"loaded": False, "error": "Model not loaded"}
        
        return {
            "loaded": True,
            "n_classes": len(self.label_encoder.classes_),
            "classes": self.label_encoder.classes_.tolist(),
            "model_type": type(self.model).__name__
        }
    
    def close(self):
        """Clean up resources."""
        pass  # No resources to clean up for simple features


# Test function
def test_recognizer():
    """Test the simple recognizer with webcam."""
    recognizer = SimpleASLRecognizer()
    
    if not recognizer.model_loaded:
        print("Model not loaded. Cannot test.")
        return
    
    cap = cv2.VideoCapture(0)
    
    print("Simple ASL Recognizer Test")
    print("Press 'q' to quit, 'space' to print current prediction")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        token, annotated, detected = recognizer.process_frame(frame)
        
        # Display info
        status = f"Detection: {'Yes' if detected else 'No'}"
        prediction = f"Letter: {token if token else 'None'}"
        
        cv2.putText(annotated, status, (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated, prediction, (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        cv2.imshow("Simple ASL Test", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            print(f"Current: {token}")
    
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    test_recognizer()