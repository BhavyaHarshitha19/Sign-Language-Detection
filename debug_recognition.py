"""
Debug ASL Recognition - Test the real-time recognizer
"""
import cv2
import numpy as np
from app.real_asl_recognizer import RealASLRecognizer

def test_webcam_recognition():
    """Test the real-time ASL recognizer with webcam."""
    print("🔍 Testing Real-Time ASL Recognition")
    print("=" * 50)
    
    recognizer = RealASLRecognizer()
    
    if not recognizer.model_loaded:
        print("❌ Model not loaded!")
        return
    
    print("✅ Model loaded successfully")
    print(f"   Classes: {recognizer.get_model_info()}")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("\n📹 Camera opened successfully")
    print("Instructions:")
    print("  - Sign ASL letters in front of camera")
    print("  - Press SPACE to capture and analyze current frame")
    print("  - Press 'q' to quit")
    print("  - Watch terminal for debug output")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        frame_count += 1
        
        # Process every 10th frame to reduce spam
        if frame_count % 10 == 0:
            letter, annotated, hand_detected = recognizer.process_frame(frame)
            
            # Print detailed debug info
            print(f"\n--- Frame {frame_count} ---")
            print(f"Hand detected: {hand_detected}")
            print(f"Letter recognized: {letter if letter else 'None'}")
            
            # Show frame
            cv2.imshow("ASL Debug", annotated)
        else:
            cv2.imshow("ASL Debug", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Detailed analysis of current frame
            print(f"\n🔍 DETAILED ANALYSIS - Frame {frame_count}")
            print("-" * 40)
            
            letter, annotated, hand_detected = recognizer.process_frame(frame)
            
            print(f"Hand detected: {hand_detected}")
            print(f"Letter: {letter}")
            
            # Try to get more debug info
            if hasattr(recognizer, 'hands'):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = recognizer.hands.process(rgb)
                
                if results.multi_hand_landmarks:
                    print(f"MediaPipe landmarks found: {len(results.multi_hand_landmarks)} hands")
                    
                    # Test feature extraction manually
                    hand_landmarks = results.multi_hand_landmarks[0]
                    features = recognizer._extract_region_features(frame, hand_landmarks)
                    
                    if features is not None:
                        print(f"Features extracted: {len(features)} dimensions")
                        print(f"Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
                        
                        # Test classification
                        prediction = recognizer._classify_gesture(features)
                        print(f"Classification result: {prediction}")
                    else:
                        print("❌ Feature extraction failed")
                else:
                    print("❌ No MediaPipe landmarks found")
    
    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()

if __name__ == "__main__":
    test_webcam_recognition()