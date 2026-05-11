"""
Quick test to verify the ASL model is working correctly
"""
import pickle
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")

def test_model():
    """Test if the model can make predictions."""
    print("Testing ASL Model...")
    
    # Load model
    model_path = MODEL_DIR / "asl_classifier.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    
    if not model_path.exists() or not encoder_path.exists():
        print("❌ Model files not found!")
        return
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        
        print("✅ Model loaded successfully")
        print(f"   Classes: {len(label_encoder.classes_)}")
        print(f"   Model type: {type(model).__name__}")
        
        # Test with random features (27 features as expected)
        test_features = np.random.rand(1, 27).astype(np.float32)
        
        # Make prediction
        prediction = model.predict(test_features)[0]
        probabilities = model.predict_proba(test_features)[0]
        confidence = np.max(probabilities)
        
        class_label = label_encoder.inverse_transform([prediction])[0]
        
        print(f"✅ Model prediction test successful")
        print(f"   Random input prediction: {class_label}")
        print(f"   Confidence: {confidence:.3f}")
        
        # Test with actual training data sample
        import pandas as pd
        dataset_path = Path("data") / "asl_features.csv"
        
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
            
            # Get first sample
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            first_sample = df[feature_cols].iloc[0].values.reshape(1, -1)
            true_label = df['label'].iloc[0]
            
            prediction = model.predict(first_sample)[0]
            probabilities = model.predict_proba(first_sample)[0]
            confidence = np.max(probabilities)
            predicted_label = label_encoder.inverse_transform([prediction])[0]
            
            print(f"✅ Training data test:")
            print(f"   True label: {true_label}")
            print(f"   Predicted: {predicted_label}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Match: {'✅' if true_label == predicted_label else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    test_model()