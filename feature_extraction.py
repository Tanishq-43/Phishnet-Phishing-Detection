# feature_extraction.py - FIXED for real-time prediction
import numpy as np
import joblib
import os
from unified_features import extract_unified_features

class URLFeatureExtractor:
    def __init__(self):
        self.scaler = None
        self.load_scaler()
    
    def load_scaler(self):
        """Load the trained scaler"""
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print("✅ Scaler loaded successfully")
                return True
            except Exception as e:
                print(f"❌ Error loading scaler: {e}")
        else:
            print("⚠️ No scaler found. Run model_training.py first.")
        return False
    
    def extract_features(self, url):
        """
        Extract features from a single URL - SAME as training
        Returns: numpy array of shape (1, 31)
        """
        # Use unified feature extractor
        features = extract_unified_features(url)
        
        # Ensure correct shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        print(f"🔍 Extracted {features.shape[1]} features for: {url[:50]}...")
        
        return features
    
    def preprocess_url(self, url):
        """
        Preprocess URL for prediction (extract + scale)
        """
        # Extract features
        features = self.extract_features(url)
        
        # Apply scaling if available
        if self.scaler is not None:
            features = self.scaler.transform(features)
            print("✅ Features scaled")
        else:
            print("⚠️ Using raw features (no scaling)")
        
        return features
    
    def predict_url(self, url, model):
        """
        Complete prediction pipeline
        """
        # Preprocess
        features = self.preprocess_url(url)
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0]
            confidence = probability[prediction]
        else:
            confidence = 1.0
        
        result = {
            'url': url,
            'is_phishing': bool(prediction == 1),
            'prediction': 'Phishing' if prediction == 1 else 'Safe',
            'confidence': float(confidence),
            'confidence_percent': f"{confidence * 100:.1f}%"
        }
        
        return result


# Quick test
if __name__ == "__main__":
    print("Testing Feature Extractor")
    print("="*40)
    
    extractor = URLFeatureExtractor()
    
    test_urls = [
        "https://www.google.com",
        "http://paypal-verify-login.xyz/secure/account"
    ]
    
    for url in test_urls:
        features = extractor.extract_features(url)
        print(f"\nURL: {url}")
        print(f"Features shape: {features.shape}")
        print(f"Non-zero features: {np.count_nonzero(features[0])}/31")