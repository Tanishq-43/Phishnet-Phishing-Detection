# app.py - FIXED VERSION
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.secret_key = 'phishnet_secret_key_2025'

# Global variables
model = None
feature_extractor = None
model_loaded = False
model_metadata = None

def load_model():
    global model, feature_extractor, model_loaded, model_metadata
    
    try:
        # Load the trained model
        if not os.path.exists('models/phishing_model.pkl'):
            print("❌ Model file not found. Please run model_training.py first.")
            print("   Command: python model_training.py")
            return
        
        model = joblib.load('models/phishing_model.pkl')
        print("✅ Machine learning model loaded successfully!")
        
        # Load model metadata if available
        if os.path.exists('models/model_metadata.pkl'):
            model_metadata = joblib.load('models/model_metadata.pkl')
            print(f"   Model: {model_metadata.get('model_name', 'Unknown')}")
            print(f"   Accuracy: {model_metadata.get('accuracy', 0):.4f}")
        
        # Import and initialize feature extractor
        from feature_extraction import URLFeatureExtractor
        feature_extractor = URLFeatureExtractor()
        
        # Verify feature dimensions match
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            print(f"   Expected features: {expected_features}")
            
            # Test with a sample URL
            test_features = feature_extractor.extract_features("https://www.google.com")
            actual_features = test_features.shape[1]
            
            if expected_features != actual_features:
                print(f"⚠️ WARNING: Feature mismatch! Model expects {expected_features}, but extractor produces {actual_features}")
                print("   Please retrain the model with the unified feature extractor.")
            else:
                print(f"✅ Feature dimensions match: {actual_features} features")
        
        model_loaded = True
        print("🎯 PhishNet is ready to detect phishing URLs!")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        model_loaded = False

# Load model when app starts
load_model()

@app.route('/')
def home():
    return render_template('index.html', model_loaded=model_loaded, model_info=model_metadata)

@app.route('/detect', methods=['POST'])
def detect_phishing():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    url = request.form.get('url', '').strip()
    if not url:
        return jsonify({'error': 'Please enter a URL'})
    
    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    try:
        print(f"\n🔍 Analyzing URL: {url}")
        
        # Extract and preprocess features
        features = feature_extractor.preprocess_url(url)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0]
            confidence = float(max(probability))
            safe_prob = float(probability[0])
            phishing_prob = float(probability[1])
        else:
            confidence = 1.0
            safe_prob = 1.0 if prediction == 0 else 0.0
            phishing_prob = 1.0 if prediction == 1 else 0.0
        
        result = {
            'url': url,
            'is_phishing': bool(prediction == 1),
            'confidence': confidence,
            'safe_probability': safe_prob,
            'phishing_probability': phishing_prob,
            'status': 'phishing' if prediction == 1 else 'safe',
            'message': '⚠️ PHISHING DETECTED! This URL appears to be malicious.' if prediction == 1 else '✅ SAFE URL. This appears to be a legitimate website.'
        }
        
        print(f"   Prediction: {'🚨 PHISHING' if result['is_phishing'] else '✅ SAFE'}")
        print(f"   Confidence: {confidence*100:.1f}%")
        print(f"   Safe: {safe_prob*100:.1f}% | Phishing: {phishing_prob*100:.1f}%")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route('/about')
def about():
    return render_template('about.html', model_loaded=model_loaded)

@app.route('/statistics')
def statistics():
    return render_template('statistics.html', model_loaded=model_loaded)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for debugging"""
    return jsonify({
        'status': 'running',
        'model_loaded': model_loaded,
        'model_info': model_metadata if model_metadata else None
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting PhishNet - Phishing URL Detector")
    print("="*50)
    print("📍 Access at: http://localhost:5000")
    print("📍 Local network: http://0.0.0.0:5000")
    print("📍 Press CTRL+C to stop")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)