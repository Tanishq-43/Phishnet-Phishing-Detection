# model_training.py - FIXED VERSION
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class PhishingDetector:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        self.best_model = None
        self.best_score = 0
        self.best_model_name = ""
        self.results = {}
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and select the best one"""
        print("\n" + "="*50)
        print("🤖 MODEL TRAINING")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Get probabilities if available
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else accuracy
                
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'predictions': y_pred
                }
                
                print(f"   ✅ Accuracy: {accuracy:.4f}")
                print(f"   📊 AUC: {auc:.4f}")
                print(f"   🎯 F1 Score: {f1:.4f}")
                
                # Select best model by AUC
                if auc > self.best_score:
                    self.best_score = auc
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return self.results
    
    def save_best_model(self, file_path='models/phishing_model.pkl'):
        """Save the best performing model"""
        if self.best_model:
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.best_model, file_path)
            print(f"\n💾 Best model saved: {self.best_model_name}")
            print(f"   Location: {file_path}")
            
            # Save model metadata
            metadata = {
                'model_name': self.best_model_name,
                'accuracy': self.best_score,
                'features_expected': self.best_model.n_features_in_ if hasattr(self.best_model, 'n_features_in_') else 31,
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            joblib.dump(metadata, 'models/model_metadata.pkl')
            print(f"💾 Metadata saved to 'models/model_metadata.pkl'")
        else:
            print("❌ No model to save!")
    
    def evaluate_and_plot(self, model, X_test, y_test, model_name):
        """Generate evaluation metrics and plots"""
        print("\n" + "="*50)
        print(f"📊 EVALUATION: {model_name}")
        print("="*50)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else accuracy
        
        print(f"\n🎯 Accuracy:  {accuracy:.4f}")
        print(f"📏 Precision: {precision:.4f}")
        print(f"📐 Recall:    {recall:.4f}")
        print(f"⚖️  F1 Score:  {f1:.4f}")
        print(f"📊 AUC:       {auc:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Safe (0)', 'Phishing (1)']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📊 Confusion Matrix:")
        print(f"   True Safe:     {cm[0,0]:6d} | False Phishing: {cm[0,1]:6d}")
        print(f"   False Safe:    {cm[1,0]:6d} | True Phishing:  {cm[1,1]:6d}")
        
        # Plot confusion matrix
        os.makedirs('static', exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                   xticklabels=['Safe', 'Phishing'],
                   yticklabels=['Safe', 'Phishing'])
        plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {accuracy:.4f} | AUC: {auc:.4f}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('static/confusion_matrix.png', dpi=150)
        plt.close()
        print("\n📁 Confusion matrix saved to 'static/confusion_matrix.png'")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }


# Main training script
if __name__ == "__main__":
    print("🚀 PHISHING URL DETECTOR - MODEL TRAINING")
    print("="*50)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load and preprocess data
    from data_preprocessing import DataPreprocessor
    
    print("\n📂 Loading datasets from 'data' folder...")
    preprocessor = DataPreprocessor()
    combined_df = preprocessor.load_and_combine_datasets()
    
    if combined_df is None or len(combined_df) == 0:
        print("\n❌ ERROR: No data loaded!")
        print("\nPlease:")
        print("   1. Create a 'data' folder in your project directory")
        print("   2. Download a phishing dataset from Kaggle")
        print("   3. Place the CSV file in the 'data' folder")
        print("   4. Run this script again")
        exit(1)
    
    # Preprocess
    result = preprocessor.preprocess_data(combined_df)
    
    if result[0] is None:
        print("\n❌ ERROR: Preprocessing failed!")
        exit(1)
    
    X_train, X_test, y_train, y_test, scaler = result
    
    # Train models
    detector = PhishingDetector()
    results = detector.train_models(X_train, X_test, y_train, y_test)
    
    # Display comparison
    print("\n" + "="*50)
    print("📊 MODEL COMPARISON")
    print("="*50)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'AUC':<12} {'F1':<12}")
    print("-"*56)
    for name, res in results.items():
        print(f"{name:<20} {res['accuracy']:.4f}       {res['auc']:.4f}       {res.get('f1', 0):.4f}")
    
    # Save best model
    detector.save_best_model()
    
    # Evaluate best model
    print("\n" + "="*50)
    print("🏆 BEST MODEL EVALUATION")
    print("="*50)
    print(f"\n🏆 Best Model: {detector.best_model_name}")
    print(f"📊 Best AUC Score: {detector.best_score:.4f}")
    
    evaluation = detector.evaluate_and_plot(detector.best_model, X_test, y_test, detector.best_model_name)
    
    # Save evaluation results
    joblib.dump(evaluation, 'models/evaluation_results.pkl')
    
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE!")
    print("="*50)
    print("\n📁 Saved files:")
    print("   - models/phishing_model.pkl (best model)")
    print("   - models/scaler.pkl (feature scaler)")
    print("   - models/model_metadata.pkl (model info)")
    print("   - models/evaluation_results.pkl (metrics)")
    print("   - static/confusion_matrix.png (visualization)")