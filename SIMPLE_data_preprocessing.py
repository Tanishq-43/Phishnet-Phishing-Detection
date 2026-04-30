# SIMPLE_data_preprocessing.py (Use this if above doesn't work)
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class SimpleDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def load_single_dataset(self, file_path='data/phishing.csv'):
        """Load a single dataset and use only numeric features"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
            print(f"📋 Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def preprocess_simple(self, df):
        """Simple preprocessing using only numeric columns"""
        print("🔄 Simple preprocessing...")
        
        # Use only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            print("❌ No numeric columns found!")
            return None, None, None, None, None
        
        print(f"📊 Using numeric columns: {numeric_df.columns.tolist()}")
        
        # Assume last column is target
        target_column = numeric_df.columns[-1]
        X = numeric_df.iloc[:, :-1]  # All columns except last
        y = numeric_df.iloc[:, -1]   # Last column as target
        
        print(f"🎯 Target: {target_column}")
        print(f"📈 Features: {X.shape[1]}")
        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        # Ensure exactly 31 features
        if X.shape[1] > 31:
            X = X.iloc[:, :31]
            print(f"⚠️ Using first 31 features")
        elif X.shape[1] < 31:
            # Pad with zeros
            padding = pd.DataFrame(np.zeros((len(X), 31 - X.shape[1])))
            X = pd.concat([X, padding], axis=1)
            print(f"⚠️ Padded to 31 features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✅ Simple preprocessing completed!")
        return X_train_scaled, X_test_scaled, y_train, y_test, self.scaler

# Use this simple version
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    preprocessor = SimpleDataPreprocessor()
    df = preprocessor.load_single_dataset('data/phishing.csv')
    
    if df is not None:
        X_train, X_test, y_train, y_test, scaler = preprocessor.preprocess_simple(df)
        
        if X_train is not None:
            joblib.dump(scaler, 'models/scaler.pkl')
            print("💾 Scaler saved successfully!")
        else:
            print("❌ Preprocessing failed")
    else:
        print("❌ Could not load dataset")