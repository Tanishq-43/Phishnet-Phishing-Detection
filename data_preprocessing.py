# data_preprocessing.py - FIXED VERSION with proper label handling
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from unified_features import extract_unified_features

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def find_dataset_files(self):
        """Find all CSV files in data directory"""
        if not os.path.exists('data'):
            os.makedirs('data')
            print("📁 Created 'data' folder. Please add your dataset CSV files there.")
            return []
        
        dataset_files = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.csv')]
        return dataset_files
    
    def extract_features_from_url(self, url):
        """Use unified feature extractor"""
        return extract_unified_features(url).tolist()
    
    def load_and_combine_datasets(self):
        """Load and combine all datasets from data folder"""
        dataset_files = self.find_dataset_files()
        
        if not dataset_files:
            print("❌ No CSV files found in 'data' folder!")
            return None
        
        print(f"📂 Found {len(dataset_files)} dataset files: {[os.path.basename(f) for f in dataset_files]}")
        
        all_dataframes = []
        for file_path in dataset_files:
            try:
                print(f"\n🔄 Loading {os.path.basename(file_path)}...")
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {df.columns.tolist()[:5]}...")
                all_dataframes.append(df)
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        if not all_dataframes:
            print("❌ No valid datasets loaded!")
            return None
        
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\n✅ Combined dataset: {len(combined_df)} total samples")
        
        return combined_df
    
    def detect_target_column(self, df):
        """Auto-detect the target/label column"""
        target_candidates = [
            'Result', 'Class', 'Label', 'label', 'target', 'phishing', 'status',
            'type', 'outcome', 'is_phishing', 'URL_Type_obf_Type', 'CLASS_LABEL'
        ]
        
        for col in target_candidates:
            if col in df.columns:
                print(f"✅ Found target column: '{col}'")
                return col
        
        # Look for binary columns
        for col in df.columns:
            if df[col].nunique() == 2:
                unique_vals = df[col].dropna().unique()
                if set(unique_vals).issubset({0, 1, -1, '0', '1', '-1'}):
                    print(f"✅ Using binary column '{col}' as target")
                    return col
        
        print(f"⚠️ Using last column '{df.columns[-1]}' as target")
        return df.columns[-1]
    
    def clean_target_values(self, df, target_col):
        """Clean and filter target values, keep only 0 and 1"""
        print(f"\n🔄 Cleaning target column: '{target_col}'")
        
        # Get target series
        y = df[target_col].copy()
        
        # Convert to string for processing
        y_str = y.astype(str).str.lower().str.strip()
        
        # Define valid mappings
        safe_terms = {'0', 'legitimate', 'benign', 'safe', 'good', 'normal', 'non-phishing', 'nonphishing'}
        phishing_terms = {'1', 'phishing', 'malicious', 'bad', 'fake', 'scam', 'deceptive', 'spam'}
        
        # Map values
        y_mapped = []
        valid_indices = []
        
        for idx, val in enumerate(y_str):
            if val in safe_terms:
                y_mapped.append(0)
                valid_indices.append(idx)
            elif val in phishing_terms:
                y_mapped.append(1)
                valid_indices.append(idx)
            else:
                # Try numeric conversion
                try:
                    num_val = float(val)
                    if num_val == 0:
                        y_mapped.append(0)
                        valid_indices.append(idx)
                    elif num_val == 1:
                        y_mapped.append(1)
                        valid_indices.append(idx)
                    # Skip any other numbers (-1, 2, etc.)
                except:
                    # Skip invalid values
                    pass
        
        print(f"   Original samples: {len(df)}")
        print(f"   Valid samples: {len(valid_indices)}")
        print(f"   Removed invalid: {len(df) - len(valid_indices)}")
        
        if len(valid_indices) == 0:
            print("❌ No valid target values found!")
            return None, None
        
        # Filter dataframe to valid indices only
        df_clean = df.iloc[valid_indices].copy()
        y_clean = np.array(y_mapped)
        
        # Show class distribution
        safe_count = sum(y_clean == 0)
        phishing_count = sum(y_clean == 1)
        print(f"\n   Class distribution:")
        print(f"   Safe (0): {safe_count} samples")
        print(f"   Phishing (1): {phishing_count} samples")
        
        return df_clean, y_clean
    
    def find_url_column(self, df):
        """Find the URL column in the dataframe"""
        url_keywords = ['url', 'website', 'domain', 'link', 'address', 'site']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in url_keywords):
                return col
        
        # Check for columns that contain URL-like strings
        for col in df.select_dtypes(include=['object']).columns:
            sample = str(df[col].iloc[0]) if len(df) > 0 else ""
            if 'http' in sample.lower() or '.com' in sample.lower() or '.org' in sample.lower():
                print(f"   Detected URL column: '{col}' (contains http or .com)")
                return col
        
        return None
    
    def extract_features_from_dataframe(self, df, url_col):
        """Extract features from URLs in dataframe"""
        print(f"\n🔄 Extracting features from URLs in column: '{url_col}'")
        
        features_list = []
        valid_indices = []
        
        for idx, url in enumerate(df[url_col]):
            try:
                if pd.isna(url) or str(url).strip() == '':
                    continue
                features = self.extract_features_from_url(str(url))
                features_list.append(features)
                valid_indices.append(idx)
            except Exception as e:
                continue
        
        if not features_list:
            print("❌ No valid URLs found!")
            return None, None
        
        X = np.array(features_list)
        print(f"   Features extracted: {X.shape}")
        print(f"   Features per URL: {X.shape[1]}")
        
        return X, valid_indices
    
    def preprocess_data(self, df):
        """Main preprocessing pipeline"""
        print("\n" + "="*50)
        print("📊 DATA PREPROCESSING")
        print("="*50)
        
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"After removing duplicates: {len(df)} samples")
        
        # Detect target column
        target_col = self.detect_target_column(df)
        
        # Clean target values and filter dataframe
        df_clean, y = self.clean_target_values(df, target_col)
        
        if df_clean is None or len(df_clean) == 0:
            print("❌ No valid data after target cleaning!")
            return None, None, None, None, None
        
        # Find URL column
        url_col = self.find_url_column(df_clean)
        
        if url_col is None:
            print("❌ Could not find URL column in dataset!")
            print(f"   Available columns: {df_clean.columns.tolist()}")
            return None, None, None, None, None
        
        print(f"\n🔗 Using URL column: '{url_col}'")
        
        # Extract features from URLs
        X, valid_indices = self.extract_features_from_dataframe(df_clean, url_col)
        
        if X is None:
            print("❌ Feature extraction failed!")
            return None, None, None, None, None
        
        # Align labels with valid indices
        y_aligned = y[valid_indices]
        
        print(f"\n✅ Final dataset:")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Safe (0): {sum(y_aligned == 0)}")
        print(f"   Phishing (1): {sum(y_aligned == 1)}")
        
        # Check if we have both classes
        if len(np.unique(y_aligned)) < 2:
            print("❌ Only one class present in the data! Need both safe and phishing URLs.")
            return None, None, None, None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_aligned, test_size=0.2, random_state=42, stratify=y_aligned
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n✅ Preprocessing complete!")
        print(f"   Training: {X_train_scaled.shape[0]} samples")
        print(f"   Testing: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.scaler


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("🚀 Starting Data Preprocessing")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    combined_df = preprocessor.load_and_combine_datasets()
    
    if combined_df is not None and len(combined_df) > 0:
        result = preprocessor.preprocess_data(combined_df)
        
        if result[0] is not None:
            X_train, X_test, y_train, y_test, scaler = result
            joblib.dump(scaler, 'models/scaler.pkl')
            print("\n💾 Scaler saved to 'models/scaler.pkl'")
            
            # Save data info
            info = {
                'n_features': X_train.shape[1],
                'n_train': len(X_train),
                'n_test': len(X_test),
                'safe_count': int(sum(y_train == 0)),
                'phishing_count': int(sum(y_train == 1))
            }
            joblib.dump(info, 'models/data_info.pkl')
            print("💾 Data info saved to 'models/data_info.pkl'")
            print("\n✅ Ready to train model! Run: python model_training.py")
        else:
            print("\n❌ Preprocessing failed!")
    else:
        print("\n❌ Please add dataset CSV files to the 'data' folder and run again.")