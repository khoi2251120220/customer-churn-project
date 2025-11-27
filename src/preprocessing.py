"""
Data Preprocessing Module
Xử lý dữ liệu: missing values, outliers, feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class ChurnDataPreprocessor:
    """Class xử lý tiền xử lý dữ liệu churn"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """Đọc dữ liệu từ file CSV"""
        df = pd.read_csv(filepath)
        print(f"Loaded data shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """Xử lý missing values"""
        # Kiểm tra missing values
        missing = df.isnull().sum()
        print("\nMissing values:")
        print(missing[missing > 0])
        
        # TotalCharges có thể có giá trị trống (string ' ')
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        return df
    
    def handle_outliers(self, df, columns=None):
        """Phát hiện và xử lý outliers bằng IQR method"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outlier_info = {}
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_info[col] = len(outliers)
            
        print("\nOutliers detected:")
        for col, count in outlier_info.items():
            if count > 0:
                print(f"  {col}: {count} outliers")
        
        return df
    
    def feature_engineering(self, df):
        """Tạo các features mới"""
        # Tính tenure groups
        if 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(df['tenure'], 
                                        bins=[0, 12, 24, 48, 72],
                                        labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
        
        # Tính average monthly charges
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Số lượng dịch vụ sử dụng
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies']
        
        for col in service_cols:
            if col in df.columns:
                df[f'{col}_binary'] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
        
        return df
    
    def encode_features(self, df, target_column='Churn'):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        # Loại bỏ customerID
        if 'customerID' in df_encoded.columns:
            df_encoded = df_encoded.drop('customerID', axis=1)
        
        # Encode target variable
        if target_column in df_encoded.columns:
            df_encoded[target_column] = df_encoded[target_column].map({'Yes': 1, 'No': 0})
        
        # Encode categorical variables (including 'object' and 'category' dtypes)
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != target_column]
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        return df_encoded
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features only"""
        # Identify numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) == 0:
            # No numerical columns to scale
            return X_train.copy(), X_test.copy()
        
        # Scale only numerical features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self, df, target_column='Churn', test_size=0.2, random_state=42):
        """Pipeline hoàn chỉnh để chuẩn bị dữ liệu"""
        # Xử lý missing values
        df = self.handle_missing_values(df)
        
        # Xử lý outliers
        df = self.handle_outliers(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode features
        df_encoded = self.encode_features(df, target_column)
        
        # Split features and target
        X = df_encoded.drop(target_column, axis=1)
        y = df_encoded[target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print(f"\nData prepared:")
        print(f"  X_train shape: {X_train_scaled.shape}")
        print(f"  X_test shape: {X_test_scaled.shape}")
        print(f"  Class distribution: {y_train.value_counts().to_dict()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ChurnDataPreprocessor()
    df = preprocessor.load_data('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    print("\nPreprocessing completed successfully!")
