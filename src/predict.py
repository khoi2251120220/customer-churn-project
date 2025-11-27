"""
Prediction Module
Thực hiện dự đoán churn cho khách hàng mới
"""

import pandas as pd
import numpy as np
import joblib


class ChurnPredictor:
    """Class để dự đoán churn cho khách hàng mới"""
    
    def __init__(self, model_path, preprocessor_path=None):
        """
        Khởi tạo predictor với mô hình đã train
        
        Args:
            model_path: Đường dẫn đến file mô hình (.pkl)
            preprocessor_path: Đường dẫn đến preprocessor (nếu có)
        """
        self.model = joblib.load(model_path)
        self.preprocessor = None
        
        if preprocessor_path:
            self.preprocessor = joblib.load(preprocessor_path)
        
        print(f"Model loaded from: {model_path}")
    
    def preprocess_input(self, data):
        """
        Tiền xử lý dữ liệu đầu vào
        
        Args:
            data: DataFrame hoặc dict chứa thông tin khách hàng
        
        Returns:
            DataFrame đã được tiền xử lý
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Nếu có preprocessor, sử dụng nó
        if self.preprocessor:
            data = self.preprocessor.transform(data)
        
        return data
    
    def predict(self, data):
        """
        Dự đoán churn cho khách hàng
        
        Args:
            data: DataFrame hoặc dict chứa thông tin khách hàng
        
        Returns:
            prediction: 0 (No Churn) hoặc 1 (Churn)
        """
        data_processed = self.preprocess_input(data)
        prediction = self.model.predict(data_processed)
        return prediction[0]
    
    def predict_proba(self, data):
        """
        Dự đoán xác suất churn cho khách hàng
        
        Args:
            data: DataFrame hoặc dict chứa thông tin khách hàng
        
        Returns:
            proba: Xác suất churn (0-1)
        """
        data_processed = self.preprocess_input(data)
        proba = self.model.predict_proba(data_processed)
        return proba[0][1]  # Xác suất class 1 (Churn)
    
    def predict_batch(self, data):
        """
        Dự đoán churn cho nhiều khách hàng
        
        Args:
            data: DataFrame chứa thông tin nhiều khách hàng
        
        Returns:
            predictions: Array của predictions
            probabilities: Array của xác suất
        """
        data_processed = self.preprocess_input(data)
        predictions = self.model.predict(data_processed)
        probabilities = self.model.predict_proba(data_processed)[:, 1]
        
        return predictions, probabilities
    
    def interpret_prediction(self, prediction, proba):
        """
        Giải thích kết quả dự đoán
        
        Args:
            prediction: Kết quả dự đoán (0 hoặc 1)
            proba: Xác suất churn
        
        Returns:
            str: Giải thích kết quả
        """
        if prediction == 1:
            risk_level = "HIGH" if proba >= 0.7 else "MEDIUM"
            message = f"⚠️ CHURN RISK: {risk_level}\n"
            message += f"Xác suất rời bỏ: {proba*100:.1f}%\n"
            message += f"Khuyến nghị: Cần có biện pháp giữ chân khách hàng ngay!"
        else:
            message = f"✅ CHURN RISK: LOW\n"
            message += f"Xác suất rời bỏ: {proba*100:.1f}%\n"
            message += f"Khách hàng có khả năng ở lại cao."
        
        return message


def predict_single_customer(model_path, customer_data):
    """
    Hàm tiện ích để dự đoán cho 1 khách hàng
    
    Args:
        model_path: Đường dẫn đến file mô hình
        customer_data: Dict chứa thông tin khách hàng
    
    Returns:
        prediction, probability, interpretation
    """
    predictor = ChurnPredictor(model_path)
    prediction = predictor.predict(customer_data)
    proba = predictor.predict_proba(customer_data)
    interpretation = predictor.interpret_prediction(prediction, proba)
    
    return prediction, proba, interpretation


if __name__ == "__main__":
    # Example usage
    print("Prediction module loaded successfully!")
    print("\nExample usage:")
    print("""
    predictor = ChurnPredictor('models/churn_model.pkl')
    
    customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'InternetService': 'Fiber optic',
        'Contract': 'Month-to-month',
        'MonthlyCharges': 70.0,
        'TotalCharges': 840.0
        # ... other features
    }
    
    prediction = predictor.predict(customer)
    proba = predictor.predict_proba(customer)
    print(predictor.interpret_prediction(prediction, proba))
    """)
