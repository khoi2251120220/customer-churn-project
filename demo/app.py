"""
Ứng dụng Demo Streamlit - Dự đoán Khách hàng Rời bỏ
Thiết kế theo Figma Design System
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Cấu hình trang
st.set_page_config(
    page_title="Dự đoán Khách hàng Rời bỏ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD MODEL ====================
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'churn_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.pkl')
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, 'models', 'label_encoders.pkl')

@st.cache_resource
def load_model():
    """Load trained model with scaler, feature names, and label encoders"""
    model = None
    scaler = None
    feature_names = None
    label_encoders = None
    
    try:
        # Try to load model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        
        # Try to load scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        
        # Try to load feature names
        if os.path.exists(FEATURE_NAMES_PATH):
            feature_names = joblib.load(FEATURE_NAMES_PATH)
        
        # Try to load label encoders
        if os.path.exists(LABEL_ENCODERS_PATH):
            label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        
        if model is not None:
            # Model successfully loaded
            if scaler is not None and feature_names is not None and label_encoders is not None:
                return model, scaler, feature_names, label_encoders, "✅ Model, Scaler, Feature Names, Label Encoders loaded successfully"
            else:
                # Model loaded but missing some components
                return model, scaler, feature_names, label_encoders, "⚠️ Model loaded but some components missing. Using fallback."
        else:
            return None, None, None, None, "⚠️ Model not found. Using rule-based prediction."
    
    except Exception as e:
        return None, None, None, None, f"⚠️ Error loading model: {str(e)}. Using rule-based prediction."

model, scaler, feature_names, label_encoders, model_status = load_model()
use_ml_model = model is not None and scaler is not None and feature_names is not None and label_encoders is not None

# Display model status
if use_ml_model:
    st.sidebar.success(model_status)
else:
    st.sidebar.warning(model_status)

# ==================== PREPROCESSING HELPER ====================
def preprocess_customer_data(customer_data, scaler=None, feature_names=None, label_encoders=None):
    """
    Preprocess customer data giống như notebook
    - Feature engineering
    - Label Encoding (giống notebook)
    - Scaling
    """
    try:
        df = pd.DataFrame([customer_data])
        
        # ===== CONVERT NUMERIC COLUMNS =====
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # ===== IMPUTE MISSING VALUES =====
        # Fill missing numeric values with median or 0
        numeric_cols_to_fill = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols_to_fill:
            if col in df.columns and df[col].isnull().any():
                # Use 0 for missing tenure/charges (conservative approach)
                df[col].fillna(0, inplace=True)
        
        # ===== FEATURE ENGINEERING =====
        # tenure group - convert to string để label encoder có thể xử lý
        if 'tenure' in df.columns:
            tenure_group = pd.cut(df['tenure'], 
                                 bins=[0, 12, 24, 48, 72],
                                 labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
            df['tenure_group'] = tenure_group.astype(str)  # Convert to string
        
        # avg monthly charges
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Binary service features
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies']
        
        for col in service_cols:
            if col in df.columns:
                df[f'{col}_binary'] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # ===== ENCODING - LABEL ENCODER (giống notebook) =====
        df_encoded = df.copy()
        
        # Drop customerID if exists
        if 'customerID' in df_encoded.columns:
            df_encoded = df_encoded.drop('customerID', axis=1)
        
        # Map binary features (Yes/No → 1/0) BEFORE label encoding
        binary_map = {'Yes': 1, 'No': 0}
        binary_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df_encoded.columns:
                if df_encoded[col].dtype == 'object':
                    df_encoded[col] = df_encoded[col].map(binary_map).fillna(df_encoded[col])
                # Nếu đã là numeric (0/1), skip
        
        # Label Encode categorical features (using loaded encoders) - SKIP binary columns
        if label_encoders is not None:
            binary_cols_set = set(binary_cols)
            for col, encoder in label_encoders.items():
                # Skip if it's a binary column (already encoded as 0/1)
                if col not in binary_cols_set and col in df_encoded.columns:
                    try:
                        # Only encode if still object type
                        if df_encoded[col].dtype == 'object':
                            df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
                    except Exception as e:
                        return None, f"Label encoding error for column '{col}': {str(e)}"
        
        # ===== SCALING =====
        if scaler is not None and feature_names is not None:
            try:
                # Check if all required feature names exist in the dataframe
                missing_cols = [col for col in feature_names if col not in df_encoded.columns]
                if missing_cols:
                    return None, f"Missing columns: {missing_cols}"
                
                # Reorder columns to match feature names
                df_encoded = df_encoded[feature_names]
                
                # Select numerical columns for scaling
                numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns
                df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
                
                return df_encoded, None
            except KeyError as e:
                return None, f"Feature mismatch: {str(e)}"
            except Exception as e:
                return None, str(e)
        
        return df_encoded, None
    
    except Exception as e:
        return None, f"Preprocessing error: {str(e)}"

# CSS Custom - Thiết kế theo Figma
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #F8FAFF;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        padding: 2rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.15);
    }
    
    .header-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.95rem;
        margin-top: 0.5rem;
        padding: 0;
    }
    
    /* Form Section */
    .form-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .form-title {
        color: #1F2937;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #EEF2FF;
    }
    
    /* Button Styling */
    .btn-predict {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .btn-predict:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* Result Cards */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    
    .result-card-high {
        border-left-color: #EF4444;
        background: linear-gradient(135deg, #FEF2F2 0%, #FFF5F5 100%);
    }
    
    .result-card-low {
        border-left-color: #10B981;
        background: linear-gradient(135deg, #F0FDF4 0%, #F7FED4 100%);
    }
    
    .result-card-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    .result-card-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
    }
    
    .result-card-label {
        font-size: 0.85rem;
        color: #6B7280;
        margin-top: 0.5rem;
    }
    
    /* Risk Level Badge */
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .risk-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .risk-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* Recommendation Cards */
    .recommendation-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border-left: 4px solid #6366F1;
        border: 1px solid #E5E7EB;
    }
    
    .recommendation-box-icon {
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }
    
    .recommendation-box-title {
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 0.25rem;
    }
    
    .recommendation-box-content {
        color: #6B7280;
        font-size: 0.9rem;
    }
    
    /* Risk Factors Table */
    .risk-table {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    /* Benefits Section */
    .benefits-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .benefit-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .benefit-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .benefit-title {
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 0.5rem;
    }
    
    .benefit-desc {
        color: #6B7280;
        font-size: 0.9rem;
    }
    
    /* CRISP-DM Steps */
    .steps-container {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .step-card {
        background: white;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        text-align: center;
    }
    
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.5rem;
        height: 2.5rem;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        border-radius: 50%;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }
    
    .step-title {
        font-weight: 600;
        color: #1F2937;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    
    .step-desc {
        font-size: 0.8rem;
        color: #6B7280;
    }
    
    /* Input Labels */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 2px solid #EEF2FF;
        margin: 2rem 0;
    }
    
    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FEF08A 100%);
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .warning-title {
        color: #92400E;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .warning-content {
        color: #78350F;
        font-size: 0.9rem;
    }
    
    /* Success Box */
    .success-box {
        background: linear-gradient(135deg, #D1FAE5 0%, #ECFDF5 100%);
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .success-title {
        color: #065F46;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .success-content {
        color: #047857;
        font-size: 0.9rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6B7280;
        font-size: 0.9rem;
        border-top: 2px solid #EEF2FF;
        margin-top: 3rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .benefits-container, .steps-container {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
    <div class="header-container">
        <p class="header-title">📊 Dự đoán Khách hàng Rời bỏ</p>
        <p class="header-subtitle">Sử dụng AI để dự đoán và giữ chân khách hàng có giá trị cao</p>
    </div>
""", unsafe_allow_html=True)

# ==================== TABS ====================
tab1, tab2 = st.tabs(["🔮 Dự đoán Đơn lẻ", "📊 Dự đoán Hàng loạt"])

# ==================== TAB 1: SINGLE PREDICTION ====================
with tab1:
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<p class="form-title">📋 Nhập Thông tin Khách hàng</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Thông tin Cá nhân", divider="blue")
        gender = st.selectbox(
            "Giới tính",
            ["Female", "Male"],
            format_func=lambda x: "👩 Nữ" if x == "Female" else "👨 Nam",
            key="gender_single"
        )
        
        senior_citizen = st.selectbox(
            "Người cao tuổi",
            ["No", "Yes"],
            format_func=lambda x: "Không" if x == "No" else "Có",
            key="senior_single"
        )
        
        partner = st.selectbox(
            "Có người đồng hành",
            ["No", "Yes"],
            format_func=lambda x: "Không" if x == "No" else "Có",
            key="partner_single"
        )
        
        dependents = st.selectbox(
            "Có người phụ thuộc",
            ["No", "Yes"],
            format_func=lambda x: "Không" if x == "No" else "Có",
            key="dependents_single"
        )
        
        st.subheader("🌐 Thông tin Dịch vụ", divider="blue")
        
        phone_service = st.selectbox(
            "Dịch vụ điện thoại",
            ["No", "Yes"],
            format_func=lambda x: "Không" if x == "No" else "Có",
            key="phone_single"
        )
        
        multiple_lines = st.selectbox(
            "Nhiều đường dây",
            ["No", "Yes", "No phone service"],
            format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có dịch vụ"),
            key="lines_single"
        )
        
        internet_service = st.selectbox(
            "Dịch vụ Internet",
            ["No", "DSL", "Fiber optic"],
            format_func=lambda x: "Không" if x == "No" else ("🔌 DSL" if x == "DSL" else "⚡ Cáp quang"),
            key="internet_single"
        )
        
        online_security = st.selectbox(
            "Bảo mật trực tuyến",
            ["No", "Yes", "No internet service"],
            format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có dịch vụ"),
            key="security_single"
        )
        
        online_backup = st.selectbox(
            "Sao lưu trực tuyến",
            ["No", "Yes", "No internet service"],
            format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có dịch vụ"),
            key="backup_single"
        )
        
        device_protection = st.selectbox(
            "Bảo vệ thiết bị",
            ["No", "Yes", "No internet service"],
            format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có dịch vụ"),
            key="device_single"
        )
    
    with col2:
        st.subheader("💳 Thông tin Tài khoản", divider="blue")
        
        tenure = st.slider("Thời gian sử dụng (tháng)", 0, 72, 12, key="tenure_single")
        
        contract = st.selectbox(
            "Loại hợp đồng",
            ["Month-to-month", "One year", "Two year"],
            format_func=lambda x: "📅 Theo tháng" if x == "Month-to-month" else ("📆 1 năm" if x == "One year" else "📅 2 năm"),
            key="contract_single"
        )
        
        payment_method = st.selectbox(
            "Phương thức thanh toán",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            format_func=lambda x: {
                "Electronic check": "📧 Séc điện tử",
                "Mailed check": "✉️ Séc qua thư",
                "Bank transfer (automatic)": "🏦 Chuyển khoản tự động",
                "Credit card (automatic)": "💳 Thẻ tín dụng tự động"
            }[x],
            key="payment_single"
        )
        
        paperless_billing = st.selectbox(
            "Hóa đơn điện tử",
            ["No", "Yes"],
            format_func=lambda x: "Không" if x == "No" else "Có",
            key="paperless_single"
        )
        
        st.subheader("💰 Thông tin Chi phí", divider="blue")
        
        monthly_charges = st.number_input(
            "Phí hàng tháng ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=5.0,
            key="monthly_single"
        )
        
        total_charges = st.number_input(
            "Tổng phí ($)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=50.0,
            key="total_single"
        )
        
        st.subheader("📺 Dịch vụ Giải trí", divider="blue")
        
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
            format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có dịch vụ"),
            key="tv_single"
        )
        
        streaming_movies = st.selectbox(
            "Streaming Phim",
            ["No", "Yes", "No internet service"],
            format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có dịch vụ"),
            key="movies_single"
        )
        
        tech_support = st.selectbox(
            "Hỗ trợ kỹ thuật",
            ["No", "Yes", "No internet service"],
            format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có dịch vụ"),
            key="tech_single"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== PREDICT BUTTON ====================
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])
    with col_btn2:
        predict_button = st.button(
            "🔮 Phân tích & Dự đoán",
            use_container_width=True,
            key="predict_btn"
        )
    
    # ==================== PREDICTION LOGIC ====================
    if predict_button:
        if use_ml_model:
            # ===== USE TRAINED ML MODEL =====
            try:
                customer_data = {
                    'gender': gender,
                    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                    'Partner': partner,
                    'Dependents': dependents,
                    'tenure': tenure,
                    'PhoneService': phone_service,
                    'MultipleLines': multiple_lines,
                    'InternetService': internet_service,
                    'OnlineSecurity': online_security,
                    'OnlineBackup': online_backup,
                    'DeviceProtection': device_protection,
                    'TechSupport': tech_support,
                    'StreamingTV': streaming_tv,
                    'StreamingMovies': streaming_movies,
                    'Contract': contract,
                    'PaperlessBilling': paperless_billing,
                    'PaymentMethod': payment_method,
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': total_charges
                }
                
                # Preprocess customer data
                df_processed, preprocess_error = preprocess_customer_data(
                    customer_data, scaler, feature_names, label_encoders
                )
                
                if preprocess_error:
                    st.error(f"❌ Lỗi tiền xử lý: {preprocess_error}")
                    st.info("Chuyển sang Rule-based Prediction...")
                    use_ml_model = False
                else:
                    # Make prediction
                    prediction = model.predict(df_processed)[0]
                    risk_score = model.predict_proba(df_processed)[0][1]
                    
                    st.success("✅ Dùng Logistic Regression Model từ Notebook")
            
            except Exception as e:
                st.error(f"❌ Lỗi sử dụng model: {str(e)}")
                st.info("Chuyển sang Rule-based Prediction...")
                use_ml_model = False
        
        if not use_ml_model:
            # ===== FALLBACK: USE RULE-BASED PREDICTION =====
            customer_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Calculate risk score (rule-based fallback)
            risk_score = 0.3
            
            if contract == "Month-to-month":
                risk_score += 0.3
            if tenure < 12:
                risk_score += 0.2
            if internet_service == "Fiber optic":
                risk_score += 0.1
            if payment_method == "Electronic check":
                risk_score += 0.15
            if monthly_charges > 80:
                risk_score += 0.1
            if online_security == "No" and internet_service != "No":
                risk_score += 0.05
            
            risk_score = min(risk_score, 0.95)
            prediction = 1 if risk_score > 0.5 else 0
            
            st.warning("⚠️ Dùng Rule-based Prediction (Model chưa được training)")
        
        # ==================== RESULTS SECTION ====================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p style="font-size: 1.5rem; font-weight: 700; color: #1F2937;">📊 Kết quả Dự đoán</p>', unsafe_allow_html=True)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-card result-card-high">
                        <p class="result-card-title">🚨 Dự đoán</p>
                        <p class="result-card-value">CHURN</p>
                        <p class="result-card-label">Khách hàng có nguy cơ rời bỏ</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card result-card-low">
                        <p class="result-card-title">✅ Dự đoán</p>
                        <p class="result-card-value">Ở LẠI</p>
                        <p class="result-card-label">Khách hàng có khả năng ở lại</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col_r2:
            risk_pct = risk_score * 100
            st.markdown(f"""
                <div class="result-card result-card-high" style="border-left-color: #F59E0B;">
                    <p class="result-card-title">📈 Xác suất Churn</p>
                    <p class="result-card-value">{risk_pct:.1f}%</p>
                    <p class="result-card-label">Mức độ rủi ro</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_r3:
            if risk_score >= 0.7:
                risk_level = "🔴 RỦI RO RẤT CAO"
                badge_class = "risk-high"
            elif risk_score >= 0.5:
                risk_level = "🟠 RỦI RO CAO"
                badge_class = "risk-high"
            elif risk_score >= 0.3:
                risk_level = "🟡 RỦI RO TRUNG BÌNH"
                badge_class = "risk-medium"
            else:
                risk_level = "🟢 RỦI RO THẤP"
                badge_class = "risk-low"
            
            st.markdown(f"""
                <div class="result-card">
                    <p class="result-card-title">⚖️ Mức Độ Rủi ro</p>
                    <div class="{badge_class}">{risk_level}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # ==================== RECOMMENDATIONS ====================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p style="font-size: 1.5rem; font-weight: 700; color: #1F2937;">💡 Khuyến nghị Hành động</p>', unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
                <div class="warning-box">
                    <p class="warning-title">⚠️ Chú ý: Khách hàng có nguy cơ cao rời bỏ dịch vụ</p>
                    <p class="warning-content">Cần thực hiện hành động giữ chân ngay lập tức để tránh mất khách hàng</p>
                </div>
            """, unsafe_allow_html=True)
            
            recommendations = []
            
            if contract == "Month-to-month":
                recommendations.append(("📅", "Nâng cấp Hợp đồng", "Khuyến khích chuyển sang hợp đồng 1-2 năm với ưu đãi đặc biệt (giảm 15-20%)"))
            
            if tenure < 12:
                recommendations.append(("🎁", "Chăm sóc Khách hàng Mới", "Tăng cường onboarding, gửi welcome package, chương trình loyalty điểm"))
            
            if internet_service == "Fiber optic":
                recommendations.append(("⚡", "Cải thiện Dịch vụ", "Kiểm tra chất lượng Fiber optic, cân nhắc giảm giá hoặc nâng cấp gói"))
            
            if payment_method == "Electronic check":
                recommendations.append(("💳", "Thay đổi Thanh toán", "Khuyến khích chuyển sang auto-payment (bank transfer/credit card) với ưu đãi"))
            
            if online_security == "No" and internet_service != "No":
                recommendations.append(("🔒", "Thêm Dịch vụ Bổ sung", "Đề xuất gói bảo mật + sao lưu miễn phí 3 tháng"))
            
            if monthly_charges > 80:
                recommendations.append(("💰", "Điều chỉnh Giá cả", "Xem xét giảm giá 10% hoặc nâng cấp gói dịch vụ với giá trị tốt hơn"))
            
            recommendations.append(("📞", "Liên hệ Trực tiếp", "Gọi điện trong 48h để tìm hiểu vấn đề, chứng tỏ quan tâm"))
            
            for icon, title, desc in recommendations:
                st.markdown(f"""
                    <div class="recommendation-box">
                        <div style="display: flex; align-items: flex-start;">
                            <span style="font-size: 1.3rem; margin-right: 1rem;">{icon}</span>
                            <div style="flex: 1;">
                                <p class="recommendation-box-title">{title}</p>
                                <p class="recommendation-box-content">{desc}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
                <div class="success-box">
                    <p class="success-title">✅ Tốt: Khách hàng có khả năng ở lại cao</p>
                    <p class="success-content">Tiếp tục duy trì chất lượng dịch vụ và xây dựng mối quan hệ lâu dài</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div class="recommendation-box">
                        <p class="recommendation-box-title">📧 Duy trì Tương tác</p>
                        <p class="recommendation-box-content">Gửi email cảm ơn định kỳ, khảo sát hài lòng, chia sẻ mẹo sử dụng</p>
                    </div>
                    <div class="recommendation-box">
                        <p class="recommendation-box-title">📈 Cơ hội Bán thêm</p>
                        <p class="recommendation-box-content">Giới thiệu dịch vụ bổ sung phù hợp, gói combo với giá ưu đãi</p>
                    </div>
                    <div class="recommendation-box">
                        <p class="recommendation-box-title">⭐ Chương trình Khách hàng Thân thiết</p>
                        <p class="recommendation-box-content">Thưởng điểm tích lũy, upgrade tới hạng membership cao hơn</p>
                    </div>
                    <div class="recommendation-box">
                        <p class="recommendation-box-title">🎊 Sự kiện & Ưu đãi Đặc biệt</p>
                        <p class="recommendation-box-content">Mời tham gia sự kiện VIP, ưu đãi sinh nhật, mã khuyến mãi riêng</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # ==================== RISK FACTORS ====================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p style="font-size: 1.5rem; font-weight: 700; color: #1F2937;">⚠️ Phân tích Yếu tố Rủi ro</p>', unsafe_allow_html=True)
        
        risk_factors = []
        
        if contract == "Month-to-month":
            risk_factors.append(("📅 Loại hợp đồng", "Theo tháng", "🔴 CAO", 0.30))
        if tenure < 12:
            risk_factors.append(("⏱️ Thời gian sử dụng", f"{tenure} tháng", "🔴 CAO", 0.20))
        if internet_service == "Fiber optic":
            risk_factors.append(("⚡ Dịch vụ Internet", "Cáp quang", "🟠 TRUNG BÌNH", 0.10))
        if payment_method == "Electronic check":
            risk_factors.append(("💳 Phương thức thanh toán", "Séc điện tử", "🟠 TRUNG BÌNH", 0.15))
        if monthly_charges > 80:
            risk_factors.append(("💰 Phí hàng tháng", f"${monthly_charges:.2f}", "🟠 TRUNG BÌNH", 0.10))
        if online_security == "No" and internet_service != "No":
            risk_factors.append(("🔒 Bảo mật", "Không có", "🟡 THẤP", 0.05))
        
        if risk_factors:
            # Convert all values to string to avoid PyArrow serialization errors
            risk_factors_display = [(f[0], str(f[1]), f[2], str(f[3])) for f in risk_factors]
            risk_df = pd.DataFrame(
                risk_factors_display,
                columns=["📊 Yếu Tố", "💾 Giá Trị", "⚠️ Mức Độ", "📈 Tác Động"]
            )
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        else:
            st.info("✅ Không xác định được yếu tố rủi ro đáng kể cho khách hàng này")
        
        # ==================== CUSTOMER SUMMARY ====================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p style="font-size: 1.5rem; font-weight: 700; color: #1F2937;">📋 Thông tin Khách hàng</p>', unsafe_allow_html=True)
        
        with st.expander("👁️ Xem chi tiết thông tin đã nhập"):
            # Convert all values to string to avoid PyArrow serialization errors
            summary_data = [(k, str(v)) for k, v in customer_data.items()]
            summary_df = pd.DataFrame(
                summary_data,
                columns=["Trường Dữ liệu", "Giá trị"]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ==================== TAB 2: BATCH PREDICTION ====================
with tab2:
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<p class="form-title">📤 Tải lên file CSV để dự đoán hàng loạt</p>', unsafe_allow_html=True)
    
    # Show required columns
    st.info("📋 **Yêu cầu:** File CSV phải chứa các cột sau: ")
    st.code("""gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges, TotalCharges""", language="text")
    
    uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'], key="batch_file")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Đã tải thành công {len(df_batch)} khách hàng")
            
            # VALIDATION: Check if CSV has the required columns
            required_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                           'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                           'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                           'MonthlyCharges', 'TotalCharges']
            
            missing_cols = [col for col in required_cols if col not in df_batch.columns]
            
            if missing_cols:
                st.error(f"""
                ❌ **File CSV không phù hợp!**
                
                **Cột bị thiếu:** {', '.join(missing_cols)}
                
                **Lý do:** Mô hình này được huấn luyện trên **Telco Customer Churn Dataset**, không phải các dataset khác (ngân hàng, thương mại điện tử, v.v.)
                
                **Giải pháp:** 
                1. Tải file CSV từ **Telco Customer Churn Dataset** trên Kaggle
                2. Hoặc sử dụng file mẫu bên dưới
                """)
                
                # Provide a template example
                st.markdown("---")
                st.subheader("📥 Tải file mẫu Telco Customer Churn", divider="blue")
                
                template_data = {
                    'gender': ['Female', 'Male'],
                    'SeniorCitizen': [0, 1],
                    'Partner': ['Yes', 'No'],
                    'Dependents': ['No', 'Yes'],
                    'tenure': [12, 24],
                    'PhoneService': ['Yes', 'No'],
                    'MultipleLines': ['Yes', 'No'],
                    'InternetService': ['Fiber optic', 'DSL'],
                    'OnlineSecurity': ['Yes', 'No'],
                    'OnlineBackup': ['Yes', 'No'],
                    'DeviceProtection': ['Yes', 'No'],
                    'TechSupport': ['Yes', 'No'],
                    'StreamingTV': ['Yes', 'No'],
                    'StreamingMovies': ['Yes', 'No'],
                    'Contract': ['Month-to-month', 'One year'],
                    'PaperlessBilling': ['Yes', 'No'],
                    'PaymentMethod': ['Electronic check', 'Mailed check'],
                    'MonthlyCharges': [65.5, 89.0],
                    'TotalCharges': [780.0, 2136.0]
                }
                template_df = pd.DataFrame(template_data)
                
                st.write("**Ví dụ định dạng dữ liệu đúng:**")
                st.dataframe(template_df, use_container_width=True)
                
                csv_template = template_df.to_csv(index=False)
                st.download_button(
                    label="📥 Tải file mẫu (CSV)",
                    data=csv_template,
                    file_name="telco_churn_template.csv",
                    mime="text/csv"
                )
                st.stop()
            
            st.subheader("📋 Xem trước dữ liệu", divider="blue")
            st.dataframe(df_batch.head(10), use_container_width=True)
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("📊 Số hàng", len(df_batch))
            with col_info2:
                st.metric("📈 Số cột", len(df_batch.columns))
            with col_info3:
                st.metric("🔍 Dung lượng", f"{df_batch.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Simulate batch prediction
            if st.button("🔮 Dự đoán Tất cả", use_container_width=True, key="batch_predict"):
                st.markdown('<p style="font-size: 1.5rem; font-weight: 700; color: #1F2937;">📊 Kết quả Dự đoán Hàng loạt</p>', unsafe_allow_html=True)
                
                # Create results dataframe
                results = []
                
                if use_ml_model:
                    # ===== USE TRAINED ML MODEL FOR BATCH PREDICTION =====
                    try:
                        # Debug: Show feature names expected
                        st.info(f"🔍 Expected columns: {feature_names}")
                        st.info(f"📋 Actual CSV columns: {df_batch.columns.tolist()}")
                        
                        # Preprocess batch data with error tracking
                        df_batch_processed_list = []
                        error_rows = []
                        success_rows = []
                        
                        for idx, row in df_batch.iterrows():
                            try:
                                customer_dict = row.to_dict()
                                df_proc, proc_error = preprocess_customer_data(
                                    customer_dict, scaler, feature_names, label_encoders
                                )
                                
                                if proc_error is None and df_proc is not None:
                                    df_batch_processed_list.append(df_proc)
                                    success_rows.append(idx)
                                else:
                                    error_rows.append((idx, proc_error))
                            except Exception as row_error:
                                # Track problematic row with error
                                error_rows.append((idx, str(row_error)))
                        
                        # Show processing summary
                        st.info(f"✅ Processed: {len(success_rows)} rows | ❌ Failed: {len(error_rows)} rows")
                        if error_rows and len(error_rows) <= 5:
                            st.error(f"First error: Row {error_rows[0][0]} - {error_rows[0][1]}")
                        
                        if len(df_batch_processed_list) > 0:
                            df_batch_processed = pd.concat(df_batch_processed_list, ignore_index=True)
                            
                            # Get predictions
                            predictions = model.predict(df_batch_processed)
                            probabilities = model.predict_proba(df_batch_processed)[:, 1]
                            
                            for idx, (pred, proba) in enumerate(zip(predictions, probabilities)):
                                results.append({
                                    'STT': idx + 1,
                                    'Xác suất Churn': f"{proba*100:.1f}%",
                                    'Dự đoán': '🔴 CHURN' if pred == 1 else '✅ Ở LẠI',
                                    'Mức độ': '🔴 CAO' if proba > 0.7 else ('🟠 TRUNG BÌNH' if proba > 0.5 else '🟢 THẤP')
                                })
                            
                            st.success(f"✅ Dùng Logistic Regression Model từ Notebook ({len(results)} khách hàng)")
                        else:
                            st.warning("⚠️ Không thể xử lý dữ liệu batch - tất cả rows có lỗi. Chuyển sang Rule-based")
                            use_ml_model = False
                    
                    except Exception as e:
                        st.error(f"❌ Lỗi xử lý batch: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                        use_ml_model = False
                
                if not use_ml_model or len(results) == 0:
                    # ===== FALLBACK: USE RULE-BASED PREDICTION FOR BATCH =====
                    for idx, row in df_batch.iterrows():
                        # Calculate risk score based on rules
                        risk = 0.3
                        
                        if 'Contract' in row and row['Contract'] == "Month-to-month":
                            risk += 0.3
                        if 'tenure' in row and row['tenure'] < 12:
                            risk += 0.2
                        if 'InternetService' in row and row['InternetService'] == "Fiber optic":
                            risk += 0.1
                        if 'PaymentMethod' in row and row['PaymentMethod'] == "Electronic check":
                            risk += 0.15
                        if 'MonthlyCharges' in row and row['MonthlyCharges'] > 80:
                            risk += 0.1
                        if 'OnlineSecurity' in row and row['OnlineSecurity'] == "No":
                            risk += 0.05
                        
                        risk = min(risk, 0.95)
                        pred = 1 if risk > 0.5 else 0
                        results.append({
                            'ID': idx + 1,
                            'Khách hàng': f"KH_{idx+1:04d}",
                            'Xác suất Churn': f"{risk*100:.1f}%",
                            'Dự đoán': '🔴 CHURN' if pred == 1 else '✅ Ở LẠI',
                            'Mức độ': '🔴 CAO' if risk > 0.7 else ('🟠 TRUNG BÌNH' if risk > 0.5 else '🟢 THẤP')
                        })
                    
                    st.warning("⚠️ Dùng Rule-based Prediction (Model chưa được training)")
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<p style="font-size: 1.2rem; font-weight: 700; color: #1F2937;">📈 Thống kê Kết quả</p>', unsafe_allow_html=True)
                
                churn_count = len([r for r in results if '🔴' in r['Dự đoán']])
                retain_count = len([r for r in results if '✅' in r['Dự đoán']])
                churn_rate = (churn_count / len(results) * 100) if len(results) > 0 else 0
                
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("🔴 Khách hàng Churn", f"{churn_count}")
                with col_stat2:
                    st.metric("✅ Khách hàng Ở lại", f"{retain_count}")
                with col_stat3:
                    st.metric("📊 Tỷ lệ Churn", f"{churn_rate:.1f}%")
                with col_stat4:
                    st.metric("👥 Tổng cộng", f"{len(results)}")
                
                # Download results
                st.markdown("<hr>", unsafe_allow_html=True)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Tải xuống kết quả (CSV)",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Lỗi xử lý file: {str(e)}")
            st.info("Vui lòng kiểm tra lại định dạng file CSV")
    
    else:
        st.info("📁 Vui lòng tải lên file CSV để bắt đầu dự đoán hàng loạt")
        
        # Template example
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("📋 Định dạng File CSV", divider="blue")
        
        template_data = {
            'gender': ['Female', 'Male'],
            'SeniorCitizen': [0, 1],
            'Partner': ['Yes', 'No'],
            'tenure': [12, 24],
            'MonthlyCharges': [65.5, 89.0],
            'Contract': ['Month-to-month', 'One year']
        }
        template_df = pd.DataFrame(template_data)
        
        st.write("**Ví dụ dữ liệu:**")
        st.dataframe(template_df, use_container_width=True)
        
        # Download template
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Tải template (CSV)",
            data=csv_template,
            file_name="template_churn_data.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==================== BENEFITS SECTION ====================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.5rem; font-weight: 700; color: #1F2937; margin-bottom: 1.5rem;">🎯 Lợi ích Sử dụng Hệ thống</p>', unsafe_allow_html=True)

st.markdown("""
    <div class="benefits-container">
        <div class="benefit-card">
            <div class="benefit-icon">🎯</div>
            <p class="benefit-title">Dự đoán Chính xác</p>
            <p class="benefit-desc">Độ chính xác 85%+ giúp xác định đúng khách hàng có nguy cơ cao</p>
        </div>
        <div class="benefit-card">
            <div class="benefit-icon">💰</div>
            <div class="benefit-title">Tiết kiệm Chi phí</div>
            <p class="benefit-desc">Giảm chi phí khách hàng mới, tập trung vào giữ chân khách hàng cũ</p>
        </div>
        <div class="benefit-card">
            <div class="benefit-icon">⚡</div>
            <div class="benefit-title">Hành động Nhanh</div>
            <p class="benefit-desc">Phát hiện sớm trước khi khách hàng quyết định rời bỏ</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==================== CRISP-DM METHODOLOGY ====================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.5rem; font-weight: 700; color: #1F2937; margin-bottom: 1.5rem;">🔬 Quy trình CRISP-DM Áp dụng</p>', unsafe_allow_html=True)

st.markdown("""
    <div class="steps-container">
        <div class="step-card">
            <div class="step-number">1</div>
            <p class="step-title">Business<br>Understanding</p>
            <p class="step-desc">Hiểu vấn đề kinh doanh và KPI</p>
        </div>
        <div class="step-card">
            <div class="step-number">2</div>
            <p class="step-title">Data<br>Understanding</p>
            <p class="step-desc">Khám phá và phân tích dữ liệu</p>
        </div>
        <div class="step-card">
            <div class="step-number">3</div>
            <p class="step-title">Data<br>Preparation</p>
            <p class="step-desc">Xử lý và chuẩn bị dữ liệu</p>
        </div>
        <div class="step-card">
            <div class="step-number">4</div>
            <p class="step-title">Modeling</p>
            <p class="step-desc">Xây dựng mô hình ML</p>
        </div>
        <div class="step-card">
            <div class="step-number">5</div>
            <p class="step-title">Evaluation</p>
            <p class="step-desc">Đánh giá hiệu suất mô hình</p>
        </div>
        <div class="step-card">
            <div class="step-number">6</div>
            <p class="step-title">Deployment</p>
            <p class="step-desc">Triển khai và ứng dụng thực tế</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
    <div class="footer">
        <p><strong>📊 Hệ thống Dự đoán Khách hàng Rời bỏ</strong></p>
        <p>Xây dựng bằng Streamlit | Áp dụng CRISP-DM | Data Mining Capstone Project</p>
        <p style="color: #9CA3AF; margin-top: 1rem;">© 2024 - Tất cả quyền được bảo lưu</p>
    </div>
""", unsafe_allow_html=True)
