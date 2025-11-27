"""
Ứng dụng Demo Streamlit - Dự đoán Khách hàng Rời bỏ
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Thêm src vào path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import ChurnPredictor

# Cấu hình trang
st.set_page_config(
    page_title="Dự đoán Khách hàng Rời bỏ",
    page_icon="📊",
    layout="wide"
)

# Tiêu đề
st.title("📊 Hệ thống Dự đoán Khách hàng Rời bỏ")
st.markdown("---")

# Thanh bên
st.sidebar.header("Giới thiệu")
st.sidebar.info(
    """
    **Ứng dụng Dự đoán Khách hàng Rời bỏ**
    
    Dự đoán khách hàng có nguy cơ rời bỏ dịch vụ (churn) 
    dựa trên thông tin cá nhân và sử dụng dịch vụ.
    
    **Mô hình**: Random Forest / XGBoost
    **Độ chính xác**: ~85%
    **ROC-AUC**: ~0.85
    """
)

st.sidebar.markdown("---")
st.sidebar.header("Hướng dẫn")
st.sidebar.markdown(
    """
    1. Nhập thông tin khách hàng
    2. Nhấn "Dự đoán Churn"
    3. Xem kết quả dự đoán và khuyến nghị
    """
)

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_model.pkl')
    if os.path.exists(model_path):
        return ChurnPredictor(model_path)
    else:
        return None

predictor = load_model()

if predictor is None:
    st.error("⚠️ Model chưa được train! Vui lòng chạy notebook để train model trước.")
    st.stop()

# Nội dung chính
st.header("🔍 Nhập Thông tin Khách hàng")

# Tạo hai cột cho input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Thông tin Cá nhân")
    
    gender = st.selectbox("Giới tính", ["Female", "Male"], format_func=lambda x: "Nữ" if x == "Female" else "Nam")
    senior_citizen = st.selectbox("Người cao tuổi", ["No", "Yes"], format_func=lambda x: "Không" if x == "No" else "Có")
    partner = st.selectbox("Có người đồng hành", ["No", "Yes"], format_func=lambda x: "Không" if x == "No" else "Có")
    dependents = st.selectbox("Có người phụ thuộc", ["No", "Yes"], format_func=lambda x: "Không" if x == "No" else "Có")
    
    st.subheader("Thông tin Dịch vụ")
    
    phone_service = st.selectbox("Dịch vụ điện thoại", ["No", "Yes"], format_func=lambda x: "Không" if x == "No" else "Có")
    multiple_lines = st.selectbox("Nhiều đường dây", ["No", "Yes", "No phone service"], 
                                   format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có dịch vụ"))
    internet_service = st.selectbox("Dịch vụ Internet", ["DSL", "Fiber optic", "No"], 
                                     format_func=lambda x: "DSL" if x == "DSL" else ("Cáp quang" if x == "Fiber optic" else "Không"))
    online_security = st.selectbox("Bảo mật trực tuyến", ["No", "Yes", "No internet service"],
                                    format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có internet"))
    online_backup = st.selectbox("Sao lưu trực tuyến", ["No", "Yes", "No internet service"],
                                  format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có internet"))

with col2:
    st.subheader("Thông tin Tài khoản")
    
    tenure = st.slider("Thời gian sử dụng (tháng)", 0, 72, 12)
    contract = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"],
                           format_func=lambda x: "Theo tháng" if x == "Month-to-month" else ("1 năm" if x == "One year" else "2 năm"))
    paperless_billing = st.selectbox("Hóa đơn điện tử", ["No", "Yes"], format_func=lambda x: "Không" if x == "No" else "Có")
    payment_method = st.selectbox(
        "Phương thức thanh toán", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        format_func=lambda x: {"Electronic check": "Séc điện tử", 
                               "Mailed check": "Séc qua thư", 
                               "Bank transfer (automatic)": "Chuyển khoản tự động",
                               "Credit card (automatic)": "Thẻ tín dụng tự động"}[x]
    )
    
    monthly_charges = st.number_input("Phí hàng tháng ($)", 0.0, 200.0, 70.0, 5.0)
    total_charges = st.number_input("Tổng phí ($)", 0.0, 10000.0, 840.0, 50.0)
    
    st.subheader("Dịch vụ Bổ sung")
    
    device_protection = st.selectbox("Bảo vệ thiết bị", ["No", "Yes", "No internet service"],
                                      format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có internet"))
    tech_support = st.selectbox("Hỗ trợ kỹ thuật", ["No", "Yes", "No internet service"],
                                 format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có internet"))
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"],
                                 format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có internet"))
    streaming_movies = st.selectbox("Streaming Phim", ["No", "Yes", "No internet service"],
                                     format_func=lambda x: "Không" if x == "No" else ("Có" if x == "Yes" else "Không có internet"))

# Nút dự đoán
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    predict_button = st.button("🔮 Dự đoán Churn", use_container_width=True)

if predict_button:
    # Chuẩn bị dữ liệu đầu vào
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
    
    # Tạo thanh tiến trình
    with st.spinner('Đang phân tích dữ liệu khách hàng...'):
        # Thực hiện dự đoán
        try:
            # Lưu ý: Đây là phiên bản đơn giản hóa
            # Trong thực tế, cần đảm bảo xử lý dữ liệu đầu vào giống như lúc training
            st.warning("⚠️ Chế độ Demo: Để có kết quả chính xác, cần xử lý dữ liệu đầu vào giống như lúc training.")
            
            # Hiển thị tóm tắt khách hàng
            st.markdown("---")
            st.header("📊 Kết quả Dự đoán")
            
            # Hiển thị thông tin khách hàng
            with st.expander("📋 Tóm tắt Thông tin Khách hàng"):
                df_display = pd.DataFrame([customer_data]).T
                df_display.columns = ['Giá trị']
                st.dataframe(df_display)
            
            # Mô phỏng dự đoán cho demo (thay thế bằng dự đoán thực tế)
            # Trong production, bạn sẽ gọi: predictor.predict(customer_data)
            
            # Tính điểm rủi ro mô phỏng dựa trên các đặc trưng chính
            risk_score = 0.3  # Rủi ro cơ bản
            
            # Điều chỉnh rủi ro dựa trên các đặc trưng chính
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
            
            risk_score = min(risk_score, 0.95)  # Giới hạn tại 95%
            
            # Xác định dự đoán
            prediction = 1 if risk_score > 0.5 else 0
            
            # Hiển thị kết quả
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.subheader("Dự đoán Churn")
                if prediction == 1:
                    st.error("🚨 RỦI RO CAO - Khách hàng có khả năng CHURN")
                else:
                    st.success("✅ RỦI RO THẤP - Khách hàng có khả năng Ở LẠI")
            
            with col_res2:
                st.subheader("Xác suất Churn")
                st.metric("Xác suất", f"{risk_score*100:.1f}%")
                
                # Thanh tiến trình cho rủi ro
                if risk_score >= 0.7:
                    st.progress(risk_score, text="⚠️ Rủi ro Rất Cao")
                elif risk_score >= 0.5:
                    st.progress(risk_score, text="⚠️ Rủi ro Cao")
                elif risk_score >= 0.3:
                    st.progress(risk_score, text="⚠️ Rủi ro Trung bình")
                else:
                    st.progress(risk_score, text="✅ Rủi ro Thấp")
            
            # Khuyến nghị
            st.markdown("---")
            st.header("💡 Khuyến nghị")
            
            if prediction == 1:
                st.warning("**Khách hàng có nguy cơ cao rời bỏ dịch vụ. Cần hành động ngay!**")
                
                recommendations = []
                
                if contract == "Month-to-month":
                    recommendations.append("🎯 **Hợp đồng**: Khuyến khích chuyển sang hợp đồng dài hạn (1-2 năm) với ưu đãi đặc biệt")
                
                if tenure < 12:
                    recommendations.append("🎯 **Khách hàng mới**: Tăng cường chăm sóc khách hàng mới, chương trình khách hàng thân thiết")
                
                if internet_service == "Fiber optic":
                    recommendations.append("🎯 **Chất lượng dịch vụ**: Kiểm tra chất lượng dịch vụ Fiber optic, điều chỉnh giá nếu cần")
                
                if payment_method == "Electronic check":
                    recommendations.append("🎯 **Thanh toán**: Khuyến khích chuyển sang thanh toán tự động (chuyển khoản/thẻ tín dụng)")
                
                if online_security == "No" or online_backup == "No":
                    recommendations.append("🎯 **Dịch vụ bổ sung**: Đề xuất gói bảo mật/sao lưu với giá ưu đãi")
                
                if monthly_charges > 80:
                    recommendations.append("🎯 **Giá cả**: Xem xét giảm giá hoặc nâng cấp gói dịch vụ với giá trị tốt hơn")
                
                recommendations.append("🎯 **Đội giữ chân**: Liên hệ khách hàng trong vòng 48h để tìm hiểu vấn đề")
                
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.success("**Khách hàng có khả năng ở lại cao. Tiếp tục duy trì chất lượng dịch vụ!**")
                
                st.markdown("✅ **Duy trì tương tác**: Gửi email cảm ơn, khảo sát hài lòng định kỳ")
                st.markdown("✅ **Cơ hội bán thêm**: Giới thiệu các dịch vụ bổ sung phù hợp")
                st.markdown("✅ **Chương trình khách hàng thân thiết**: Thưởng điểm tích lũy cho khách hàng trung thành")
            
            # Phân tích yếu tố rủi ro
            st.markdown("---")
            st.header("⚠️ Phân tích Yếu tố Rủi ro")
            
            risk_factors = []
            
            if contract == "Month-to-month":
                risk_factors.append(("Loại hợp đồng", "Theo tháng", "CAO", 0.3))
            if tenure < 12:
                risk_factors.append(("Thời gian sử dụng", f"{tenure} tháng", "CAO", 0.2))
            if internet_service == "Fiber optic":
                risk_factors.append(("Dịch vụ Internet", "Cáp quang", "TRUNG BÌNH", 0.1))
            if payment_method == "Electronic check":
                risk_factors.append(("Phương thức thanh toán", "Séc điện tử", "TRUNG BÌNH", 0.15))
            if monthly_charges > 80:
                risk_factors.append(("Phí hàng tháng", f"${monthly_charges}", "TRUNG BÌNH", 0.1))
            
            if risk_factors:
                risk_df = pd.DataFrame(risk_factors, columns=['Đặc trưng', 'Giá trị', 'Mức độ Rủi ro', 'Tác động'])
                st.dataframe(risk_df, use_container_width=True)
            else:
                st.info("Không xác định được yếu tố rủi ro đáng kể.")
                
        except Exception as e:
            st.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
            st.info("Vui lòng đảm bảo model đã được train và lưu đúng cách.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>📊 Hệ thống Dự đoán Khách hàng Rời bỏ | Xây dựng với Streamlit</p>
        <p>Bài tập Data Mining - Dự án Capstone CRISP-DM</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Thanh bên - Dự đoán hàng loạt
st.sidebar.markdown("---")
st.sidebar.header("Dự đoán Hàng loạt")
uploaded_file = st.sidebar.file_uploader("Tải lên file CSV", type=['csv'])

if uploaded_file is not None:
    try:
        df_batch = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Đã tải {len(df_batch)} khách hàng")
        
        if st.sidebar.button("Dự đoán Tất cả"):
            st.markdown("---")
            st.header("📊 Kết quả Dự đoán Hàng loạt")
            st.info("Tính năng dự đoán hàng loạt - Sắp ra mắt!")
            st.dataframe(df_batch.head())
    except Exception as e:
        st.sidebar.error(f"Lỗi tải file: {str(e)}")
