# Customer Churn Prediction Project

## Mục tiêu dự án

Dự đoán khách hàng có khả năng rời bỏ dịch vụ (churn) để có chiến lược giữ chân khách hàng phù hợp.

## Cấu trúc dự án

```
customer-churn-project/
├── data/                           # Dữ liệu thô
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/                      # Jupyter notebooks phân tích
│   └── customer_churn_analysis.ipynb
├── src/                            # Source code
│   ├── preprocessing.py            # Tiền xử lý dữ liệu
│   ├── modeling.py                 # Huấn luyện mô hình
│   └── predict.py                  # Dự đoán
├── demo/                           # Demo application
│   └── app.py                      # Streamlit app
├── models/                         # Mô hình đã lưu
│   └── churn_model.pkl
├── requirements.txt                # Dependencies
└── README.md                       # Tài liệu hướng dẫn
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy notebook phân tích

```bash
jupyter notebook notebooks/customer_churn_analysis.ipynb
```

## Chạy demo application

```bash
streamlit run demo/app.py
```

## Quy trình CRISP-DM

1. **Business Understanding**: Xác định mục tiêu giảm tỷ lệ churn
2. **Data Understanding**: Phân tích EDA dữ liệu khách hàng
3. **Data Preparation**: Xử lý missing values, encoding, scaling
4. **Modeling**: Thử nghiệm Random Forest, XGBoost, Logistic Regression
5. **Evaluation**: Đánh giá với ROC-AUC, Precision, Recall, F1-Score
6. **Deployment**: Triển khai web app dự đoán

## Kết quả

- Model tốt nhất: [Sẽ cập nhật sau khi training]
- ROC-AUC Score: [Sẽ cập nhật]
- Precision/Recall: [Sẽ cập nhật]

## Tác giả

Data Mining Practice - CRISP-DM Capstone Project
