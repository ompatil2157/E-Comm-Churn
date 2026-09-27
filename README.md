# 🛍️ E-Commerce Customer Churn Prediction Using Machine Learning

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-91%25_Acc-118D57?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **Academic Project Affiliation:**  
> **Department of Artificial Intelligence & Machine Learning (AIML)**  
> **G H Raisoni College of Engineering and Management, Jalgaon**

---

## 📌 Executive Summary

Customer churn is one of the most critical challenges facing the e-commerce and retail sector, where acquiring a new customer is **5 to 7 times more expensive** than retaining an existing one. 

This project provides an end-to-end Machine Learning pipeline and an interactive enterprise Streamlit dashboard designed to detect customer churn vulnerability before it occurs. The model evaluates customer demographics, purchase recency, order frequency, monetary value (**RFM**), discount coupon engagement, and customer satisfaction complaints to produce real-time churn risk probabilities along with prescriptive retention strategies.

---

## 🏆 Model Performance Benchmark

Four distinct machine learning algorithms were trained and evaluated on stratified hold-out test sets matching the project requirements:

| Algorithm | Role | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | ⭐ **Primary Production Model** | **91.0%** | **0.890** | **0.86** | **0.84** | **0.85** |
| **Random Forest** | Challenger Ensemble | **87.0%** | 0.845 | 0.81 | 0.78 | 0.79 |
| **Decision Tree (CART)** | Interpretable Rules | **79.0%** | 0.760 | 0.72 | 0.69 | 0.70 |
| **Logistic Regression** | Linear Baseline | **73.0%** | 0.710 | 0.65 | 0.62 | 0.63 |

---

## 📊 Dataset & Feature Engineering (Kaggle E-Commerce Schema)

The synthetic data generation engine strictly follows the real-world **Kaggle E-Commerce Churn Dataset** schema:

### 1. Customer Demographics & Account Tenure:
- `CustomerID`: Unique identifier
- `Tenure`: Customer lifetime duration in months (0 to 60)
- `CityTier`: Urbanization tier (1, 2, or 3)
- `WarehouseToHome`: Distance to delivery hub (km)
- `PreferredLoginDevice`: Mobile Phone, Phone, or Computer
- `PreferredPaymentMode`: Debit Card, Credit Card, E-wallet, UPI, COD
- `Gender`, `MaritalStatus`, and `NumberOfAddress`

### 2. RFM & Transactional Drivers:
- **Recency:** `DaySinceLastOrder` (0 to 45 days since last purchase)
- **Frequency:** `OrderCount` (Number of orders placed)
- **Monetary:** `Monetary` (Calculated spend based on order frequency and cashback rewards)
- **Discount Usage:** `CouponUsed` and `DiscountRatio` (`CouponUsed / OrderCount`)
- **Order Hike:** `OrderAmountHikeFromlastYear` (Year-over-year purchase growth)

### 3. Behavioral Risk Factors:
- `Complain`: Whether a dispute was registered in the last month (0 or 1) — **#1 Churn Driver**
- `SatisfactionScore`: Rated on a 1 to 5 scale
- `HourSpendOnApp`: Daily app engagement

### 4. Data Preprocessing & Imbalance Handling:
- **Missing Values:** Median imputation for continuous variables; mode imputation for categorical attributes.
- **Class Imbalance:** Stratified sampling and `scale_pos_weight` / balanced class weighting to handle ~18.5% churn prevalence.
- **Scaling:** Robust z-score standardization via `StandardScaler`.
- **Encoding:** One-Hot Encoding for multi-class nominal features.

---

## 🎨 UI & Micro-Interaction Features

- **Interactive Risk Gauge:** Plotly circular gauge with dynamic color zones (Green `<35%`, Yellow `35-65%`, Red `>65%`).
- **Lottie CSS/JS Animations:** Micro-interactions for loading spinners, success state (low risk), and alert state (high risk).
- **Explainable AI Chart:** Interactive horizontal bar chart visualizing the top 10 behavioral drivers of churn.
- **Batch CSV Scoring:** Ability to upload bulk CSV customer records and download instant risk-scored predictions.
- **Actionable Retention Advisory:** Generates automated business recommendations (e.g., win-back discounts, dedicated VIP managers, express shipping perks) tailored to each customer profile.

---

## 📂 Repository Structure

```
ecommerce-churn-ml/
├── dataset/
│   ├── generate_synthetic_data.py   # Kaggle-schema synthetic data generator
│   └── ecommerce_churn_dataset.csv  # Generated dataset (5,630 records)
├── models/
│   ├── train_models.py              # ML pipeline training 4 algorithms
│   ├── xgboost_model.joblib         # Serialized Primary XGBoost model
│   ├── random_forest_model.joblib   # Serialized Random Forest model
│   ├── decision_tree_model.joblib   # Serialized Decision Tree model
│   ├── logistic_regression_model.joblib # Serialized Logistic Regression
│   ├── preprocessor.joblib          # Scikit-Learn ColumnTransformer pipeline
│   ├── feature_importance.json      # Sorted feature importance scores
│   └── model_metrics.json           # Comprehensive benchmark metrics
├── app.py                           # Enterprise Streamlit Dashboard
├── utils.py                         # Plotly charts, CSS, Lottie states & logic
├── requirements.txt                 # Pinned project dependencies
├── README.md                        # Documentation & setup guide
└── .gitignore                       # Ignored files for clean git tracking
```

---

## 🚀 Quickstart & Setup Instructions

### 1. Prerequisites
Ensure you have **Python 3.10+** or **Python 3.11** installed.

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/ecommerce-churn-ml.git
cd ecommerce-churn-ml
```

### 3. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Generate Dataset & Train Models
```bash
# Generate the Kaggle-schema dataset
python dataset/generate_synthetic_data.py --samples 5630

# Train and benchmark all 4 machine learning models
python models/train_models.py
```

### 6. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this entire repository to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "feat: complete e-commerce churn prediction platform"
   git branch -M main
   git remote add origin https://github.com/<your-username>/ecommerce-churn-ml.git
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository.
4. Set the **Main file path** to `app.py`.
5. Click **Deploy!**

---

## 🏛️ Academic Credits & Attribution

- **Institution:** G H Raisoni College of Engineering and Management, Jalgaon
- **Department:** Artificial Intelligence & Machine Learning (AIML)
- **Domain:** Predictive Analytics, Customer Retention & Explainable Machine Learning
