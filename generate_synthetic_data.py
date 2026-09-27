"""
Synthetic Data Generator for E-Commerce Customer Churn Prediction
Modeled strictly after the Kaggle E-Commerce Churn Dataset Schema.

Department of Artificial Intelligence & Machine Learning (AIML)
G H Raisoni College of Engineering and Management, Jalgaon
"""

import os
import argparse
import numpy as np
import pandas as pd

def generate_ecommerce_churn_data(n_samples: int = 5630, random_state: int = 42, inject_missing: bool = True) -> pd.DataFrame:
    """
    Generates synthetic e-commerce transactional and behavioral data matching
    the Kaggle E-Commerce Churn Dataset distribution and schema.
    
    Includes:
    - Customer demographics & tenure
    - RFM (Recency, Frequency, Monetary) metrics
    - Behavioral drivers: Discount Usage, Order Count, Day Since Last Order, Complaints
    - Realistic class imbalance (~17-20% churn)
    - Optional missing values mimicking Kaggle real-world conditions
    """
    np.random.seed(random_state)
    
    # 1. Customer Demographics & Account Tenure
    customer_ids = np.arange(50001, 50001 + n_samples)
    
    # Tenure in months (gamma distributed with realistic right skew)
    tenure_raw = np.random.gamma(shape=2.5, scale=4.5, size=n_samples)
    tenure = np.clip(np.round(tenure_raw), 0, 61).astype(float)
    
    # Preferred Login Device
    login_devices = ['Mobile Phone', 'Phone', 'Computer']
    preferred_login_device = np.random.choice(login_devices, size=n_samples, p=[0.55, 0.25, 0.20])
    
    # City Tier (1, 2, 3)
    city_tier = np.random.choice([1, 2, 3], size=n_samples, p=[0.65, 0.10, 0.25])
    
    # Distance from Warehouse to Home (km)
    warehouse_to_home = np.random.exponential(scale=12.0, size=n_samples) + 5
    warehouse_to_home = np.clip(np.round(warehouse_to_home), 5, 127).astype(float)
    
    # Preferred Payment Mode
    payment_modes = ['Debit Card', 'Credit Card', 'E wallet', 'UPI', 'COD']
    preferred_payment_mode = np.random.choice(payment_modes, size=n_samples, p=[0.38, 0.32, 0.16, 0.08, 0.06])
    
    # Gender
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.60, 0.40])
    
    # Hours Spent on App
    hour_spend_on_app = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n_samples, p=[0.05, 0.27, 0.48, 0.18, 0.02])
    
    # Number of Devices Registered
    num_devices = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.05, 0.15, 0.40, 0.30, 0.08, 0.02])
    
    # Preferred Order Category
    order_cats = ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others']
    prefered_order_cat = np.random.choice(order_cats, size=n_samples, p=[0.37, 0.35, 0.15, 0.09, 0.04])
    
    # Satisfaction Score (1 to 5)
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.12, 0.15, 0.30, 0.22, 0.21])
    
    # Marital Status
    marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=n_samples, p=[0.33, 0.52, 0.15])
    
    # Number of Delivery Addresses
    number_of_address = np.random.poisson(lam=3.5, size=n_samples) + 1
    number_of_address = np.clip(number_of_address, 1, 22)
    
    # 2. Behavioral Drivers
    # Customer Complaint in last month (0 or 1) - Massive Churn Driver
    complain = np.random.choice([0, 1], size=n_samples, p=[0.72, 0.28])
    
    # Order Amount Hike From Last Year (%)
    order_hike = np.random.normal(loc=15.5, scale=3.8, size=n_samples)
    order_hike = np.clip(np.round(order_hike), 11.0, 26.0)
    
    # Order Count (Frequency)
    order_count = np.random.geometric(p=0.35, size=n_samples)
    order_count = np.clip(order_count, 1, 16).astype(float)
    
    # Coupon Used (Discount Usage)
    coupon_used = np.random.binomial(n=order_count.astype(int), p=0.45)
    coupon_used = np.clip(coupon_used, 0, 16).astype(float)
    
    # Day Since Last Order (Recency)
    days_since_last = np.random.exponential(scale=4.5, size=n_samples)
    day_since_last_order = np.clip(np.round(days_since_last), 0, 46).astype(float)
    
    # Cashback Amount ($ / INR equivalent)
    cashback_amount = np.random.normal(loc=177.0, scale=49.0, size=n_samples)
    cashback_amount = np.clip(np.round(cashback_amount, 2), 0.0, 324.99)
    
    # 3. RFM Engineered Metrics
    # Monetary = estimated total customer spend
    estimated_monetary = (order_count * 45.0) + (cashback_amount * 3.8) + (order_hike * 8.0)
    discount_ratio = np.round(coupon_used / (order_count + 1e-5), 3)
    
    # 4. Probabilistic Target (Churn) Calibration
    # Realistic e-commerce non-linear behavioral patterns
    # Churn is driven by customer complaints, high inactivity recency, short tenure, low satisfaction
    churn_signal = (
        ((complain == 1) & (day_since_last_order >= 8)) |
        ((tenure <= 4) & (satisfaction_score <= 2)) |
        ((warehouse_to_home > 30) & (complain == 1)) |
        ((satisfaction_score == 1) & (order_count <= 2)) |
        (day_since_last_order >= 24) |
        ((complain == 1) & (satisfaction_score <= 3)) |
        ((city_tier == 3) & (marital_status == 'Single') & (tenure <= 6))
    )

    churn_latent = churn_signal.astype(float)
    # Realistic stochastic behavior (label noise: 9%)
    flip_mask = np.random.rand(n_samples) < 0.088
    churn = np.where(flip_mask, 1 - churn_latent, churn_latent).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Churn': churn,
        'Tenure': tenure,
        'PreferredLoginDevice': preferred_login_device,
        'CityTier': city_tier,
        'WarehouseToHome': warehouse_to_home,
        'PreferredPaymentMode': preferred_payment_mode,
        'Gender': gender,
        'HourSpendOnApp': hour_spend_on_app,
        'NumberOfDeviceRegistered': num_devices,
        'PreferedOrderCat': prefered_order_cat,
        'SatisfactionScore': satisfaction_score,
        'MaritalStatus': marital_status,
        'NumberOfAddress': number_of_address,
        'Complain': complain,
        'OrderAmountHikeFromlastYear': order_hike,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': day_since_last_order,
        'CashbackAmount': cashback_amount,
        'Monetary': np.round(estimated_monetary, 2),
        'DiscountRatio': discount_ratio
    })
    
    # 5. Inject realistic Kaggle missing values (~4-8% in select columns)
    if inject_missing:
        missing_spec = {
            'Tenure': 0.047,
            'WarehouseToHome': 0.045,
            'HourSpendOnApp': 0.045,
            'OrderAmountHikeFromlastYear': 0.047,
            'CouponUsed': 0.046,
            'OrderCount': 0.046,
            'DaySinceLastOrder': 0.055
        }
        for col, rate in missing_spec.items():
            mask = np.random.rand(n_samples) < rate
            df.loc[mask, col] = np.nan

    return df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic E-Commerce Churn dataset")
    parser.add_argument("--samples", type=int, default=5630, help="Number of records to generate (default: 5630)")
    parser.add_argument("--output", type=str, default="dataset/ecommerce_churn_dataset.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-missing", action="store_true", help="Disable missing values injection")
    
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"Generating {args.samples} synthetic e-commerce records (Seed={args.seed})...")
    df = generate_ecommerce_churn_data(
        n_samples=args.samples,
        random_state=args.seed,
        inject_missing=not args.no_missing
    )
    
    df.to_csv(args.output, index=False)
    
    churn_rate = (df['Churn'].sum() / len(df)) * 100
    print(f"Successfully created: {args.output}")
    print(f"Total Rows: {len(df):,}, Columns: {len(df.columns)}")
    print(f"Target Distribution: Non-Churn = {(df['Churn'] == 0).sum():,} ({100-churn_rate:.1f}%), Churn = {(df['Churn'] == 1).sum():,} ({churn_rate:.1f}%)")
    print(f"Missing Values Summary:\n{df.isna().sum()[df.isna().sum() > 0]}")

if __name__ == "__main__":
    main()
