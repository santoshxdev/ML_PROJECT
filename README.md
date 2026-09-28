# Customer Lifetime Value Prediction Using Elastic Net Regression

## Project Title
Customer Lifetime Value (CLV) Prediction Using Elastic Net Regression

## Subject
Machine Learning — B.Tech Project

## Objective
Predict how much revenue a customer will generate in a future period, using historical transactional behaviour modelled with Elastic Net Regression. The project demonstrates correct temporal feature engineering to prevent target leakage.

## Dataset
- **Source**: UCI Online Retail Dataset (`data/Online Retail.xlsx`)
- **Original Size**: 541,909 rows × 8 columns
- **Cleaned Size**: 392,692 transactions from 4,338 customers
- **Date Range**: 2010-12-01 to 2011-12-09

### Columns
| Column | Description |
|---|---|
| InvoiceNo | Unique transaction ID (prefix 'C' = cancellation) |
| StockCode | Product identifier |
| Description | Product description |
| Quantity | Units purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit (GBP) |
| CustomerID | Unique customer identifier |
| Country | Customer country |

---

## Data Cleaning (`src/preprocess_data.py`)
1. Remove exact duplicate rows
2. Drop rows with missing `CustomerID`
3. Remove cancelled invoices (`InvoiceNo` starting with `C`)
4. Remove rows where `Quantity <= 0` or `UnitPrice <= 0`
5. Parse `InvoiceDate` as datetime
6. Compute `TotalAmount = Quantity × UnitPrice`

Output: `processed_data/clean_transactions.csv`

---

## Temporal CLV Methodology (`src/create_clv_dataset.py`)

To prevent **target leakage**, a strict chronological split is applied:

| Period | Dates | Purpose |
|---|---|---|
| Historical (80%) | 2010-12-01 → 2011-09-25 | Feature engineering |
| Future (20%) | 2011-09-25 → 2011-12-09 | CLV target calculation |

Customer features are built **exclusively from historical transactions**.  
`Future_CLV` is calculated **exclusively from future transactions**.  
Customers with no future purchases receive `Future_CLV = 0`.

---

## Feature Engineering

| Feature | Definition |
|---|---|
| Recency | Days from last historical purchase to the split date |
| Frequency | Number of unique invoices in the historical period |
| MonetaryValue | Total revenue in the historical period |
| AverageOrderValue | MonetaryValue ÷ Frequency |
| TotalQuantity | Total units purchased historically |
| CustomerTenure | Days between first and last historical purchase |
| PurchaseFrequency | Orders ÷ (Tenure + 1) |
| UniqueProducts | Number of distinct products bought |

---

## Target Variable
```
Future_CLV = Sum(TotalAmount) over future period per customer
```

---

## Elastic Net Regression (`src/train_model.py`)

- **Algorithm**: `ElasticNetCV` with 5-fold cross-validation
- **Preprocessing**: `StandardScaler`
- **Pipeline**: `StandardScaler → ElasticNetCV`
- **Hyperparameters tuned**:
  - `alpha`: 80 log-spaced values from 1e-4 to 1e4
  - `l1_ratio`: [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
- **Train/Test Split**: 80% train / 20% test (stratified, random_state=42)
- **Best alpha**: 74.72
- **Best l1_ratio**: 1.0 (pure Lasso penalty selected by CV)

---

## Model Evaluation (`src/evaluate_model.py`)

Evaluated on the **held-out test set only** (710 customers):

| Metric | Value |
|---|---|
| MAE | 478.77 |
| MSE | 1,749,435.77 |
| RMSE | 1,322.66 |
| R² | 0.4493 |

---

## Project Structure

```
ML_anti.project/
├── data/
│   └── Online Retail.xlsx          # Original dataset (untouched)
├── src/
│   ├── __init__.py
│   ├── inspect_dataset.py          # Step 1: Dataset inspection
│   ├── preprocess_data.py          # Step 2: Data cleaning
│   ├── create_clv_dataset.py       # Step 3: Feature engineering + CLV target
│   ├── train_model.py              # Step 4: Model training
│   └── evaluate_model.py           # Step 5: Evaluation + plots
├── processed_data/
│   ├── clean_transactions.csv      # Cleaned transaction data
│   └── customer_clv_dataset.csv    # Customer-level features + Future_CLV
├── models/
│   ├── elastic_net_clv_pipeline.joblib
│   ├── model_features.txt
│   ├── X_test.npy
│   ├── y_test.npy
│   └── feature_names.npy
├── outputs/
│   ├── dataset_summary.txt
│   ├── clv_dataset_validation.txt
│   ├── metrics/
│   │   ├── elastic_net_metrics.csv
│   │   ├── elastic_net_coefficients.csv
│   │   └── model_comparison.csv
│   └── plots/
│       ├── actual_vs_predicted.png
│       ├── residual_plot.png
│       └── elastic_net_coefficients.png
├── notebooks/
├── requirements.txt
└── README.md
```

---

## How to Install Dependencies
```bash
pip install -r requirements.txt
```

## How to Run the Pipeline

```bash
# 1. Inspect dataset
python src/inspect_dataset.py

# 2. Clean transactions
python src/preprocess_data.py

# 3. Create customer-level CLV dataset
python src/create_clv_dataset.py

# 4. Train Elastic Net model
python src/train_model.py

# 5. Evaluate model and generate plots
python src/evaluate_model.py
```
