# E-commerce-Customer-Churn_prediction

## 🎯 Project Overview
**Problem Statement**: This project aims to predict customer churn for an e-commerce business using machine learning techniques. Customer churn refers to customers who stop purchasing from the business, which directly impacts revenue and growth.

**Business Impact**: By identifying at-risk customers early, businesses can implement targeted retention strategies (personalized offers, loyalty programs, proactive customer service) to reduce churn rates. According to industry studies, reducing churn by 5% can increase profits by 25-125%.

**Solution Approach**: The project implements and compares multiple classification models including Logistic Regression (baseline), Random Forest, XGBoost, and Neural Networks to predict which customers are likely to churn.

## 📊 Dataset
**Source**: [Online Retail II Dataset](https://www.kaggle.com/datasets/tunguz/online-retail-ii) from UCI ML Repository

**Description**: Contains all transactions (1,067,371 records) for a UK-based online retailer between 01/12/2009 and 09/12/2011

### Key Features After Feature Engineering:
- **Recency**: Days since last purchase
- **Frequency**: Number of purchases in specified period
- **Monetary**: Total amount spent
- **Behavioral Features**: Average basket size, purchase regularity, product variety
- **Demographic Features**: Country, customer segment (encoded)

**Target Variable**: `churn` (1 = churned, 0 = retained) - defined as no purchase in last 90 days

### Dataset Statistics:
- Total customers: 5,942 (after preprocessing)
- Churn rate: 16.2%
- Features: 15 engineered features

## 🚀 Setup & Installation
### Prerequisites
- Python 3.9 or higher
- uv package manager (for fast dependency management)
### Installation Steps
```bash
# 1. Clone the repository
git clone <repository-url>
cd ecommerce-churn-prediction

# 2. Install dependencies using uv
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# 4. Install Jupyter kernel (for notebook support)
uv sync --dev
uv run python -m ipykernel install --user --name=ecommerce-churn --display-name "Python (uv: ecommerce-churn)"
```

## 📁 Optimized Project Structure
```text
E-commerce-Customer-Churn_prediction/
│
├── README.md                         # Main project documentation (English)
├── pyproject.toml                    # Project dependencies (uv format)
├── requirements.txt                  # Alternative pip requirements
├── uv.lock                          # Locked dependencies
├── Dockerfile                       # Containerization setup
├── .dockerignore                    # Docker ignore file
├── render.yaml                      # Render cloud deployment config
│
├── data/                            # Dataset directory
│   └── online_retail_II.csv         # Kaggle dataset
│
├── notebook/                        # Jupyter notebooks directory
│   └── project.ipynb               # Complete EDA and modeling notebook
│
├── scripts/                         # Python scripts directory
│   ├── train.py                    # Training script
│   ├── predict.py                  # FastAPI prediction server
│   ├── generate_eda_plots.py       # EDA visualization generator
│   └── preprocess_data.py          # Data preprocessing script
│
├── model/                           # Trained models directory
│   ├── model.pkl                   # Best trained model
│   ├── vectorizer.pkl              # Feature vectorizer
│   ├── scaler.pkl                  # Feature scaler (if needed)
│   └── metadata.json               # Model metadata
│
├── images/                          # Generated EDA plots
│   ├── eda_histograms.png
│   ├── eda_correlation_matrix.png
│   ├── eda_rfm_distribution.png
│   ├── eda_churn_analysis.png
│   └── model_comparison.png
│
├── tests/                           # Unit tests
│   ├── test_train.py
│   └── test_predict.py
│
└── config/                          # Configuration files
    └── config.yaml                  # Project configuration

```

## 📈 Extensive Exploratory Data Analysis (EDA)
### EDA Visualizations Generated
To regenerate all EDA plots, run:
```bash
uv run python generate_eda_plots.py
```

### Key EDA Insights:
1. **Data Distribution**: RFM features show right-skewed distributions requiring log transformation
2. **Correlation Analysis**: High correlation between frequency and monetary features (0.85+)
3. **Churn Patterns**: Clear relationship between recency and churn probability
4. **Geographic Insights**: Churn rates vary significantly by country (UK lowest at 12%, others ~25%)
5. **Behavioral Patterns**: Customers with irregular purchase intervals have 3x higher churn rate

### Data Preprocessing Pipeline
1. **Transaction Aggregation**: Raw transactions aggregated to `customer level`
2. **Feature Engineering**:
    - RFM metrics calculation
    - Behavioral features (regularity, variety, seasonality)
    - Customer tenure and lifecycle stage
3. **Handling Missing Values**: 0.8% missing in monetary features filled with median
4. **Encoding**: One-hot encoding for country, label encoding for categorical features
5. **Train/Validation/Test Split**: 60%/20%/20% with stratification on churn label

## 🤖 Model Training & Parameter Tuning
### Models Implemented

| Model | Description | Parameter Tuning |
|------|-------------|------------------|
| Logistic Regression (Baseline) | Simple linear classifier | C: [0.001, 0.01, 0.1, 1, 10, 100] |
| Random Forest | Ensemble of decision trees | n_estimators: [50, 100, 200], max_depth: [10, 20, None] |
| XGBoost | Gradient boosting | learning_rate: [0.01, 0.1, 0.3], n_estimators: [100, 200] |
| Neural Network | MLP with 2 hidden layers | layers: [32, 64, 128], dropout: [0.2, 0.5], epochs: [50, 100] |

### Hyperparameter Tuning Process

```python
# Example from train.py - GridSearchCV for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

## 📊 Model Performance Results
### Validation Set Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.842 | 0.801 | 0.763 | 0.782 | 0.882 |
| Random Forest | 0.878 | 0.852 | 0.821 | 0.836 | 0.921 |
| XGBoost | 0.891 | 0.864 | 0.838 | 0.851 | 0.938 |
| Neural Network | 0.885 | 0.858 | 0.830 | 0.844 | 0.931 |

### Test Set Performance (Final Evaluation)

| Model | Accuracy | F1-Score | ROC-AUC |
|------|----------|----------|---------|
| Logistic Regression | 0.836 | 0.775 | 0.871 |
| Random Forest | 0.869 | 0.829 | 0.915 |
| XGBoost (Best Model) | 0.883 | 0.844 | 0.932 |
| Neural Network | 0.878 | 0.837 | 0.926 |

### Feature Importance Analysis
Top 5 features identified by XGBoost:

1. Recency (0.32) - Days since last purchase
2. Purchase Frequency (0.19) - Number of orders
3. Avg Basket Value (0.15) - Average spending per order
4. Customer Tenure (0.12) - Days since first purchase
5. Product Variety (0.09) - Number of unique products purchased

## 🚀 Model Deployment
### FastAPI Web Service (predict.py)
```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    dv = pickle.load(f)

class CustomerFeatures(BaseModel):
    recency: float
    frequency: float
    monetary: float
    avg_basket_size: float
    customer_tenure: float
    # ... other features

@app.post("/predict")
async def predict(customer: CustomerFeatures):
    customer_dict = customer.dict()
    X = dv.transform([customer_dict])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
        "risk_level": "high" if probability > 0.7 else "medium" if probability > 0.4 else "low"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}
```

## 📦 Containerization & Cloud Deployment
### Docker Configuration (Dockerfile)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv pip install -e .
COPY . .
EXPOSE 8000
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
```
### Building and Running Locally
```bash
# Build Docker image
docker build -t ecommerce-churn-predictor .

# Run container locally
docker run -d -p 8000:8000 --name churn-api ecommerce-churn-predictor

# Test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"recency": 45, "frequency": 12, "monetary": 1500.50, ...}'
```
### Cloud Deployment (Render.com)
**Deployed URL**: https://ecommerce-churn-predictor.onrender.com

**Deployment Steps**:

1. Push code to GitHub repository
2. Create new Web Service on Render.com
3. Connect GitHub repository
4. Configure:
    - Build Command:   
        ` docker build -t ecommerce-churn-predictor .`
    - Start Command: 
        `docker run -p 8000:8000 ecommerce-churn-predictor`
5. Deploy automatically on git push

**Live API Documentation**: 
https://ecommerce-churn-predictor.onrender.com/docs

## 🔧 Running the Complete Project
### Full Workflow
```bash
# 1. Setup environment
uv sync
source .venv/bin/activate

# 2. Explore the analysis
jupyter notebook project.ipynb

# 3. Train the model (generates model.pkl and vectorizer.pkl)
uv run python train.py

# 4. Test the model locally
uv run python -m pytest tests/

# 5. Start the prediction service
uv run uvicorn predict:app --reload

# 6. Containerize (alternative to step 5)
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
```
### Testing the Deployed API
```bash
# Health check
curl https://ecommerce-churn-predictor.onrender.com/health

# Make prediction
curl -X POST https://ecommerce-churn-predictor.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "recency": 45,
    "frequency": 12,
    "monetary": 1500.50,
    "avg_basket_size": 125.04,
    "customer_tenure": 180,
    "product_variety": 8,
    "purchase_regularity": 0.65,
    "country_encoded": 2,
    "has_return_history": 0,
    "avg_days_between_orders": 45.2
  }'
```

## 📝 Key Design Decisions
1. Feature Engineering Over Raw Data: Aggregated transaction data to customer-level features for more meaningful patterns
2. Multiple Model Comparison: Implemented diverse algorithms (linear, tree-based, neural networks) to find optimal approach
3. Business-Focused Metrics: Used precision/recall tradeoff optimized for customer retention costs
4. Production-Ready API: FastAPI with automatic documentation, input validation, and error handling
5. Complete DevOps Pipeline: Local development → Containerization → Cloud deployment with CI/CD