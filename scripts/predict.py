
"""
predict.py - FastAPI server for customer churn prediction
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Optional, List, Dict, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn in e-commerce",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for loaded artifacts
model = None
preprocessor = None
features = None
metadata = None

class CustomerData(BaseModel):
    """Pydantic model for customer data input"""
    recency: float = Field(..., ge=0, le=365, description="Days since last purchase (0-365)")
    frequency: float = Field(..., ge=1, le=1000, description="Number of purchases (1-1000)")
    monetary: float = Field(..., ge=0, le=100000, description="Total amount spent (0-100000)")
    avg_basket_size: float = Field(..., ge=0, le=10000, description="Average basket size (0-10000)")
    product_variety: int = Field(..., ge=1, le=1000, description="Number of unique products purchased (1-1000)")
    customer_tenure: float = Field(..., ge=0, le=3650, description="Customer tenure in days (0-3650)")
    purchase_regularity: float = Field(..., ge=0, le=100, description="Purchase regularity (standard deviation of days between purchases)")
    favorite_day: int = Field(..., ge=-1, le=6, description="Favorite day (0=Monday, 6=Sunday, -1=unknown)")
    favorite_hour: int = Field(..., ge=-1, le=23, description="Favorite hour (-1=unknown, 0-23)")
    avg_days_between_orders: float = Field(..., ge=0, le=365, description="Average days between orders (0-365)")
    has_return_history: int = Field(..., ge=0, le=1, description="Has return history (0=no, 1=yes)")
    country: str = Field(..., min_length=2, max_length=50, description="Country code (e.g., 'United Kingdom', 'Germany')")

class BatchPredictionRequest(BaseModel):
    """Pydantic model for batch prediction"""
    customers: List[CustomerData]

class PredictionResponse(BaseModel):
    """Pydantic model for prediction response"""
    churn_prediction: int = Field(..., ge=0, le=1, description="Prediction (0=no churn, 1=churn)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    risk_level: str = Field(..., description="Risk level (low, medium, high)")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    features_used: List[str] = Field(..., description="List of features used for prediction")

class BatchPredictionResponse(BaseModel):
    """Pydantic model for batch prediction response"""
    predictions: List[PredictionResponse]
    total_customers: int
    churn_count: int
    churn_rate: float

def load_model_artifacts():
    """Load model and preprocessing artifacts"""
    global model, preprocessor, features, metadata

    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'model')

        # Load model
        model_path = os.path.join(model_dir, 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)

        # Load preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)

        # Load features
        features_path = os.path.join(model_dir, 'features.pkl')
        if os.path.exists(features_path):
            features = joblib.load(features_path)
        else:
            features = [
                'recency', 'frequency', 'monetary', 'avg_basket_size',
                'product_variety', 'customer_tenure', 'purchase_regularity',
                'favorite_day', 'favorite_hour', 'avg_days_between_orders',
                'has_return_history', 'country'
            ]

        # Load metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {"model_name": "Unknown", "training_date": "Unknown"}

        print(f"Model loaded: {metadata.get('model_name', 'Unknown')}")
        print(f"Model type: {type(model).__name__}")
        print(f"Features: {len(features)}")

        return True

    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return False

def preprocess_customer_data(customer_data: CustomerData) -> pd.DataFrame:
    """Preprocess single customer data for prediction"""
    customer_dict = customer_data.dict()
    df = pd.DataFrame([customer_dict])
    return df

def predict_churn(customer_df: pd.DataFrame) -> tuple:
    """Make churn prediction for customer data"""
    # Ensure all features are present
    for feature in features:
        if feature not in customer_df.columns:
            customer_df[feature] = 0

    # Reorder columns to match training
    customer_df = customer_df[features]

    # Preprocess
    processed_data = preprocessor.transform(customer_df)

    # Predict
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    return int(prediction), float(probability)

def interpret_prediction(probability: float) -> Dict[str, Any]:
    """Interpret prediction probability into business terms"""
    if probability >= 0.7:
        risk_level = "high"
        recommendation = "Immediate retention action required"
        confidence = probability
    elif probability >= 0.4:
        risk_level = "medium"
        recommendation = "Proactive engagement recommended"
        confidence = probability
    else:
        risk_level = "low"
        recommendation = "Monitor for changes in behavior"
        confidence = 1 - probability

    return {
        "risk_level": risk_level,
        "recommendation": recommendation,
        "confidence": confidence
    }

@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup"""
    print("Starting Customer Churn Prediction API...")
    if not load_model_artifacts():
        print("Warning: Could not load model artifacts. API will not work properly.")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    return status

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": metadata.get("model_name", "Unknown"),
        "model_type": type(model).__name__ if model else "Unknown",
        "training_date": metadata.get("training_date", "Unknown"),
        "features": features if features else [],
        "performance": metadata.get("test_performance", {}),
        "dataset_info": metadata.get("dataset_info", {})
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """Predict churn for a single customer"""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    try:
        # Preprocess customer data
        customer_df = preprocess_customer_data(customer)

        # Make prediction
        prediction, probability = predict_churn(customer_df)

        # Interpret prediction
        interpretation = interpret_prediction(probability)

        # Prepare response
        response = PredictionResponse(
            churn_prediction=prediction,
            churn_probability=probability,
            risk_level=interpretation["risk_level"],
            confidence=interpretation["confidence"],
            features_used=features
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for multiple customers"""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    try:
        predictions = []
        churn_count = 0

        for customer in request.customers:
            # Preprocess customer data
            customer_df = preprocess_customer_data(customer)

            # Make prediction
            prediction, probability = predict_churn(customer_df)

            # Interpret prediction
            interpretation = interpret_prediction(probability)

            # Count churn predictions
            if prediction == 1:
                churn_count += 1

            # Create response
            pred_response = PredictionResponse(
                churn_prediction=prediction,
                churn_probability=probability,
                risk_level=interpretation["risk_level"],
                confidence=interpretation["confidence"],
                features_used=features
            )

            predictions.append(pred_response)

        # Calculate churn rate
        churn_rate = churn_count / len(request.customers) if request.customers else 0

        response = BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(request.customers),
            churn_count=churn_count,
            churn_rate=churn_rate
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/example-request")
async def example_request():
    """Get example request for testing"""
    example_customer = CustomerData(
        recency=45,
        frequency=12,
        monetary=1500.50,
        avg_basket_size=125.04,
        product_variety=8,
        customer_tenure=180,
        purchase_regularity=15.5,
        favorite_day=2,
        favorite_hour=14,
        avg_days_between_orders=45.2,
        has_return_history=0,
        country="United Kingdom"
    )

    return {
        "example_request": example_customer.dict(),
        "curl_command": """
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "recency": 45,
    "frequency": 12,
    "monetary": 1500.50,
    "avg_basket_size": 125.04,
    "product_variety": 8,
    "customer_tenure": 180,
    "purchase_regularity": 15.5,
    "favorite_day": 2,
    "favorite_hour": 14,
    "avg_days_between_orders": 45.2,
    "has_return_history": 0,
    "country": "United Kingdom"
  }'
        """
    }

if __name__ == "__main__":
    uvicorn.run(
        "predict:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
