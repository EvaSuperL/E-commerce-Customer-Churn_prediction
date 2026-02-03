
"""
train.py - Script to train the customer churn prediction model
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
import yaml
import sys
import os

# Add parent directory to path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_and_preprocess_data(config):
    """Load and preprocess the data"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'online_retail_II.csv')

    print("Loading data...")
    df = pd.read_csv(data_path, encoding=config['data']['encoding'])

    # Data cleaning
    print("Cleaning data...")
    df = df.dropna(subset=['Customer ID'])
    df['Customer ID'] = df['Customer ID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['Price']

    return df

def create_customer_features(df, config):
    """Create customer-level features"""
    analysis_date = df['InvoiceDate'].max() + timedelta(days=1)
    churn_threshold = config['churn']['definition_days']

    print("Creating customer features...")

    # Group by customer
    customer_features = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (analysis_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum',
        'StockCode': 'nunique',
        'Country': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    }).rename(columns={
        'InvoiceDate': 'recency',
        'Invoice': 'frequency',
        'TotalPrice': 'monetary',
        'StockCode': 'product_variety',
        'Country': 'country'
    })

    # Calculate additional features
    customer_features['avg_basket_size'] = df.groupby('Customer ID')['TotalPrice'].mean()
    customer_features['customer_tenure'] = df.groupby('Customer ID').apply(
        lambda x: (analysis_date - x['InvoiceDate'].min()).days
    )

    customer_features['purchase_regularity'] = df.groupby('Customer ID').apply(
        lambda x: x['InvoiceDate'].sort_values().diff().dt.days.std() if len(x) > 1 else 0
    )

    customer_features['favorite_day'] = df.groupby('Customer ID')['InvoiceDate'].apply(
        lambda x: x.dt.dayofweek.mode()[0] if len(x.mode()) > 0 else -1
    )

    customer_features['favorite_hour'] = df.groupby('Customer ID')['InvoiceDate'].apply(
        lambda x: x.dt.hour.mode()[0] if len(x.mode()) > 0 else -1
    )

    customer_features['has_return_history'] = df.groupby('Customer ID').apply(
        lambda x: (x['Quantity'] < 0).any()
    ).astype(int)

    customer_features['avg_days_between_orders'] = df.groupby('Customer ID').apply(
        lambda x: x['InvoiceDate'].sort_values().diff().dt.days.mean() if len(x) > 1 else 0
    )

    # Define churn
    customer_features['churn'] = (customer_features['recency'] > churn_threshold).astype(int)

    return customer_features.reset_index()

def prepare_model_data(customer_df, config):
    """Prepare data for modeling"""
    features_to_use = [
        'recency', 'frequency', 'monetary', 'avg_basket_size',
        'product_variety', 'customer_tenure', 'purchase_regularity',
        'favorite_day', 'favorite_hour', 'avg_days_between_orders',
        'has_return_history', 'country'
    ]

    target = 'churn'

    X = customer_df[features_to_use].copy()
    y = customer_df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['model']['test_size'], 
        random_state=config['model']['random_state'], 
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config['model']['validation_size'],
        random_state=config['model']['random_state'], 
        stratify=y_train
    )

    # Create preprocessing pipeline
    numeric_features = [col for col in X.columns if col != 'country']
    categorical_features = ['country']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, preprocessor, features_to_use

def train_models(X_train, X_val, y_train, y_val, config):
    """Train and compare multiple models"""
    models = {}
    results = []

    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(
        random_state=config['model']['random_state'],
        max_iter=1000,
        class_weight='balanced'
    )
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model

    print("\nTraining Random Forest with hyperparameter tuning...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }

    rf_model = RandomForestClassifier(random_state=config['model']['random_state'])
    rf_grid = GridSearchCV(
        rf_model, rf_param_grid, cv=config['model']['cv_folds'],
        scoring='roc_auc', n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train, y_train)
    models['Random Forest'] = rf_grid.best_estimator_

    print("\nTraining XGBoost with hyperparameter tuning...")
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0]
    }

    xgb_model = XGBClassifier(
        random_state=config['model']['random_state'],
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_grid = GridSearchCV(
        xgb_model, xgb_param_grid, cv=config['model']['cv_folds'],
        scoring='roc_auc', n_jobs=-1, verbose=0
    )
    xgb_grid.fit(X_train, y_train)
    models['XGBoost'] = xgb_grid.best_estimator_

    print("\nTraining Neural Network...")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=config['model']['random_state'],
        early_stopping=True,
        validation_fraction=0.1
    )
    nn_model.fit(X_train, y_train)
    models['Neural Network'] = nn_model

    # Evaluate all models
    for name, model in models.items():
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        results.append({
            'model': name,
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred),
            'recall': recall_score(y_val, y_val_pred),
            'f1': f1_score(y_val, y_val_pred),
            'roc_auc': roc_auc_score(y_val, y_val_proba)
        })

    return models, results

def select_best_model(models, results):
    """Select the best model based on F1-Score"""
    results_df = pd.DataFrame(results)
    best_idx = results_df['f1'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name}")
    print(f"F1-Score: {results_df.loc[best_idx, 'f1']:.4f}")

    return best_model, best_model_name

def save_model_artifacts(model, preprocessor, features, metadata, config):
    """Save model and artifacts"""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save preprocessor
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")

    # Save features
    features_path = os.path.join(model_dir, 'features.pkl')
    joblib.dump(features, features_path)
    print(f"Features saved to {features_path}")

    # Save metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    return {
        'model_path': model_path,
        'preprocessor_path': preprocessor_path,
        'features_path': features_path,
        'metadata_path': metadata_path
    }

def main():
    """Main training function"""
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # Load configuration
    config = load_config()

    # Load and preprocess data
    df = load_and_preprocess_data(config)

    # Create customer features
    customer_df = create_customer_features(df, config)
    print(f"Created features for {len(customer_df)} customers")
    print(f"Churn rate: {customer_df['churn'].mean():.2%}")

    # Prepare model data
    (X_train, X_val, X_test, y_train, y_val, y_test, 
     preprocessor, features) = prepare_model_data(customer_df, config)

    # Train models
    models, results = train_models(X_train, X_val, y_train, y_val, config)

    # Select best model
    best_model, best_model_name = select_best_model(models, results)

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }

    print("\nTest Set Performance:")
    for metric, value in test_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    # Create metadata
    metadata = {
        'model_name': best_model_name,
        'model_type': type(best_model).__name__,
        'training_date': datetime.now().isoformat(),
        'features': features,
        'test_performance': test_metrics,
        'dataset_info': {
            'n_customers': len(customer_df),
            'churn_rate': float(customer_df['churn'].mean()),
            'n_features': X_train.shape[1]
        }
    }

    # Save artifacts
    artifacts = save_model_artifacts(best_model, preprocessor, features, metadata, config)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return artifacts

if __name__ == "__main__":
    main()
