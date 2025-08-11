"""
Data Preprocessing Script for California Housing Dataset
Part of MLOps Pipeline - FIXED VERSION
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """Load and preprocess California Housing dataset"""
    logger.info("Loading California Housing dataset...")
    
    # Load dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    logger.info(f"Dataset shape: {X.shape}, Target shape: {y.shape}")
    
    # Create DataFrame for easier handling
    df = pd.DataFrame(X, columns=housing.feature_names)
    df['target'] = y
    
    # Basic data validation
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Target statistics:\n{df['target'].describe()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save processed data
    np.save('data/processed/X_train.npy', X_train_scaled)
    np.save('data/processed/X_test.npy', X_test_scaled)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    # Save raw data for reference
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/california_housing.csv', index=False)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save feature names - FIX: Check if it's already a list
    feature_names = housing.feature_names
    if hasattr(feature_names, 'tolist'):
        # It's a numpy array, convert to list
        feature_names_list = feature_names.tolist()
    else:
        # It's already a list
        feature_names_list = list(feature_names)
    
    feature_info = {
        'feature_names': feature_names_list,
        'target_name': 'median_house_value',
        'description': housing.DESCR,
        'n_features': len(feature_names_list),
        'n_samples': X.shape[0]
    }
    
    with open('data/processed/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    logger.info("Data preprocessing completed successfully!")
    logger.info(f"Features: {feature_names_list}")
    logger.info(f"Scaled training data mean: {X_train_scaled.mean():.6f}")
    logger.info(f"Scaled training data std: {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def validate_data_quality(X, y):
    """Validate data quality"""
    issues = []
    
    # Check for NaN values
    if np.isnan(X).any():
        issues.append("Features contain NaN values")
    
    if np.isnan(y).any():
        issues.append("Target contains NaN values")
    
    # Check for infinite values
    if np.isinf(X).any():
        issues.append("Features contain infinite values")
    
    # Check data ranges
    if (y < 0).any():
        issues.append("Target contains negative values")
    
    return issues

def print_data_info():
    """Print information about the loaded dataset"""
    housing = fetch_california_housing()
    
    print("=== California Housing Dataset Info ===")
    print(f"Number of samples: {housing.data.shape[0]}")
    print(f"Number of features: {housing.data.shape[1]}")
    print(f"Feature names: {list(housing.feature_names)}")
    print(f"Target name: {housing.target_names if hasattr(housing, 'target_names') else 'median_house_value'}")
    print(f"Data type: {type(housing.feature_names)}")
    print("\nFeature descriptions:")
    feature_descriptions = {
        'MedInc': 'Median income in block group',
        'HouseAge': 'Median house age in block group', 
        'AveRooms': 'Average number of rooms per household',
        'AveBedrms': 'Average number of bedrooms per household',
        'Population': 'Block group population',
        'AveOccup': 'Average household occupancy',
        'Latitude': 'Block group latitude',
        'Longitude': 'Block group longitude'
    }
    
    for feature in housing.feature_names:
        desc = feature_descriptions.get(feature, 'No description available')
        print(f"  {feature}: {desc}")
    
    print(f"\nTarget range: ${housing.target.min():.1f}k - ${housing.target.max():.1f}k")
    print(f"Target mean: ${housing.target.mean():.1f}k")

if __name__ == "__main__":
    try:
        # Print dataset information first
        print_data_info()
        print("\n" + "="*50 + "\n")
        
        # Run preprocessing
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
        
        # Validate data quality
        train_issues = validate_data_quality(X_train, y_train)
        test_issues = validate_data_quality(X_test, y_test)
        
        if train_issues or test_issues:
            logger.warning(f"Data quality issues found: {train_issues + test_issues}")
        else:
            logger.info("Data quality validation passed!")
        
        # Print summary
        print("\n=== Preprocessing Summary ===")
        print(f"✅ Training samples: {X_train.shape[0]}")
        print(f"✅ Test samples: {X_test.shape[0]}")
        print(f"✅ Features: {X_train.shape[1]}")
        print(f"✅ Files saved:")
        print(f"   - data/processed/X_train.npy")
        print(f"   - data/processed/X_test.npy") 
        print(f"   - data/processed/y_train.npy")
        print(f"   - data/processed/y_test.npy")
        print(f"   - models/scaler.pkl")
        print(f"   - data/processed/feature_info.json")
        print(f"   - data/raw/california_housing.csv")
        
        print("\n🎉 Data preprocessing completed successfully!")
            
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        import traceback
        traceback.print_exc()
        raise