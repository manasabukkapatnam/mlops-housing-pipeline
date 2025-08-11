from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging
import sqlite3
from datetime import datetime
import os
from typing import List

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to load model and scaler
model = None
scaler = None

try:
    # Try different model names
    model_files = [
        'models/best_model_randomforest.pkl',
        'models/best_model_linearregression.pkl',
        'models/best_model_decisiontree.pkl'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            logger.info(f"Model loaded from {model_file}")
            break
    
    if os.path.exists('models/scaler.pkl'):
        scaler = joblib.load('models/scaler.pkl')
        logger.info("Scaler loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")

app = FastAPI(
    title="Housing Price Prediction API",
    description="API for predicting California housing prices using ML",
    version="1.0.0"
)

class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average household occupancy")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str = "1.0.0"
    timestamp: str

# Initialize SQLite database for logging
def init_db():
    conn = sqlite3.connect('logs/predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            features TEXT,
            prediction REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.get("/")
async def root():
    return {
        "message": "Housing Price Prediction API", 
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": ["/predict", "/health", "/metrics"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HousingFeatures):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Model or scaler not loaded. Please train the model first."
        )
    
    try:
        # Convert to numpy array
        feature_array = np.array([[
            features.MedInc, features.HouseAge, features.AveRooms,
            features.AveBedrms, features.Population, features.AveOccup,
            features.Latitude, features.Longitude
        ]])
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        # Log prediction
        timestamp = datetime.now().isoformat()
        
        # Log to database
        try:
            conn = sqlite3.connect('logs/predictions.db')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO predictions (timestamp, features, prediction) VALUES (?, ?, ?)",
                (timestamp, str(features.dict()), float(prediction))
            )
            conn.commit()
            conn.close()
        except Exception as db_error:
            logger.warning(f"Database logging failed: {db_error}")
        
        # Log to file
        logger.info(f"Prediction: {prediction:.4f} for {features.dict()}")
        
        return PredictionResponse(
            prediction=float(prediction),
            timestamp=timestamp
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint"""
    try:
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        
        # Get prediction count
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        # Get recent predictions
        cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE timestamp > datetime('now', '-1 hour')
        """)
        recent_predictions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_predictions": total_predictions,
            "predictions_last_hour": recent_predictions,
            "model_version": "1.0.0",
            "status": "active",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return {"error": "Unable to fetch metrics", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)