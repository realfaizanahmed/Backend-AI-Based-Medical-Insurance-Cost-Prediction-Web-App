```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Medical Insurance Cost Prediction API",
    description="Hybrid Ridge Regression + FNN + Linear Regression API",
    version="1.0.0"
)


# --------------------------------------------------
# Model file paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")
RIDGE_MODEL_PATH = os.path.join(BASE_DIR, "ridge_model.pkl")
FNN_MODEL_PATH = os.path.join(BASE_DIR, "fnn_model.keras")
META_MODEL_PATH = os.path.join(BASE_DIR, "meta_model.pkl")


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    ridge_model = joblib.load(RIDGE_MODEL_PATH)
    fnn_model = load_model(FNN_MODEL_PATH)
    meta_model = joblib.load(META_MODEL_PATH)

    models_loaded = True
    model_error = None

except Exception as e:
    models_loaded = False
    model_error = str(e)

    preprocessor = None
    ridge_model = None
    fnn_model = None
    meta_model = None


# --------------------------------------------------
# Input data model
# --------------------------------------------------

class UserInput(BaseModel):
    age: float
    bmi: float
    children: int
    sex: str
    smoker: str
    region: str


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
async def home():
    return {
        "message": "Medical Insurance Cost Prediction API",
        "status": "running",
        "models_loaded": models_loaded
    }


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get("/health")
async def health():
    if models_loaded:
        return {
            "status": "healthy",
            "models_loaded": True
        }

    return JSONResponse(
        {
            "status": "error",
            "models_loaded": False,
            "error": model_error
        },
        status_code=500
    )


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict/")
async def predict(data: UserInput):

    if not models_loaded:
        return JSONResponse(
            {
                "error": "Models could not be loaded.",
                "details": model_error
            },
            status_code=500
        )

    try:

        # Convert input into DataFrame
        user_df = pd.DataFrame([
            {
                "age": data.age,
                "bmi": data.bmi,
                "children": data.children,
                "sex": data.sex,
                "smoker": data.smoker,
                "region": data.region
            }
        ])

        # Apply the same preprocessing used during training
        user_preprocessed = preprocessor.transform(user_df)

        # Ridge prediction
        ridge_pred = ridge_model.predict(user_preprocessed)

        # FNN prediction
        fnn_pred = fnn_model.predict(
            user_preprocessed,
            verbose=0
        ).flatten()

        # Combine base-model predictions
        combined_pred = np.column_stack(
            (
                ridge_pred,
                fnn_pred
            )
        )

        # Meta-model prediction
        final_pred = meta_model.predict(combined_pred)

        predicted_cost = float(final_pred[0])

        return {
            "predicted_cost": predicted_cost
        }

    except Exception as e:

        return JSONResponse(
            {
                "error": str(e)
            },
            status_code=400
        )
```
