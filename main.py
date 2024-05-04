# -*- coding: utf-8 -*-
"""
Created on Sat May  4 19:31:59 2024

@author: sumaiyya
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Load the trained model
model_path = 'C:\\Users\\mannh\\Dta science By Upgrad\\ASSIGHNMENTS\\ml-api-project\\creditcardfault.sav'
credit_card_model = joblib.load(model_path)

@app.post('/predict_fraud')
def predict_fraud(input_data: ModelInput):
    input_values = np.array([getattr(input_data, f'V{i}') for i in range(1, 29)])  # Extract values for V1 to V28
    input_values = np.append(input_values, input_data.Time)  # Append Time feature
    input_values = np.append(input_values, input_data.Amount)  # Append Amount feature
    prediction = credit_card_model.predict([input_values])
    if prediction[0] == 1:
        return {'fraud_prediction': 'Fraudulent transaction detected'}
    else:
        return {'fraud_prediction': 'Normal transaction'}
