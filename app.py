#!  C:\Users\ODUTOPX\OneDrive - AbbVie Inc (O365)\Desktop\Project Heart Attack\myenv\Scripts\python.exe


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import streamlit as st
import joblib

# Try to import SMOTE, but provide an alternative if it's not available
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("SMOTE is not available. We'll proceed without handling class imbalance.")

# ... [rest of the code remains the same until the train_model function]

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=8)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Handle class imbalance if SMOTE is available
    if SMOTE_AVAILABLE:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_selected, y_train
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train_resampled)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, selector, accuracy

# ... [rest of the code remains the same]
