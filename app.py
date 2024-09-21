#!  C:\Users\ODUTOPX\OneDrive - AbbVie Inc (O365)\Desktop\Project Heart Attack\myenv\Scripts\python.exe


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import streamlit as st
import joblib

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("GDSC_data.csv")
    return data

class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns
        for col in self.columns:
            self.encoders[col] = LabelEncoder().fit(X[col].astype(str))
        return self

    def transform(self, X):
        output = X.copy()
        for col in self.columns:
            output[col] = self.encoders[col].transform(X[col].astype(str))
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        output = X.copy()
        for col in self.columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output

def preprocess_data(data):
    # Feature engineering
    data['TCGA_PATHWAY'] = data['TCGA_DESC'] + '_' + data['PATHWAY_NAME']
    
    features = ['CELL_LINE_NAME', 'TCGA_DESC', 'PATHWAY_NAME', 'TCGA_PATHWAY', 'PUTATIVE_TARGET', 'MIN_CONC', 'MAX_CONC', 'LN_IC50', 'AUC', 'RMSE', 'Z_SCORE']
    target = 'DRUG_NAME'
    
    X = data[features]
    y = data[target]
    
    # Encode categorical features
    categorical_features = ['CELL_LINE_NAME', 'TCGA_DESC', 'PATHWAY_NAME', 'TCGA_PATHWAY', 'PUTATIVE_TARGET']
    feature_encoder = MultiColumnLabelEncoder(columns=categorical_features)
    X_encoded = feature_encoder.fit_transform(X)
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X_encoded, y_encoded, feature_encoder, target_encoder, features

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=8)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
    
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

# Streamlit app
def main():
    st.title("Cancer Drug Prediction App")
    
    data = load_data()
    X, y, feature_encoder, target_encoder, features = preprocess_data(data)
    model, scaler, selector, accuracy = train_model(X, y)
    
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    st.header("Predict Drug")
    cell_line_name = st.selectbox("Cell Line Name", data['CELL_LINE_NAME'].unique())
    tcga_desc = st.selectbox("TCGA Description", data['TCGA_DESC'].unique())
    pathway_name = st.selectbox("Pathway Name", data['PATHWAY_NAME'].unique())
    putative_target = st.selectbox("Putative Target", data['PUTATIVE_TARGET'].unique())
    min_conc = st.number_input("Minimum Concentration", value=data['MIN_CONC'].mean())
    max_conc = st.number_input("Maximum Concentration", value=data['MAX_CONC'].mean())
    ln_ic50 = st.number_input("LN IC50", value=data['LN_IC50'].mean())
    auc = st.number_input("AUC", value=data['AUC'].mean())
    rmse = st.number_input("RMSE", value=data['RMSE'].mean())
    z_score = st.number_input("Z Score", value=data['Z_SCORE'].mean())
    
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'CELL_LINE_NAME': [cell_line_name],
            'TCGA_DESC': [tcga_desc],
            'PATHWAY_NAME': [pathway_name],
            'TCGA_PATHWAY': [f"{tcga_desc}_{pathway_name}"],
            'PUTATIVE_TARGET': [putative_target],
            'MIN_CONC': [min_conc],
            'MAX_CONC': [max_conc],
            'LN_IC50': [ln_ic50],
            'AUC': [auc],
            'RMSE': [rmse],
            'Z_SCORE': [z_score]
        })
        
        input_encoded = feature_encoder.transform(input_data)
        input_selected = selector.transform(input_encoded)
        input_scaled = scaler.transform(input_selected)
        prediction = model.predict(input_scaled)
        predicted_drug = target_encoder.inverse_transform(prediction)[0]
        
        st.success(f"Predicted Drug: {predicted_drug}")

if __name__ == "__main__":
    main()
