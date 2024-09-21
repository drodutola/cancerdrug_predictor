#!  C:\Users\ODUTOPX\OneDrive - AbbVie Inc (O365)\Desktop\Project Heart Attack\myenv\Scripts\python.exe


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import joblib

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("GDSC_data.csv")
    return data

def preprocess_data(data):
    # Select relevant features
    features = ['CELL_LINE_NAME', 'TCGA_DESC', 'PATHWAY_NAME']
    target = 'DRUG_NAME'
    
    # Encode categorical variables
    le = LabelEncoder()
    for feature in features:
        data[feature] = le.fit_transform(data[feature])
    
    data[target] = le.fit_transform(data[target])
    
    X = data[features]
    y = data[target]
    
    return X, y, le

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy

# Streamlit app
def main():
    st.title("Cancer Drug Prediction App")
    
    data = load_data()
    X, y, le = preprocess_data(data)
    model, scaler, accuracy = train_model(X, y)
    
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    st.header("Predict Drug")
    cell_line_name = st.selectbox("Cell Line Name", data['CELL_LINE_NAME'].unique())
    tcga_desc = st.selectbox("TCGA Description", data['TCGA_DESC'].unique())
    pathway_name = st.selectbox("Pathway Name", data['PATHWAY_NAME'].unique())
    
    if st.button("Predict"):
        input_data = np.array([[
            le.transform([cell_line_name])[0],
            le.transform([tcga_desc])[0],
            le.transform([pathway_name])[0]
        ]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        predicted_drug = le.inverse_transform(prediction)[0]
        
        st.success(f"Predicted Drug: {predicted_drug}")

if __name__ == "__main__":
    main()
