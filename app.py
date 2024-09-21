#!  C:\Users\ODUTOPX\OneDrive - AbbVie Inc (O365)\Desktop\Project Heart Attack\myenv\Scripts\python.exe



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import time

# Global variable to check if SMOTE is available
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("SMOTE not available, will use simple oversampling")

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

def simple_oversample(X, y):
    class_counts = np.bincount(y)
    max_count = np.max(class_counts)
    X_resampled = []
    y_resampled = []
    for class_label in range(len(class_counts)):
        class_indices = np.where(y == class_label)[0]
        X_class = X[class_indices]
        y_class = y[class_indices]
        n_samples = len(class_indices)
        n_repeats = max_count // n_samples
        remainder = max_count % n_samples
        X_resampled.append(np.repeat(X_class, n_repeats, axis=0))
        X_resampled.append(X_class[:remainder])
        y_resampled.append(np.repeat(y_class, n_repeats))
        y_resampled.append(y_class[:remainder])
    return np.vstack(X_resampled), np.concatenate(y_resampled)

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if SMOTE_AVAILABLE:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = simple_oversample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train_resampled)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, None, accuracy, training_time

def main():
    st.title("Cancer Drug Prediction App")
    
    data = load_data()
    
    if data is not None:
        sample_size = min(50000, len(data))
        data_sample = data.sample(n=sample_size, random_state=42)
        
        X, y, feature_encoder, target_encoder, features = preprocess_data(data_sample)
        
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a minute."):
                model, scaler, selector, accuracy, training_time = train_model(X, y)
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.selector = selector
                st.session_state.accuracy = accuracy
                st.session_state.feature_encoder = feature_encoder
                st.session_state.target_encoder = target_encoder
            st.success(f"Model training completed in {training_time:.2f} seconds. Accuracy: {accuracy:.2f}")
        
        if 'model' in st.session_state and st.session_state.model is not None:
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
                
                input_encoded = st.session_state.feature_encoder.transform(input_data)
                input_selected = st.session_state.selector.transform(input_encoded)
                input_scaled = st.session_state.scaler.transform(input_selected)
                prediction = st.session_state.model.predict(input_scaled)
                predicted_drug = st.session_state.target_encoder.inverse_transform(prediction)[0]
                
                st.success(f"Predicted Drug: {predicted_drug}")
        else:
            st.info("Please train the model to make predictions.")

if __name__ == "__main__":
    main()
