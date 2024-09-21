#!  C:\Users\ODUTOPX\OneDrive - AbbVie Inc (O365)\Desktop\Project Heart Attack\myenv\Scripts\python.exe



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import time
import lightgbm as lgb

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

@st.cache_resource
def train_model_lgb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 100
    }
    
    model = lgb.LGBMClassifier(**params)
    
    start_time = time.time()
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=10,
              verbose=False)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    feature_importance = model.feature_importances_
    
    return model, feature_importance, accuracy, training_time

def main():
    st.title("Cancer Drug Prediction App")
    
    data = load_data()
    
    if data is not None:
        sample_size = min(50000, len(data))
        data_sample = data.sample(n=sample_size, random_state=42)
        
        X, y, feature_encoder, target_encoder, features = preprocess_data(data_sample)
        
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a minute."):
                try:
                    model, feature_importance, accuracy, training_time = train_model_lgb(X, y)
                    st.session_state.model = model
                    st.session_state.accuracy = accuracy
                    st.session_state.feature_encoder = feature_encoder
                    st.session_state.target_encoder = target_encoder
                    st.session_state.feature_importance = feature_importance
                    st.session_state.features = features
                    st.success(f"Model training completed in {training_time:.2f} seconds. Accuracy: {accuracy:.2f}")
                    
                    # Display feature importance
                    st.subheader("Feature Importance")
                    feature_imp_df = pd.DataFrame({'feature': features, 'importance': feature_importance})
                    feature_imp_df = feature_imp_df.sort_values('importance', ascending=False).head(10)
                    st.bar_chart(feature_imp_df.set_index('feature'))
                except Exception as e:
                    st.error(f"An error occurred during model training: {str(e)}")
        
        if 'model' in st.session_state and st.session_state.model is not None:
            st.header("Predict Drug")
            cell_line_name = st.selectbox("Cell Line Name", data['CELL_LINE_NAME'].unique())
            tcga_desc = st.selectbox("TCGA Description", data['TCGA_DESC'].unique())
            pathway_name = st.selectbox("Pathway Name", data['PATHWAY_NAME'].unique())
            putative_target = st.selectbox("Putative Target", data['PUTATIVE_TARGET'].unique())
            min_conc = st.number_input("Minimum Concentration", value=float(data['MIN_CONC'].mean()))
            max_conc = st.number_input("Maximum Concentration", value=float(data['MAX_CONC'].mean()))
            ln_ic50 = st.number_input("LN IC50", value=float(data['LN_IC50'].mean()))
            auc = st.number_input("AUC", value=float(data['AUC'].mean()))
            rmse = st.number_input("RMSE", value=float(data['RMSE'].mean()))
            z_score = st.number_input("Z Score", value=float(data['Z_SCORE'].mean()))
            
            if st.button("Predict"):
                try:
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
                    prediction = st.session_state.model.predict(input_encoded)
                    predicted_drug = st.session_state.target_encoder.inverse_transform(prediction)[0]
                    
                    st.success(f"Predicted Drug: {predicted_drug}")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.info("Please train the model to make predictions.")

if __name__ == "__main__":
    main()
