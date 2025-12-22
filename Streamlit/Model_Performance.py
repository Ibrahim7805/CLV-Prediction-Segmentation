import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from tensorflow.keras.models import load_model


# Load Models
LR = joblib.load(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Building models\Supervised ML\LR_clv.pkl")
RF = joblib.load(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Building models\Supervised ML\RF_clv.pkl")
XGB = joblib.load(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Building models\Supervised ML\XGB_Best_clv.pkl")
DNN = load_model(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Building models\DL\DNN_CLV_predict.keras", compile=False)
Kmeans = joblib.load(r"C:\Users\USER\AI\Route\My  Project\Unsuper\KMeans_5.pkl")

scaler_reg = joblib.load(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Preprocessing\scaler_rfm.pkl")
scaler_clus = joblib.load(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Building models\Unsupervised ML\MinMaxScaler_rmf.pkl")



st.title("📊 Model Performance Monitoring & Prediction Confidence")

st.header("Select Model")
model_reg = st.selectbox("Model for Regression", ['None', 'Linear Regression', 'Random Forest', 'XGBoost', 'DNN'])
model_clus = st.selectbox("Model for clustering", ['None', 'KMeans clustering'])

# ===============================
# Regression
# ===============================

if model_reg != 'None':
    input_data = []
    for feature in ['Recency', 'Frequency', 'Monetary']:
        value = st.number_input(f"{feature}")
        input_data.append(value)

    if st.button('predict_reg'):
        input_arr = np.array(input_data).reshape(1, -1)
        input_std = scaler_reg.transform(input_arr)

        if model_reg == 'Linear Regression':
            pred = LR.predict(input_std)
            st.metric("MAE", "0.63")
            st.metric("MSE", "0.73")
            st.metric("R2", "50.86%")

        elif model_reg == 'Random Forest':
            pred = RF.predict(input_std)
            st.metric("MAE", "0.12")
            st.metric("MSE", "0.03")
            st.metric("R2", "97.78%")

        elif model_reg == 'XGBoost':
            pred = XGB.predict(input_std)
            st.metric("MAE", "0.12")
            st.metric("MSE", "0.02")
            st.metric("R2", "98.33%")

        elif model_reg == 'DNN':
            pred = DNN.predict(input_std)

            st.metric("MAE", "0.24")
            st.metric("MSE", "0.1")
            st.metric("R2", "93.17%")

            # Load saved history
            with open("history.json", "r") as f:
                history_loaded = json.load(f)

            # Plot Training vs Validation MSE
            fig, ax = plt.subplots()
            ax.plot(history_loaded['loss'], label='Train MSE')
            ax.plot(history_loaded['val_loss'], label='Val MSE')
            ax.legend()
            ax.set_title("Training vs Validation MSE")
            st.pyplot(fig)

            # Plot Training vs Validation R2
            fig, ax = plt.subplots()
            ax.plot(history_loaded['r2_score'], label='Train R2')
            ax.plot(history_loaded['val_r2_score'], label='Val R2')
            ax.legend()
            ax.set_title("Training vs Validation R2")
            st.pyplot(fig)

        pred_value = pred[0]

        pred_value = pred_value.item()

        if pred_value > 2000:
            st.success(f"CLV Category: HIGH Predicted CLV = {pred_value:.2f}")
        elif pred_value > 1000:
            st.warning(f"CLV Category: MEDIUM Predicted CLV = {pred_value:.2f}")
        else:
            st.error(f"CLV Category: LOW Predicted CLV = {pred_value:.2f}")


# ===============================
# Clustering S
# ===============================

if model_clus == 'KMeans clustering':
    input_data = []
    for feature in ['Recency', 'Frequency', 'Monetary']:
        value = st.number_input(f"{feature}")
        input_data.append(value)

    if st.button('predict_cluster'):
        input_arr = np.array(input_data, dtype=np.float64).reshape(1, -1)
        input_std = scaler_clus.transform(input_arr)
        input_std = input_std.astype(np.float64)
        pred = Kmeans.predict(input_std)


        pred_label = int(pred[0])

        persona_map = {
            0: 'VIP',
            3: 'Loyal',
            1: 'New',
            4: 'At Risk',
            2: 'Lost'
        }

        pred_Persona = persona_map[pred_label]

        if pred_Persona in ['VIP', 'Loyal']:
            st.success(f"Customer Persona: {pred_Persona}")
        elif pred_Persona in ['New', 'At Risk']:
            st.warning(f"Customer Persona: {pred_Persona}")
        else:
            st.error(f"Customer Persona: {pred_Persona}")

        # Plot silhouette scores
        s_scores = [0.7022, 0.6161, 0.6204, 0.5829, 0.4943, 0.4738, 0.5139, 0.4352]

        fig, ax = plt.subplots()
        ax.plot(range(2, 10), s_scores, marker='o')
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Method")
        st.pyplot(fig)
