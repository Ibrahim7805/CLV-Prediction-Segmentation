import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load data
data = pd.read_csv(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Preprocessing\data.csv")
RFM = pd.read_csv(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Preprocessing\RFM.csv")

# load model & scaler
model = joblib.load(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Building models\Supervised ML\XGB_Best_clv.pkl")
scaler = joblib.load(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Preprocessing\scaler_rfm.pkl")


# ==================================
# Page Config
# ==================================

st.set_page_config( page_title="CLV Analytics", page_icon="📊", layout="wide")
st.title("📊 Customer Lifetime Value Analytics")
st.markdown("Search customers, explore RFM data, and predict CLV scores (1–5).")


# ==================================
# Sidebar Navigation
# ==================================
with st.sidebar:
    st.header("⚙️ Control Panel")

    mode = st.radio(
        "Select Mode",
        ["🔍 Customer Search", "🤖 CLV Prediction"]
    )

    if mode == "🔍 Customer Search":
        data_option = st.multiselect("Select Data to Display", ["Transactions", "RFM"])
        rfm_features = st.multiselect("Select RFM Features", RFM.columns.tolist())
        data_features = st.multiselect("Select Transactions Features",data.columns.tolist())


# ==================================
# 🔍 CUSTOMER SEARCH
# ==================================

if mode == "🔍 Customer Search":
    st.header("🔍 Customer Search")

    customer_id = st.number_input(
        f"Enter Customer ID,  min_value={int(RFM.CustomerID.min())},  max_value={int(RFM.CustomerID.max())}",
        min_value=int(RFM.CustomerID.min()),
        max_value=int(RFM.CustomerID.max())
    )

    cust_data = data[data["CustomerID"] == customer_id]
    cust_rfm = RFM[RFM["CustomerID"] == customer_id]

    if cust_rfm.empty and cust_data.empty:
        st.warning(f"⚠️ No data found for Customer ID {customer_id}. Please try another ID.")
    else:
        if st.button("Search Customer"):

            if "Transactions" in data_option:
                st.header("🧾 Transaction Data")
                st.subheader("Customer Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Country ", cust_data["Country"].iloc[0])
                col2.metric("Total Spend", f"${int(cust_data["TotalPrice"].sum()):,.0f}")
                col3.metric("Quantity", f"{cust_data['Quantity'].sum()} Piece")
                months = ", ".join(cust_data["Month"].astype(str).unique())
                col4.metric("Month", months)

                st.dataframe(cust_data[data_features])


            if "RFM" in data_option:
                st.header("📈 RFM Data")
                st.subheader("Customer Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Recency (days)", int(cust_rfm["Recency"]))
                col2.metric("Frequency", int(cust_rfm["Frequency"]))
                col3.metric("Monetary", f"${cust_rfm['Monetary'].values[0]:,.0f}")

                tab1, tab2 = st.tabs(["📈 RFM Table", "📊 CLV Indicator"])
                with tab1:
                    st.dataframe(cust_rfm[rfm_features])
                with tab2:
                    clv_value = cust_rfm["CLV"].values[0]
                    st.metric("CLV Score", f"{clv_value} / 5")
                    st.progress(clv_value / 5)


else:
    st.header("🤖 CLV Prediction")
    st.write("Enter customer RFM values to predict CLV score (1–5).")

    recency = st.number_input("Recency (days since last purchase), min_value=1, max_value=400")
    frequency = st.number_input("Frequency (number of purchases), min_value=1, max_value=300")
    monetary = st.number_input("Monetary (total spend), min_value=1.0")

    if st.button("Predict CLV"):
        input_arr = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_arr)
        clv_pred = model.predict(input_scaled)[0]

        st.subheader("📊 Prediction Result")
        st.metric("Predicted CLV Score", f"{clv_pred:.1f} / 5")
        st.progress(float(clv_pred / 5))


        if clv_pred >= 4:
            st.success(f"🟢 High CLV Customer — Score: {clv_pred:.1f}")
        elif clv_pred >= 2.5:
            st.warning(f"🟡 Medium CLV Customer — Score: {clv_pred:.1f}")
        else:
            st.error(f"🔴 Low CLV Customer — Score: {clv_pred:.1f}")
