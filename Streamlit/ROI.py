import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============================================================
# Load Data
# ============================================================


RFM_5 = pd.read_csv(r"C:\Users\USER\AI\Route\My  Project\Unsuper\RFM_5.csv")
cluster_by_country = pd.read_csv(r"C:\Users\USER\AI\Route\My  Project\Unsuper\cluster_by_country.csv")
cluster_by_category = pd.read_csv(r"C:\Users\USER\AI\Route\My  Project\Unsuper\cluster_by_category.csv")
cluster_valid = pd.read_csv(r"C:\Users\USER\AI\Route\My  Project\Unsuper\cluster_validation.csv")
probs_customers = pd.read_csv(r"C:\Users\USER\AI\Route\My  Project\Unsuper\probs_customers.csv")

data = pd.read_csv(r"C:\Users\USER\AI\Route\My  Project\Preprocessing\data.csv")
Country = data.groupby('CustomerID')['Country'].unique().apply(lambda x: x[0])
new_RFM = RFM_5.merge(Country, on='CustomerID')

# ============================================================
# Page Title
# ============================================================

st.title("Campaign Performance Simulator")
st.caption("Estimate ROI for marketing campaigns by customer segment.")

# ============================================================
# Input
# ============================================================

list_of_segments = list(new_RFM['Customer Persona'].value_counts().index)
segments = st.multiselect("Select Segments:", list_of_segments)


cost_per_contact = st.number_input("Cost per Contact ($):", value=0.5)      # تكلفة التواصل مع عميل واحد في الحملة
response_rate = st.slider("Expected Response Rate (%):", 1, 100, 10) / 100  # النسبة المتوقعة من العملاء اللي هيتجاوبوا مع الحملة
uplift = st.number_input("Average Uplift per Responder ($):", value=20.0)   # متوسط الزيادة في الإنفاق لكل عميل استجاب للحملة
margin = st.slider("Margin (%):", 1, 100, 30) / 100                         # هامش الربح كنسبة مئوية من الإيراد
budget_cap = st.number_input("Budget Cap ($):", value=10000.0)              #  الحد الأقصى للميزانية المسموح بها للحملة.


# ============================================================
# Equation
# ============================================================


selected_data = new_RFM[new_RFM['Customer Persona'].isin(segments)]  # جدول فيه كل العملاء المستهدفين في الحملة.
num_Audience = len(selected_data)                                    # إجمالي تكلفة الحملة

cost = num_Audience * cost_per_contact
if  cost > budget_cap:
    st.warning("Campaign exceeds budget cap!")

revenue = num_Audience * response_rate * uplift * margin              # الإيراد المتوقع

net_profit = revenue - cost                                           #صافي الربح

roi = (net_profit / cost) if cost > 0 else 0                          # العائد على الاستثمار

# ============================================================
# Output
# ============================================================

st.metric("Total Contacts", num_Audience)
st.metric("Estimated Revenue", f"${revenue:,.2f}")
st.metric("Estimated Cost", f"${cost:,.2f}")
st.metric("Net Profit", f"${net_profit:,.2f}")
st.metric("ROI", f"{roi:.2f}")




fig, ax = plt.subplots()
rates = [0.05, 0.10, 0.15, 0.20]
roi_values = [(num_Audience * r * uplift * margin - cost) / cost for r in rates]
ax.plot([r*100 for r in rates], roi_values, marker='o')
ax.set_xlabel("Response Rate (%)")
ax.set_ylabel("ROI")
ax.set_title("ROI Sensitivity Analysis")
st.pyplot(fig)


st.download_button("Download Target List",
                   selected_data.to_csv(index=False).encode('utf-8'),
                   "target_list.csv", "text/csv")