import streamlit as st
import pandas as pd
from io import BytesIO


st.set_page_config(page_title="Export Target Lists", page_icon="⬇️", layout="wide")
st.title("⬇️ Export Targeted Customer Lists")

# ==================================================================================================
# Load Data
RFM_5 = pd.read_csv(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Building models\Unsupervised ML\RFM_5.csv")
data = pd.read_csv(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Preprocessing\data.csv")

# Add Country
Country = data.groupby('CustomerID')['Country'].unique().apply(lambda x: x[0])
new_RFM = RFM_5.merge(Country, on='CustomerID')

with st.expander("🔎 Columns in file"):
    st.write(new_RFM.columns.tolist())

# ==================================================================================================
#  Filters
with st.sidebar:
    st.header("Filters")

    personas = st.multiselect("Customer Personas", new_RFM['Customer Persona'].unique())
    countries = st.multiselect("Countries", new_RFM['Country'].unique())

    st.subheader("Thresholds")
    r_min = st.number_input("Min Recency (≥)", value=0)
    f_min = st.number_input("Min Frequency (≥)", value=0)
    m_min = st.number_input("Min Monetary (≥)", value=0.0)

    st.subheader("Columns to Export")
    cols = st.multiselect("Select columns", new_RFM.columns.tolist())

    st.subheader("File Format")
    fmt = st.radio("Choose format:", ["CSV", "XLSX"], index=0)

# ==================================================================================================
# Apply Filters
filtered = new_RFM.copy()

if personas:
    filtered = filtered[filtered["Customer Persona"].isin(personas)]

if countries:
    filtered = filtered[filtered["Country"].isin(countries)]

filtered = filtered[filtered["Recency"] >= r_min]
filtered = filtered[filtered["Frequency"] >= f_min]
filtered = filtered[filtered["Monetary"] >= m_min]

# Keep only selected columns
if cols:
    filtered = filtered[cols]

count = len(filtered)
st.subheader(f"Preview — rows selected: **{count:,}**")

# ==================================================================================================
# Preview + Export
if count == 0:
    st.warning("No data matches the current filter settings. Modify the filters and try again.")
else:
    st.dataframe(filtered.head(10), use_container_width=True)


    filename_base =  "target_customers_export"

    # CSV Export
    if fmt == "CSV":
        st.download_button(
            "⬇️ Download CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"{filename_base}.csv",
            mime="text/csv"
        )
    # XLSX Export
    else:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            filtered.to_excel(writer, index=False, sheet_name="audience")

        buffer.seek(0)
        st.download_button(
            "⬇️ Download Excel (XLSX)",
            buffer,
            file_name=f"{filename_base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

