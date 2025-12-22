import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


data = pd.read_csv(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Preprocessing\data.csv")


st.set_page_config(page_title="Customer Behavior & Trend", page_icon="📊", layout="wide")
st.title("Customer Behavior & Trend")
st.caption("Explore sales trends, seasonality, and geographic patterns.")

with st.sidebar:

    st.header("Filters")
    countries = st.multiselect("Countries", data['Country'].unique())
    season = st.multiselect("Season", data['Season'].unique())
    month = st.multiselect("Month", data['Month'].unique())


    st.subheader("Thresholds")
    TotalPrice = st.number_input("Min TotalPrice (≥)", value=0)
    Quantity = st.number_input("Min Quantity (≥)", value=0)

    st.subheader("Features")
    Feature1 = st.selectbox("Select Features 1 ", ['None'] + data.columns.tolist())
    Feature2 = st.selectbox("Select Features 2 ", ['None'] + data.columns.tolist())
    Feature3 = st.selectbox("Select Features 3 (optional)", ['None'] + data.columns.tolist())
    Feature4 = st.selectbox("Select Features 4 (optional)", ['None'] + data.columns.tolist())

    st.subheader("Visualization Type")
    vis_typ = st.selectbox("Select Visualization Type", ['None','Line chart', 'Heatmap', 'Bar chart', 'Box Plot'])


#====================================================================================================

# Apply Filters

filtered = data.copy()

if countries:
    filtered = filtered[filtered["Country"].isin(countries)]
if season:
    filtered = filtered[filtered["Season"].isin(season)]
if month:
    filtered = filtered[filtered["Month"].isin(month)]

filtered = filtered[filtered["TotalPrice"] >= TotalPrice]
filtered = filtered[filtered["Quantity"] >= Quantity]


def data_for_vis():
    if Feature1 != 'None' and Feature2 != 'None':
        cols = [Feature1, Feature2]

        if Feature3 != 'None':
            cols.append(Feature3)
        if Feature4 != 'None':
            cols.append(Feature4)

        return filtered[cols]
    return pd.DataFrame()


data_vis = data_for_vis()

count_rows = len(filtered)
st.markdown(f"**Rows after filtering:** {count_rows:,}")


if count_rows == 0:
    st.warning("No data matches the current filter settings. Modify the filters and try again.")
else:
    st.dataframe(filtered.head(10), use_container_width=True)

    if vis_typ == 'Line chart':

        st.info("""
        ✔ Suitable for trends over time or continuous variables.
        - Feature 1 (X-axis): must be **Categorical or Date or Numeric**.
        - Feature 2 (Y-axis): must be **Numeric**.
        """)

        fig = px.line(data_vis, x=Feature1, y=Feature2)
        fig.update_layout(
            title=f"{Feature1} vs {Feature2}",
            yaxis_title=f"{Feature2}",
            xaxis_title=f"{Feature1}",
            title_font_size=32,
        )
        st.plotly_chart(fig, use_container_width=True)


    elif vis_typ == 'Heatmap':

        st.info("""
        ✔ Shows correlation between numerical variables. 
        - All selected features must be **Numeric only**.
        """)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(data_vis.corr(numeric_only = True), annot=True, cmap="viridis", linewidths=.5, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)


    elif vis_typ == 'Bar chart':

        st.info("""
        ✔ Suitable for comparing categories.
        - Feature 1 (X-axis): must be **Categorical** (Ex- Country, Season, Month).
        - Feature 2 (Y-axis): must be **Numeric** (Ex- TotalPrice, Quantity).
        """)

        fig = px.bar(data_vis, x=Feature1, y=Feature2)
        fig.update_layout(
            title=f"{Feature1} vs {Feature2}",
            yaxis_title=f"{Feature2}",
            xaxis_title=f"{Feature1}",
            title_font_size=32,
        )
        st.plotly_chart(fig, use_container_width=True)


    elif vis_typ == 'Box Plot':
        st.info("""
        ✔ Suitable for distribution and detecting outliers.
        - Feature 1 (X-axis): must be **Categorical**.
        - Feature 2 (Y-axis): must be **Numeric**.
        """)

        fig = px.box(data_vis, x=Feature1, y=Feature2)
        fig.update_layout(
            title=f"{Feature1} vs {Feature2}",
            yaxis_title=f"{Feature2}",
            xaxis_title=f"{Feature1}",
            title_font_size=32,
        )

        st.plotly_chart(fig, use_container_width=True)
