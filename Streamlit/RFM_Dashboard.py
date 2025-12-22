import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ============================================================
# Load Data
# ============================================================


RFM_5 = pd.read_csv(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Building models\Unsupervised ML\RFM_5.csv")
data = pd.read_csv(r"C:\Users\USER\ِAI_Project\ML Projects\CLV Prediction & Segmentation\Preprocessing\data.csv")


Country = data.groupby('CustomerID')['Country'].unique().apply(lambda x: x[0])
new_RFM = RFM_5.merge(Country, on='CustomerID')


# ============================================================
# Page Title
# ============================================================

st.title("RFM Analysis Dashboard")
st.caption("Explore customers, personas, spending behavior, and patterns across segments.")


# ============================================================
# Searching Section
# ============================================================

st.header("Customer Search & Filtering")
options_searching = st.selectbox("Choose a Search Method:", ('None', 'Persona', 'Country', 'CustomerID'))

if options_searching == 'Persona':
    st.subheader('Search by Persona 🎭 ')
    list_of_customer= list(new_RFM['Customer Persona'].value_counts().index)
    options_Clusters = st.selectbox("Select a Cluster:", list_of_customer)
    data_Cluster = new_RFM[new_RFM['Customer Persona'] == options_Clusters]
    st.write(f"### Results for persona: **{options_Clusters}**")
    st.dataframe(data_Cluster)

elif options_searching == 'Country':
    st.subheader('Search by Country 🌍')
    list_of_Country= list(new_RFM['Country'].value_counts().index)
    options_Country = st.selectbox("Select a Country:", list_of_Country)
    data_Country = new_RFM[new_RFM['Country'] == options_Country]
    st.write(f"### Results for country: **{options_Country}**")
    st.dataframe(data_Country)

elif options_searching == 'CustomerID':
    st.subheader('Search by Customer ID 🆔')
    st.write('Enter The Customer ID  below...')
    customer_id = st.number_input("Customer ID, min_value = 12347 , max_value = 18287")
    data_CustomerID = new_RFM[new_RFM['CustomerID'] == customer_id]
    st.write(f"### Results for Customer ID: **{customer_id}**")
    st.dataframe(data_CustomerID)


# ============================================================
# Sorting Section
# ============================================================

st.header("Customer Sorting Options")
options_Sorting = st.selectbox("Sort by:", ('None' ,'CLV', 'Persona'))

if options_Sorting == 'CLV':
    st.subheader("Sort by Customer Lifetime Value  💰")
    st.dataframe(new_RFM.sort_values(by='CLV', ascending=False))

elif options_Sorting == 'Persona':
    st.subheader("Sort by Persona Segment 🎭")
    st.dataframe(new_RFM.sort_values(by='cluster_5', ascending=True))

# ============================================================
# Visualization Section
# ============================================================

st.header("Data Visualization Explorer")
list_of_options = (['None']+list(new_RFM.columns))
features_1 = st.selectbox('Select Feature 1 ', list_of_options)
features_2 = st.selectbox('Select Feature 2 ', list_of_options)
Type_plot = st.selectbox("Select Visualization Type:", ('None' ,'bar', 'scatter', 'heatmap'))


if Type_plot == 'None':
    pass

elif Type_plot == 'heatmap':
    st.subheader('Correlation Heatmap (RFM Features)')
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_data = new_RFM[['Recency', 'Frequency', 'Monetary', 'cluster_5']]
    sns.heatmap(numeric_data.corr(), annot=True)
    st.pyplot(fig)

else:
    st.subheader(f'{Type_plot.capitalize()} Plot"')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel(f'{features_1}')
    ax.set_ylabel(f'{features_2}')
    ax.set_title(f"{features_1} vs {features_2}")

    if Type_plot == 'bar':
        st.info("Showing bar chart for first 10 customers.")
        ax.bar(new_RFM[features_1].iloc[:10], new_RFM[features_2].iloc[:10], color='skyblue')
        st.pyplot(fig)

    elif Type_plot == 'scatter':
        ax.scatter(new_RFM[features_1], new_RFM[features_2], color='skyblue')
        st.pyplot(fig)

