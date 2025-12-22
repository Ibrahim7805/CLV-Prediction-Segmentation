import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

cluster_by_country = pd.read_csv(r"C:\Users\USER\AI\Route\My  Project\Unsuper\cluster_by_country.csv")
cluster_by_category = pd.read_csv(r"C:\Users\USER\AI\Route\My  Project\Unsuper\cluster_by_category.csv")
cluster_valid = pd.read_csv(r"C:\Users\USER\AI\Route\My  Project\Unsuper\cluster_validation.csv")
RFM_5 = pd.read_csv(r"C:\Users\USER\AI\Route\My  Project\Unsuper\RFM_5.csv")


#================================================================================================================


st.title("Customer Segmentation Explorer")

list_of_customer= list(RFM_5['Customer Persona'].value_counts().index)
options_Clusters = st.selectbox("Select a Cluster:", list_of_customer)

Country_Cluster = cluster_by_country[cluster_by_country['Customer Persona'] == options_Clusters]
Category_Cluster = cluster_by_category[cluster_by_category['Customer Persona'] == options_Clusters]
Valid_Cluster = cluster_valid[cluster_valid['Customer Persona'] == options_Clusters]

Country_Cluster = Country_Cluster.sort_values(by='Count', ascending=False)
Category_Cluster = Category_Cluster.sort_values(by='Count', ascending=False)
Valid_Cluster = Valid_Cluster.sort_values(by='ClusterSize', ascending=False)


#================================================================================================================


st.header("Cluster By Country")
st.write(Country_Cluster)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(Country_Cluster['Country'].iloc[:10], Country_Cluster['Count'].iloc[:10], color='skyblue')
ax.set_xlabel("Number of Customers")
ax.set_ylabel("Country")
ax.set_title(f"Customer Distribution for {options_Clusters} Cluster")
st.pyplot(fig)


#================================================================================================================


st.header("Cluster By Category")
st.write(Category_Cluster)


#================================================================================================================


st.header("Cluster Validation")
st.write(Valid_Cluster)

fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.pie(
    cluster_valid['ClusterSize'],
    labels=cluster_valid['Customer Persona'],
    autopct='%1.1f%%',
    startangle=90)
ax1.set_title("Cluster Distribution by Size")
st.pyplot(fig1)
