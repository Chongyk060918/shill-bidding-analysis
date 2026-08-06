import streamlit as st
import pandas as pd

# 1. A simple title to confirm the app is live
st.title("🚧 Shill Bidding Project - Work in Progress")
st.write("Hello! My Streamlit app is successfully connected to GitHub.")

# 2. Load the data to test the file connection
# Ensure 'Shill Bidding Dataset.csv' is uploaded to your GitHub repo!
@st.cache_data
def load_data():
    return pd.read_csv('Shill Bidding Dataset.csv')

df = load_data()

# 3. Display the raw data
st.subheader("Raw Dataset Preview")
st.write("If you can see the table below, your data loaded successfully!")
st.dataframe(df.head(10))