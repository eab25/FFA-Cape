import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("FFA Trading")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    st.write("File uploaded...")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())



