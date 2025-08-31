import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="FFA Trading", layout="wide")
st.title("FFA Trading")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin-1")

    st.success("File uploaded ✅")
    st.subheader("Preview")
    st.dataframe(df.head(50))

    # Let the user pick a numeric column to plot
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        y_col = st.selectbox("Pick a column to plot", numeric_cols, index=0)
        fig, ax = plt.subplots()
        ax.plot(df.index, df[y_col])
        ax.set_xlabel("Row index")
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} (line chart)")
        st.pyplot(fig)
    else:
        st.info("No numeric columns found to plot.")
else:
    st.info("Upload a CSV to get started.")


