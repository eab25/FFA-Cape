import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()


def show_summary(df: pd.DataFrame):
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())


def plot_column(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found for plotting.")
        return

    column = st.selectbox("Select a column to plot", numeric_cols)
    
    fig, ax = plt.subplots()
    df[column].plot(kind='line', ax=ax, title=f"Line Plot: {column}")
    st.pyplot(fig)


def main():
    st.title("FFA Trading")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = load_csv(uploaded_file)

        if not df.empty:
            st.success("File successfully uploaded and read.")
            st.subheader("Preview of Data")
            st.dataframe(df.head())

            show_summary(df)
            plot_column(df)
        else:
            st.warning("Uploaded file could not be read or is empty.")


if __name__ == "__main__":
    main()


