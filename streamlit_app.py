import streamlit as st
import pandas as pd


def load_csv(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()


def main():
    st.title("FFA Trading")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        st.success("File successfully uploaded.")
        df = load_csv(uploaded_file)

        if not df.empty:
            st.subheader("Preview of Data")
            st.dataframe(df.head())
        else:
            st.warning("Uploaded file could not be read or is empty.")


if __name__ == "__main__":
    main()



