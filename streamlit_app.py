import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title for your app
st.title("Baltic Exchange Historic Data Visualization")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload Baltic Exchange CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    st.write("Data Preview:", df.head())

    # Select columns to plot
    options = st.multiselect(
        "Select columns to plot",
        ['C5TC', 'HS7TC', 'P4TC', 'S10TC'],
        default=['C5TC', 'HS7TC', 'P4TC', 'S10TC']
    )

    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(18, 8))
    for col in options:
        ax.plot(df['Date'], df[col], label=col)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Baltic Exchange Historic Rates')
    ax.legend()
    st.pyplot(fig)
