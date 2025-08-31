import streamlit as st

st.title('🎈 FFA Cape V1 ')

st.write('Lets builde a trading strategies')
your-app/
  streamlit_app.py
  requirements.txt
  data/
    Baltic Exchange - Historic Data 020120 290825.csv
# At the top of streamlit_app.py
DEFAULT_FILE = "data/Baltic Exchange - Historic Data 020120 290825.csv"

# Replace the else-block where it shows "(none)" with:
elif os.path.exists(DEFAULT_FILE):
    dfs[os.path.basename(DEFAULT_FILE)] = load_csv(DEFAULT_FILE)
