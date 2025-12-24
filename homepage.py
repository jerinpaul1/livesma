import streamlit as st


# ----------------------------- PAGE SETUP -----------------------------
st.set_page_config(page_title="Jerin's Financial Dashboard", layout="wide")

# ----------------------------- FRONT PAGE -----------------------------
st.title("📊 Jerin's Financial Dashboard")
if st.button("📂 View All My Projects"):
        st.markdown('<meta http-equiv="refresh" content="0; url=https://jerinpaul.com/projects">', unsafe_allow_html=True)

st.text("""
        Welcome! 
        Here you can find some of the programs i've worked on while at university.
        I typically make them using Google Colab and then try to translate it into streamlit compatible pages to put here.
        If you would like to see more or even just contact me, head on over to my website using the view projects button!
        
        Choose an app from the left:
        - Live SMA Dashboard: View live moving averages and trading signals.
        - Multi-Asset Monte Carlo Simulator: Run portfolio simulations using Monte Carlo.
        """)