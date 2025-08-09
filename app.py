import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Live SMA Dashboard", layout="wide")
st.title("📈 Live Moving Average Crossover Dashboard")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
short_window = st.sidebar.number_input("Short SMA Window (minutes)", min_value=1, max_value=60, value=5)
long_window = st.sidebar.number_input("Long SMA Window (minutes)", min_value=1, max_value=120, value=15)

@st.cache_data(ttl=60)
def get_data(ticker):
    df = yf.download(ticker, period="1d", interval="1m")
    if df.empty:
        return pd.DataFrame()  # Return empty if no data
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    df.dropna(inplace=True)
    return df

df = get_data(ticker)

if df.empty or len(df) < 2:
    st.warning("Not enough data to calculate signals yet. Try a different ticker or wait for market data.")
else:
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Check for NaNs in SMA columns
    if pd.isna(latest['SMA_short']) or pd.isna(latest['SMA_long']) or pd.isna(prev['SMA_short']) or pd.isna(prev['SMA_long']):
        signal = "Waiting for SMA data..."
    else:
        if latest['SMA_short'] > latest['SMA_long'] and prev['SMA_short'] <= prev['SMA_long']:
            signal = "BUY"
        elif latest['SMA_short'] < latest['SMA_long'] and prev['SMA_short'] >= prev['SMA_long']:
            signal = "SELL"
        else:
            signal = "HOLD"

    # Plot candlestick chart with SMAs
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlesticks"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_short'],
        mode='lines', name=f"SMA {short_window}"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_long'],
        mode='lines', name=f"SMA {long_window}"
    ))
    fig.update_layout(
        title=f"{ticker} Live Chart",
        xaxis_rangeslider_visible=False
    )

    # Show metrics and chart
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Current Price", value=f"${latest['Close']:.2f}")
        st.metric(label="Short SMA", value=f"${latest['SMA_short']:.2f}")
    with col2:
        st.metric(label="Long SMA", value=f"${latest['SMA_long']:.2f}")
        st.metric(label="Signal", value=signal)

    st.plotly_chart(fig, use_container_width=True)
