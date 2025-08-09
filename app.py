import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# SETTINGS
st.set_page_config(page_title="Live SMA Dashboard", layout="wide")
st.title("📈 Live Moving Average Crossover Dashboard")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
short_window = st.sidebar.number_input("Short SMA Window (minutes)", min_value=1, value=5)
long_window = st.sidebar.number_input("Long SMA Window (minutes)", min_value=1, value=15)

# Fetch data
@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_data(ticker):
    return yf.download(ticker, period="1d", interval="1m")

# Get raw data for plotting
df_raw = fetch_data(ticker)

if df_raw.empty:
    st.warning("No market data found.")
else:
    # Copy for SMA calculations
    df = df_raw.copy()
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()

    # For signal detection, drop NaNs
    df_signals = df.dropna().copy()

    # Determine signal
    if len(df_signals) < 2:
        signal = "Waiting for enough data..."
    else:
        latest = df_signals.iloc[-1]
        prev = df_signals.iloc[-2]

        try:
            if latest['SMA_short'] > latest['SMA_long'] and prev['SMA_short'] <= prev['SMA_long']:
                signal = "BUY"
            elif latest['SMA_short'] < latest['SMA_long'] and prev['SMA_short'] >= prev['SMA_long']:
                signal = "SELL"
            else:
                signal = "HOLD"
        except Exception as e:
            st.error(f"Error calculating signal: {e}")
            signal = "Error"

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Current Price", value=f"${df_raw['Close'].iloc[-1]:.2f}")
        st.metric(label=f"Short SMA ({short_window}m)",
                  value=f"${df['SMA_short'].iloc[-1]:.2f}" if pd.notna(df['SMA_short'].iloc[-1]) else "N/A")
    with col2:
        st.metric(label=f"Long SMA ({long_window}m)",
                  value=f"${df['SMA_long'].iloc[-1]:.2f}" if pd.notna(df['SMA_long'].iloc[-1]) else "N/A")
        st.metric(label="Signal", value=signal)

    # Plot candlestick chart with SMAs
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_raw.index,
        open=df_raw['Open'],
        high=df_raw['High'],
        low=df_raw['Low'],
        close=df_raw['Close'],
        name="Candlesticks"
    ))

    # Short SMA
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_short'],
        mode='lines',
        name=f"SMA {short_window}"
    ))

    # Long SMA
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_long'],
        mode='lines',
        name=f"SMA {long_window}"
    ))

    fig.update_layout(
        title=f"{ticker} Live Chart",
        xaxis_rangeslider_visible=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
