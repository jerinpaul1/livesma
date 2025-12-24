import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("📈 Live Moving Average Crossover Dashboard")
# Sidebar inputs
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
short_window = st.sidebar.number_input("Short SMA Window (minutes)", min_value=1, value=5)
long_window = st.sidebar.number_input("Long SMA Window (minutes)", min_value=1, value=15)

# Validate SMA windows
if short_window >= long_window:
    st.warning("Short window should be smaller than long window for crossovers.")

# Fetch data function (can be cached)
@st.cache_data(ttl=60) # Cache data for 60 seconds
def fetch_data(ticker, short_window, long_window):
    # Adjusted period and interval for potentially longer lookback if needed, but sticking to 1d for 1m interval
    df = yf.download(ticker, period="1d", interval="1m") # Use 1m interval for live
    if df.empty:
        return pd.DataFrame() # Return empty if no data

    # Ensure windows are not larger than available data
    if len(df) > 0:
        short_window = min(short_window, len(df))
        long_window = min(long_window, len(df))
    else:
        return pd.DataFrame() # No data to calculate SMAs

    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    df.dropna(inplace=True) # Drop rows with NaN created by rolling window
    return df

# Get data
df = fetch_data(ticker, short_window, long_window)

# Check if enough data exists
if df.empty or len(df) < max(short_window, long_window):
    st.warning("Not enough data to calculate SMAs or signals. Try a different ticker or wait for market data.")
else:
    # Market closed check
    if df.index[-1].date() < pd.Timestamp.now().date():
        st.warning("Market is closed. Showing last available data.")
    # check_signal logic (adapted for Streamlit)
    # We need at least two data points after dropping NaNs for signal calculation
    if len(df) < 2:
         signal = "Waiting for enough data..."
    else:
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Robustly check for signal after ensuring valid data
        try:
            if (latest['SMA_short'] > latest['SMA_long']).item() and (prev['SMA_short'] <= prev['SMA_long']).item():
                signal = "BUY"
            elif (latest['SMA_short'] < latest['SMA_long']).item() and (prev['SMA_short'] >= prev['SMA_long']).item():
                signal = "SELL"
            else:
                signal = "HOLD"
        except Exception as e:
            st.error(f"Error calculating signal: {e}")
            signal = "Error"
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        # Ensure we handle cases where latest or prev might not exist if df is very small but passed the initial check
        if not df.empty:
             st.metric(label="Current Price", value=f"${df['Close'].iloc[-1].item():.2f}")
             if 'SMA_short' in df.columns and not pd.isna(df['SMA_short'].iloc[-1]):
                st.metric(label=f"Short SMA ({short_window}m)", value=f"${df['SMA_short'].iloc[-1].item():.2f}")
             else:
                 st.metric(label=f"Short SMA ({short_window}m)", value="N/A")
        else:
             st.metric(label="Current Price", value="N/A")
             st.metric(label=f"Short SMA ({short_window}m)", value="N/A")

    with col2:
         if not df.empty:
            if 'SMA_long' in df.columns and not pd.isna(df['SMA_long'].iloc[-1]):
                st.metric(label=f"Long SMA ({long_window}m)", value=f"${df['SMA_long'].iloc[-1].item():.2f}")
            else:
                st.metric(label=f"Long SMA ({long_window}m)", value="N/A")
            st.metric(label="Signal", value=signal)
         else:
            st.metric(label=f"Long SMA ({long_window}m)", value="N/A")
            st.metric(label="Signal", value="N/A")
    # Plot chart with SMAs
    if not df.empty:
        fig = go.Figure()
        if 'SMA_short' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['SMA_short'],
                mode='lines', name=f"SMA {short_window}"
            ))
        if 'SMA_long' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['SMA_long'],
                mode='lines', name=f"SMA {long_window}"
            ))
        fig.update_layout(
            title=f"{ticker} Live Chart",
            xaxis_rangeslider_visible=False,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)    