import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import time # Keep time for sleep if needed for refresh logic

# SETTINGS (adjust as needed for Streamlit inputs)
# These will be replaced by Streamlit widgets

st.set_page_config(page_title="Live SMA Dashboard", layout="wide")
st.title("📈 Jerin's Live Moving Average Crossover Dashboard")
if st.button("📂 View All My Projects"):
    st.markdown('<meta http-equiv="refresh" content="0; url=https://jerinpaul.com/projects">', unsafe_allow_html=True)

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
    """
    Fetches live stock data, resamples it to a continuous 1-minute series,
    and calculates Simple Moving Averages.
    """
    try:
        df = yf.download(ticker, period="1d", interval="1m")
    except Exception as e:
        st.error(f"Error fetching data for ticker {ticker}: {e}")
        return pd.DataFrame()

    # If the dataframe is empty, return it immediately
    if df.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
        return pd.DataFrame()

    # --- NEW: VALIDATION STEP ---
    # Ensure the necessary columns exist before proceeding
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Data for '{ticker}' is missing required columns. It might be an invalid ticker.")
        return pd.DataFrame()
    # --- END OF NEW CODE ---

    # Resample the data to ensure a continuous 1-minute time series.
    df_resampled = df.resample('1T').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).ffill().bfill()

    # Ensure windows are not larger than available data
    if len(df_resampled) > 0:
        short_window = min(short_window, len(df_resampled))
        long_window = min(long_window, len(df_resampled))
    else:
        return pd.DataFrame()

    # Calculate SMAs
    df_resampled['SMA_short'] = df_resampled['Close'].rolling(window=short_window).mean()
    df_resampled['SMA_long'] = df_resampled['Close'].rolling(window=long_window).mean()

    # Drop initial rows that don't have enough data for the rolling window
    df_resampled.dropna(inplace=True)
    
    return df_resampled

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
            # Removed .item() for cleaner, more robust comparison
            if (latest['SMA_short'] > latest['SMA_long']) and (prev['SMA_short'] <= prev['SMA_long']):
                signal = "BUY"
            elif (latest['SMA_short'] < latest['SMA_long']) and (prev['SMA_short'] >= prev['SMA_long']):
                signal = "SELL"
            else:
                signal = "HOLD"
        except Exception as e:
            st.error(f"Error calculating signal: {e}")
            signal = "Error"

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        if not df.empty and 'Close' in df.columns:
            st.metric(label="Current Price", value=f"${df['Close'].iloc[-1]:.2f}")
            if 'SMA_short' in df.columns and not pd.isna(df['SMA_short'].iloc[-1]):
                st.metric(label=f"Short SMA ({short_window}m)", value=f"${df['SMA_short'].iloc[-1]:.2f}")
            else:
                st.metric(label=f"Short SMA ({short_window}m)", value="N/A")
        else:
            st.metric(label="Current Price", value="N/A")
            st.metric(label=f"Short SMA ({short_window}m)", value="N/A")

    with col2:
        if not df.empty:
            if 'SMA_long' in df.columns and not pd.isna(df['SMA_long'].iloc[-1]):
                st.metric(label=f"Long SMA ({long_window}m)", value=f"${df['SMA_long'].iloc[-1]:.2f}")
            else:
                st.metric(label=f"Long SMA ({long_window}m)", value="N/A")
            st.metric(label="Signal", value=signal)
        else:
            st.metric(label=f"Long SMA ({long_window}m)", value="N/A")
            st.metric(label="Signal", value="N/A")

    # Plot candlestick chart with SMAs
    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlesticks"
        ))
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
