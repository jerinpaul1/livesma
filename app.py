import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import time # Keep time for sleep if needed for refresh logic

# SETTINGS (adjust as needed for Streamlit inputs)
# These will be replaced by Streamlit widgets

st.set_page_config(page_title="Live SMA Dashboard", layout="wide")
st.title("📈 Live Moving Average Crossover Dashboard")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
short_window = st.sidebar.number_input("Short SMA Window (minutes)", min_value=1, value=5)
long_window = st.sidebar.number_input("Long SMA Window (minutes)", min_value=1, value=15)

# Fetch data function (can be cached)
@st.cache_data(ttl=60) # Cache data for 60 seconds
def fetch_data(ticker, short_window, long_window):
    try:
        # Adjusted period and interval for potentially longer lookback if needed, but sticking to 1d for 1m interval
        df = yf.download(ticker, period="1d", interval="1m") # Use 1m interval for live
        if df.empty:
            st.warning(f"No data fetched for {ticker}. Check ticker symbol and market hours.")
            return pd.DataFrame() # Return empty if no data

        # Normalize column names: flatten MultiIndex if it exists
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Use normalized column name for Close price
        close_column = 'Close' if 'Close' in df.columns else 'Price_Close'
        if close_column not in df.columns:
             st.warning(f"'{close_column}' column not found in fetched data for {ticker}. Check data source.")
             return pd.DataFrame() # Return empty if Close column not found

        # Ensure windows are not larger than available data
        if len(df) > 0:
            short_window = min(short_window, len(df[close_column]))
            long_window = min(long_window, len(df[close_column]))
        else:
             st.warning(f"Not enough data points after column check for {ticker}.")
             return pd.DataFrame() # No data to calculate SMAs

        # Calculate SMAs using the normalized Close price column
        df['SMA_short'] = df[close_column].rolling(window=short_window).mean()
        df['SMA_long'] = df[close_column].rolling(window=long_window).mean()

        df.dropna(inplace=True) # Drop rows with NaN created by rolling window

        return df
    except Exception as e:
        st.error(f"Error fetching or processing data for {ticker}: {e}")
        return pd.DataFrame()

# Get data
df = fetch_data(ticker, short_window, long_window)

# Check if enough data exists after all processing
if df.empty or len(df) < max(short_window, long_window):
    # The warning is already displayed within fetch_data or here
    pass # Keep the existing warning logic if needed, but fetch_data now warns


else:
    # check_signal logic (adapted for Streamlit)
    # We need at least two data points after dropping NaNs for signal calculation
    if len(df) < 2:
         signal = "Waiting for enough data..."
    else:
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Robustly check for signal after ensuring valid data
        try:
            # Access SMA columns directly as they are not MultiIndex after calculation
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
             # Use normalized column name for Current Price
             close_column = 'Close' if 'Close' in df.columns else 'Price_Close'
             # Fix: Use .item() to extract scalar before formatting
             st.metric(label="Current Price", value=f"${df[close_column].iloc[-1].item():.2f}")

             # Access SMA columns directly
             if 'SMA_short' in df.columns and not pd.isna(df['SMA_short'].iloc[-1]):
                 # Fix: Use .item() to extract scalar before formatting
                st.metric(label=f"Short SMA ({short_window}m)", value=f"${df['SMA_short'].iloc[-1].item():.2f}")
             else:
                 st.metric(label=f"Short SMA ({short_window}m)", value="N/A")
        else:
             st.metric(label="Current Price", value="N/A")
             st.metric(label=f"Short SMA ({short_window}m)", value="N/A")


    with col2:
         if not df.empty:
            # Access SMA columns directly
            if 'SMA_long' in df.columns and not pd.isna(df['SMA_long'].iloc[-1]):
                 # Fix: Use .item() to extract scalar before formatting
                st.metric(label=f"Long SMA ({long_window}m)", value=f"${df['SMA_long'].iloc[-1].item():.2f}")
            else:
                st.metric(label=f"Long SMA ({long_window}m)", value="N/A")
            st.metric(label="Signal", value=signal)
         else:
            st.metric(label=f"Long SMA ({long_window}m)", value="N/A")
            st.metric(label="Signal", value="N/A")


    # Plot candlestick chart with SMAs
    if not df.empty:
        fig = go.Figure()

        # Use normalized column names for candlestick data
        open_column = 'Open' if 'Open' in df.columns else 'Price_Open'
        high_column = 'High' if 'High' in df.columns else 'Price_High'
        low_column = 'Low' if 'Low' in df.columns else 'Price_Low'
        close_column = 'Close' if 'Close' in df.columns else 'Price_Close'


        # Check if necessary columns exist after normalization
        if all(col in df.columns for col in [open_column, high_column, low_column, close_column]):
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df[open_column],
                high=df[high_column],
                low=df[low_column],
                close=df[close_column],
                name="Candlesticks"
            ))
        else:
            st.warning("Candlestick data columns not found after processing.")


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
