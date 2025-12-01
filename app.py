import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----------------------------- PAGE SETUP -----------------------------
st.set_page_config(page_title="Jerin's Financial Dashboard", layout="wide")

# ----------------------------- FRONT PAGE -----------------------------
st.title("📊 Jerin's Financial Dashboard")
app_choice = st.radio("Select an App", ["Home", "Live SMA Dashboard", "Multi-Asset Monte Carlo Simulator"])

# ----------------------------- HOME -----------------------------
if app_choice == "Home":
    st.write("""
    Welcome! Choose an app from above:

    - **Live SMA Dashboard**: View live moving averages and trading signals.
    - **Multi-Asset Monte Carlo Simulator**: Run portfolio simulations using Monte Carlo.
    """)
    if st.button("📂 View All My Projects"):
        st.markdown('<meta http-equiv="refresh" content="0; url=https://jerinpaul.com/projects">', unsafe_allow_html=True)

# ----------------------------- LIVE SMA DASHBOARD -----------------------------
elif app_choice == "Live SMA Dashboard":
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

# ----------------------------- MULTI-ASSET MONTE CARLO SIMULATOR -----------------------------
elif app_choice == "Multi-Asset Monte Carlo Simulator":
    st.title("📈 Multi-Asset Monte Carlo Simulator")

    # Sidebar for inputs
    st.sidebar.subheader("Simulation Settings")
    num_tickers = st.sidebar.number_input("Number of Tickers", min_value=1, max_value=5, value=2, step=1)
    tickers = []
    weights = []

    # Input tickers and weights dynamically
    for i in range(num_tickers):
        tick = st.sidebar.text_input(f"Ticker {i+1}", value="AAPL").upper()
        tickers.append(tick)

    st.sidebar.markdown("### Portfolio Weights (%)")
    for i in range(num_tickers):
        w = st.sidebar.number_input(f"Weight {tickers[i]}", min_value=0.0, max_value=100.0, value=round(100/num_tickers, 2))
        weights.append(w / 100)

    if sum(weights) != 1:
        st.sidebar.warning("Weights should sum to 100%.")

    n_simulations = st.sidebar.number_input("Number of Simulations", min_value=10, max_value=1000, value=100, step=10)
    days = st.sidebar.number_input("Simulation Days", min_value=30, max_value=1000, value=252, step=10)

    if st.sidebar.button("Run Simulation") and sum(weights) == 1:
        with st.spinner("Fetching data and running Monte Carlo simulations..."):
            # Fetch historical data
            data = yf.download(tickers, period="1y", auto_adjust=False)
            returns = data['Close'].pct_change().dropna()
            mu = returns.mean()
            sigma = returns.std()
            initial_price = data['Close'].iloc[-1].values
            n_stocks = len(tickers)
            prices = np.zeros((days, n_simulations, n_stocks))
            prices[0] = initial_price
            dt = 1
            cov_matrix = returns.cov().values

            # Monte Carlo simulation
            for sim in range(n_simulations):
                for t in range(1, days):
                    z = np.random.multivariate_normal(np.zeros(n_stocks), cov_matrix)
                    prices[t, sim, :] = prices[t-1, sim, :] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

            weights_arr = np.array(weights)
            portfolio_prices = np.sum(prices * weights_arr, axis=2)

            # ---------------- PLOT PORTFOLIO ----------------
            future_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days)
            fig = go.Figure()

            # First 10 simulations
            for i in range(min(10, n_simulations)):
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=portfolio_prices[:, i],
                    mode='lines',
                    line=dict(color='lightgrey'),
                    showlegend=False
                ))

            # Mean portfolio
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=portfolio_prices.mean(axis=1),
                mode='lines',
                line=dict(color='red', width=2),
                name='Portfolio Mean'
            ))

            # 5th-95th percentile shading
            fig.add_trace(go.Scatter(
                x=future_dates.tolist() + future_dates[::-1].tolist(),
                y=np.percentile(portfolio_prices, 95, axis=1).tolist() + np.percentile(portfolio_prices, 5, axis=1)[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(128,128,128,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='5th-95th percentile'
            ))

            fig.update_layout(
                title="Monte Carlo Portfolio Simulation",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---------------- SUMMARY ----------------
            final_prices = portfolio_prices[-1, :]
            portfolio_current = np.sum(data["Close"].iloc[-1].values * weights_arr)
            st.subheader("Simulation Summary")
            st.write(f"Current Portfolio Value: ${portfolio_current:.2f}")
            st.write(f"Mean Final Value: ${np.mean(final_prices):.2f}")
            st.write(f"Median Final Value: ${np.median(final_prices):.2f}")
            st.write(f"5th Percentile: ${np.percentile(final_prices, 5):.2f}")
            st.write(f"95th Percentile: ${np.percentile(final_prices, 95):.2f}")
            annual_return = (np.mean(final_prices) / portfolio_current - 1)
            annual_volatility = np.std(final_prices / portfolio_current)
            st.write(f"Expected Annual Return: {annual_return*100:.2f}%")
            st.write(f"Simulated Annual Volatility: {annual_volatility*100:.2f}%")