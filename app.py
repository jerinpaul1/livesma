import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ----------------------------- PAGE SETUP -----------------------------
st.set_page_config(page_title="Jerin's Financial Dashboard", layout="wide")

# ----------------------------- FRONT PAGE -----------------------------
st.title("📊 Jerin's Financial Dashboard")
if st.button("📂 View All My Projects"):
        st.markdown('<meta http-equiv="refresh" content="0; url=https://jerinpaul.com/projects">', unsafe_allow_html=True)
st.write("""
#    Welcome! Choose an app from below:
#
#    - **Live SMA Dashboard**: View live moving averages and trading signals.
#    - **Multi-Asset Monte Carlo Simulator**: Run portfolio simulations using Monte Carlo.
#    """)
app_choice = st.radio("Select an App", ["Home", "Live SMA Dashboard", "Multi-Asset Monte Carlo Simulator"])

# ----------------------------- HOME -----------------------------

#if app_choice == "Home":
#    st.write("""
#    Welcome! Choose an app from above:
#
#    - **Live SMA Dashboard**: View live moving averages and trading signals.
#    - **Multi-Asset Monte Carlo Simulator**: Run portfolio simulations using Monte Carlo.
#    """)
#    if st.button("📂 View All My Projects"):
#        st.markdown('<meta http-equiv="refresh" content="0; url=https://jerinpaul.com/projects">', unsafe_allow_html=True)

if app_choice == "Home":
    st.markdown(
        """
        Here you can find some of the programs i've worked on while at university.
        I typically make them using Google Colab and then try to translate it into streamlit compatible pages to put here.
        If you would like to see more or even just contact me, head on over to my website using the view projects button!
        """
    )


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

    # Validate ticker
    @st.cache_data
    def validate_ticker(ticker):
        """Return True if ticker exists on Yahoo Finance."""
        try:
            dataa = yf.Ticker(ticker).history(period="1d")
            return not dataa.empty
        except Exception:
            return False
    
    # --- Sidebar Inputs ---
    st.sidebar.header("Simulation Parameters")
    
    # Number of tickers
    num_tickers = st.sidebar.number_input("Number of stock ticker symbols", min_value=1, step=1)
    
    # Tickers
    tickers = []
    for i in range(num_tickers):
        while True:
            ticker = st.sidebar.text_input(f"Ticker {i+1}", "").upper()
            if ticker == "":
                break  # skip if empty
            if ticker in tickers:
                st.sidebar.warning(f"{ticker} ❌ Already entered. Try again.")
                break
            if validate_ticker(ticker):
                st.sidebar.success(f"{ticker} ✅ Valid ticker")
                tickers.append(ticker)
                break
            else:
                st.sidebar.error(f"{ticker} ❌ Invalid or not found on Yahoo Finance. Try again.")
                break
            
    # Weights
    weights = []
    if len(tickers) == 1:
        weights = [1.0]
    else:
        st.sidebar.subheader("Portfolio Weights (%)")
        total_weight = 0
        for ticker in tickers:
            w = st.sidebar.number_input(f"Weight for {ticker}", min_value=0.0, max_value=100.0, step=1.0)
            weights.append(w / 100)
            total_weight += w
        if abs(sum(weights) - 1.0) > 1e-5:
            st.sidebar.warning("Weights do not sum to 100%. Please adjust.")
    
    # Number of simulations
    n_simulations = st.sidebar.number_input("Number of Monte Carlo simulations", min_value=1, value=100, step=1)
    
    # Number of days to simulate
    days = st.sidebar.number_input("Number of trading days to simulate", min_value=1, value=252, step=1)
    
    if st.sidebar.button("Run Simulation"):
        if len(tickers) == 0:
            st.error("Please enter at least one valid ticker.")
        else:
            # --- Download data ---
            data = yf.download(tickers, period="1y", auto_adjust=False)
            st.success("Historical data downloaded ✅")
    
            # Daily returns
            returns = data["Close"].pct_change().dropna()
            mu = returns.mean()
            sigma = returns.std()
    
            # Initial price
            initial_price = data["Close"].iloc[-1].values
            n_stocks = len(tickers)
            prices = np.zeros((days, n_simulations, n_stocks))
            prices[0] = initial_price
            dt = 1
    
            # --- Monte Carlo simulation ---
            cov_matrix = returns.cov().values
            for sim in range(n_simulations):
                for t in range(1, days):
                    z = np.random.multivariate_normal(np.zeros(n_stocks), cov_matrix)
                    prices[t, sim, :] = prices[t - 1, sim, :] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    
            # Portfolio prices
            weights = np.array(weights)
            portfolio_prices = np.sum(prices * weights, axis=2)
            portfolio_history = (data["Close"] * weights).sum(axis=1)
    
            # Date axes
            past_dates = data.index
            future_dates = pd.bdate_range(start=past_dates[-1] + pd.Timedelta(days=1), periods=days)
    
            # --- Plotting ---
            st.subheader("Monte Carlo Simulation Plot")
            fig, ax = plt.subplots(figsize=(12, 6))
            for i, ticker in enumerate(tickers):
                ax.plot(future_dates, prices[:, :10, i], alpha=0.3)
                ax.plot(future_dates, prices[:, :, i].mean(axis=1), label=f'{ticker} mean')
                ax.plot(past_dates, data['Close'][tickers[i]], label=f'{tickers[i]} historical')
    
            ax.fill_between(future_dates,
                            np.percentile(portfolio_prices, 5, axis=1),
                            np.percentile(portfolio_prices, 95, axis=1),
                            color='grey', alpha=0.2, label='5th-95th percentile')
    
            ax.plot(past_dates, portfolio_history, color='black', label="Historical Portfolio Value", linewidth=2)
            ax.plot(future_dates, portfolio_prices.mean(axis=1), lw=2, color='red', label='Portfolio mean')
            ax.set_title(f"Monte Carlo Simulation ({n_simulations} Paths)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
            # --- Summary ---
            st.subheader("Simulation Summary")
            for i, ticker in enumerate(tickers):
                current_price = data["Close"][ticker].iloc[-1]
                stock_final = prices[-1, :, i]
                st.markdown(f"**{ticker} Summary:**")
                st.write(f"Current price: ${current_price:.2f}")
                st.write(f"Mean final price: ${np.mean(stock_final):.2f}")
                st.write(f"5th percentile: ${np.percentile(stock_final, 5):.2f}")
                st.write(f"95th percentile: ${np.percentile(stock_final, 95):.2f}")
    
            final_prices = portfolio_prices[-1, :]
            portfolio_current = np.sum(data["Close"].iloc[-1].values * weights)
            st.markdown("**Portfolio Summary:**")
            st.write(f"Current portfolio value: ${portfolio_current:.2f}")
            st.write(f"Mean final value: ${np.mean(final_prices):.2f}")
            st.write(f"Median final value: ${np.median(final_prices):.2f}")
            st.write(f"5th percentile: ${np.percentile(final_prices, 5):.2f}")
            st.write(f"95th percentile: ${np.percentile(final_prices, 95):.2f}")
            annual_return = (np.mean(final_prices) / portfolio_current - 1)
            annual_volatility = np.std(final_prices / portfolio_current)
            st.write(f"Expected Annual Return: {annual_return*100:.2f}%")
            st.write(f"Simulated Annual Volatility: {annual_volatility*100:.2f}%")
    
            # Histogram
            st.subheader("Distribution of Simulated Final Portfolio Values")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.hist(final_prices, bins=30, color="skyblue", edgecolor="black")
            ax2.set_xlabel("Portfolio Value ($)")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)