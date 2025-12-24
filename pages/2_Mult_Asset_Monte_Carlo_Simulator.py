import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
num_tickers = st.sidebar.number_input("Number of stock ticker symbols", min_value=1, step=1)
tickers = []
for i in range(num_tickers):
    tick = False
    while tick == False:
        ticker = st.sidebar.text_input(f"Ticker {i+1}", "").upper()
        if ticker in tickers:
            st.warning("Ticker already used please enter another one!")
        else:
            tick = True
            if ticker:
                tickers.append(ticker)
weights = []
if len(tickers) <= 1:
    weights = [1.0]
else:
    st.sidebar.subheader("Portfolio Weights (%)")
    for ticker in tickers:
        w = st.sidebar.number_input(f"Weight for {ticker}", value=100/len(tickers), min_value=0.0, max_value=100.0, step=1.0)
        weights.append(w / 100)

n_simulations = st.sidebar.number_input("Number of Monte Carlo simulations", min_value=1, value=10, step=1)
days = st.sidebar.number_input("Number of trading days to simulate", min_value=1, value=252, step=1)
if st.sidebar.button("Run Simulation"):
    if len(tickers) == 0:
        st.error("Please enter at least one ticker.")
    else:
        weights = np.array(weights)
        if not np.isclose(weights.sum(), 1.0):
            st.error(f"❌ Weights add up to {weights.sum()*100:.2f}. They must equal 100.")
            st.stop()
        # Download data
        data = yf.download(tickers, period="1y", auto_adjust=False)
        returns = data["Close"].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        initial_price = data["Close"].iloc[-1].values
        n_stocks = len(tickers)
        prices = np.zeros((days, n_simulations, n_stocks))
        prices[0] = initial_price
        cov_matrix = returns.cov().values
        # Monte Carlo simulation
        for sim in range(n_simulations):
            for t in range(1, days):
                z = np.random.multivariate_normal(np.zeros(n_stocks), cov_matrix)
                prices[t, sim, :] = prices[t - 1, sim, :] * np.exp((mu - 0.5 * sigma ** 2) + sigma * z)
        portfolio_prices = np.sum(prices * weights, axis=2)
        portfolio_history = (data["Close"] * weights).sum(axis=1)
        past_dates = data.index
        future_dates = pd.bdate_range(start=past_dates[-1] + pd.Timedelta(days=1), periods=days)
        # --- Plotly Interactive Plot ---
        fig = go.Figure()
        for i, ticker in enumerate(tickers):
            for sim in range(min(10, n_simulations)):
                fig.add_trace(go.Scatter(x=future_dates, y=prices[:, sim, i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
            fig.add_trace(go.Scatter(x=future_dates, y=prices[:, :, i].mean(axis=1), mode='lines', name=f'{ticker} mean', line=dict(width=2)))
            fig.add_trace(go.Scatter(x=past_dates, y=data['Close'][tickers[i]], mode='lines', name=f'{ticker} historical', line=dict(dash='dot', width=2)))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=np.percentile(portfolio_prices, 5, axis=1),
            fill=None,
            mode='lines',
            line_color='lightgrey',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=np.percentile(portfolio_prices, 95, axis=1),
            fill='tonexty',
            mode='lines',
            line_color='lightgrey',
            name='5th-95th percentile'
        ))
        fig.add_trace(go.Scatter(x=past_dates, y=portfolio_history, mode='lines', name='Historical Portfolio', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=portfolio_prices.mean(axis=1), mode='lines', name='Portfolio Mean', line=dict(color='red', width=2)))
        fig.update_layout(title="Monte Carlo Simulation", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # ---- SUMMARY SECTION ----
        st.subheader("📊 Simulation Summary")
        # Individual Ticker Summaries
        for i, ticker in enumerate(tickers):
            current_price = data["Close"][ticker].iloc[-1]
            stock_final = prices[-1, :, i]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"{ticker} — Current Price", f"${current_price:.2f}")
            col2.metric(f"{ticker} — Mean Final Price", f"${np.mean(stock_final):.2f}")
            col3.metric(f"{ticker} — 5th %ile", f"${np.percentile(stock_final, 5):.2f}")
            col4.metric(f"{ticker} — 95th %ile", f"${np.percentile(stock_final, 95):.2f}")
            st.write("---")
        # Portfolio Summary
        final_prices = portfolio_prices[-1, :]
        portfolio_current = np.sum(data["Close"].iloc[-1].values * np.array(weights))
        st.subheader("📁 Portfolio Summary")
        if len(tickers) > 1:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Portfolio Value", f"${portfolio_current:.2f}")
            col2.metric("Mean Final Value", f"${np.mean(final_prices):.2f}")
            col3.metric("5th Percentile", f"${np.percentile(final_prices, 5):.2f}")
            col4.metric("95th Percentile", f"${np.percentile(final_prices, 95):.2f}")
    
        # Annualized metrics
        annual_return = (np.mean(final_prices) / portfolio_current - 1)
        annual_volatility = np.std(final_prices / portfolio_current)
        colA, colB,colC = st.columns(3)
        colA.metric("Expected Annual Return", f"{annual_return*100:.2f}%")
        colB.metric("Simulated Annual Volatility", f"{annual_volatility*100:.2f}%")
        colC.metric("Median Final Value", f"${np.median(final_prices):.2f}")