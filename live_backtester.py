import yfinance as yf
import pandas as pd
import time
import datetime as dt

# SETTINGS
ticker = "AAPL"
short_window = 5   # short SMA in minutes
long_window = 15   # long SMA in minutes
interval = "1m"    # intraday interval
lookback = "1d"    # only today's data

def fetch_data():
    df = yf.download(ticker, period=lookback, interval=interval)
    df = df[['Close']]
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    df.dropna(inplace=True)
    return df

def check_signal(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    if latest['SMA_short'] > latest['SMA_long'] and prev['SMA_short'] <= prev['SMA_long']:
        return "BUY"
    elif latest['SMA_short'] < latest['SMA_long'] and prev['SMA_short'] >= prev['SMA_long']:
        return "SELL"
    else:
        return "HOLD"

print(f"Starting live monitor for {ticker}... (press CTRL+C to stop)")

try:
    while True:
        df = fetch_data()
        signal = check_signal(df)
        last_price = df['Close'].iloc[-1]
        now = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Price: ${last_price:.2f} | Signal: {signal}")
        time.sleep(60)  # wait 1 minute before next update
except KeyboardInterrupt:
    print("\nStopped by user.")
