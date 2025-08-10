📈 Jerin's Live Moving Average Crossover Dashboard
A real-time stock market analysis dashboard built with Python, Streamlit, Plotly, and Yahoo Finance (yfinance).
The app calculates short-term and long-term Simple Moving Averages (SMAs), identifies BUY / SELL / HOLD signals based on crossovers, and displays an interactive candlestick chart.
Live Demo: [Add your Streamlit Cloud or deployed link here]
Portfolio Page: https://jerinpaul.com/projects

🚀 Features
Live Market Data
Fetches minute-by-minute price data for any ticker symbol using the Yahoo Finance API.
SMA Crossover Signals
Calculates short and long SMAs and detects crossover points to generate trading signals:
BUY – Short SMA crosses above Long SMA
SELL – Short SMA crosses below Long SMA
HOLD – No crossover
Interactive Visualization
Fully interactive candlestick charts with overlayed SMA lines, built using Plotly.
User Controls
Choose any stock ticker (e.g., AAPL, TSLA, MSFT)
Set short and long SMA periods (in minutes)
Auto-Refresh
Automatically refreshes every 60 seconds to keep market data up to date.
Market-Aware
Detects when the market is closed and shows the last available data instead of stale or empty values.
Error Handling
Includes safeguards for invalid SMA settings (e.g., short ≥ long) and insufficient data.
🖥️ Technologies Used
Python 3.10+
Streamlit – Web app framework
yFinance – Stock market data
Pandas – Data manipulation
Plotly – Interactive charting
Yahoo Finance API – Real-time price data
📷 Screenshots
(Add screenshots here — candlestick chart + metrics section)
⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/live-sma-dashboard.git
cd live-sma-dashboard
2️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run the app
streamlit run app.py
📄 Usage
Enter a valid stock ticker in the sidebar (e.g., AAPL, TSLA).
Set your Short SMA Window and Long SMA Window (in minutes).
View real-time chart updates every 60 seconds.
Interpret signals:
BUY – Potential upward trend starting
SELL – Potential downward trend starting
HOLD – No clear trend change
📊 Example Signals
Short SMA	Long SMA	Signal
Crosses above	—	BUY
Crosses below	—	SELL
—	—	HOLD
📌 Roadmap
Planned enhancements:
Add Exponential Moving Averages (EMA)
Include RSI & MACD indicators
Backtesting with historical data
Strategy performance metrics (Sharpe ratio, win rate, etc.)
Deploy to Streamlit Cloud or custom server
📜 License
This project is licensed under the MIT License – feel free to use and modify.
👨‍💻 Author
Jerin Paul – Portfolio | LinkedIn
