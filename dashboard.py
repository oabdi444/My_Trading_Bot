import streamlit as st
from main import UltraAITradingBot

st.title("UltraAI Trading Bot Dashboard")

symbols = [s.strip() for s in st.text_input("Symbols (comma separated)", "AAPL,MSFT,TSLA").split(",")]
period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)
interval = st.selectbox("Interval", ["1h", "1d"], index=0)

bot = UltraAITradingBot(symbols=symbols)
st.write("Fetching data and running backtest...")

data = {symbol: bot.fetch_data(symbol, period=period, interval=interval) for symbol in symbols}
equity_curve = bot.backtest_multiasset(data)

if len(equity_curve) > 0:
    st.line_chart(equity_curve)
else:
    st.write("No equity curve data to display.")

st.write("Performance Metrics", bot.performance_metrics)
st.write("Recent Trades", bot.trade_log)
st.write("Portfolio", bot.portfolio)
st.write("Cash", bot.cash)