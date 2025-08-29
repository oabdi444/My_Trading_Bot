import pandas as pd
import numpy as np
from indicators import *
from signals import SignalEngine
from risk import calculate_position_size, get_risk_management_levels
from ml import train_ml
from broker import BrokerAPI  
from datetime import datetime
import yfinance as yf
from collections import deque
import time

class UltraAITradingBot:
    def __init__(self, account_balance: float = 10000, symbols=None, live=False, broker_cfg=None):
        self.live = live
        self.signal_weights = {
            'trend_following': 0.25,
            'mean_reversion': 0.15,
            'momentum': 0.20,
            'volume_analysis': 0.15,
            'volatility': 0.10,
            'ml_prediction': 0.15
        }
        self.confidence_threshold = 0.7
        self.min_confluence = 3
        self.signal_history = deque(maxlen=1000)
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        self.trade_returns = []
        self.max_position_size = 0.05
        self.stop_loss_pct = 0.015
        self.take_profit_pct = 0.03
        self.cash = account_balance
        self.portfolio = {}  # symbol: {position, avg_entry}
        self.trade_log = []
        self.commission_pct = 0.0005
        self.symbols = symbols if symbols else []
        self.ml_model = None
        self.signal_engine = SignalEngine(self.signal_weights, self.confidence_threshold, self.min_confluence)
        self.broker = BrokerAPI(broker_cfg) if self.live else None

    def fetch_data(self, symbol, period="6mo", interval="1h"):
        if self.live:
            return self.broker.get_historical_bars(symbol, period, interval)
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]

    def _calculate_comprehensive_indicators(self, df):
        df = df.copy()
        # Trend
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        # Volatility
        df['atr'] = calculate_atr(df, window=14)
        df['volatility'] = df['close'].rolling(20).std()
        # Oscillators
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 14, 3)
        df['williams_r'] = calculate_williams_r(df, 14)
        # Volume
        df['obv'] = calculate_obv(df)
        df['ad_line'] = calculate_ad_line(df)
        df['cmf'] = calculate_cmf(df, 20)
        # Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        # Support/Resistance
        df['support'], df['resistance'] = calculate_support_resistance(df)
        return df

    def backtest_multiasset(self, data, ml_pred=None, sentiment=None, tf_consensus=None):
        self.cash = 10000
        self.portfolio = {symbol: {"position": 0, "avg_entry": 0} for symbol in self.symbols}
        self.trade_log = []
        self.trade_returns = []
        self.performance_metrics = {
            'total_trades': 0, 'winning_trades': 0, 'total_return': 0.0,
            'max_drawdown': 0.0, 'sharpe_ratio': 0.0
        }
        min_len = min([len(df) for df in data.values()])
        index = list(data.values())[0].index[-min_len:]
        equity_curve = []
        for i in range(50, min_len):
            for symbol in self.symbols:
                dfi = self._calculate_comprehensive_indicators(data[symbol].iloc[:i+1])
                latest = dfi.iloc[-1]
                prev = dfi.iloc[-2] if len(dfi) > 1 else latest
                # Example: ML prediction (dummy here)
                mlp = ml_pred[symbol][i] if ml_pred and symbol in ml_pred else (0.5, 0.5)
                sentiment_val = sentiment[symbol][i] if sentiment and symbol in sentiment else None
                tf_cons = tf_consensus[symbol][i] if tf_consensus and symbol in tf_consensus else None
                signal, confidence, reasons = self.signal_engine.generate_enhanced_signal(
                    dfi, latest, prev, "TRENDING", mlp, sentiment_val, tf_cons
                )
                price = latest['close']
                self.execute_signal(symbol, signal, price, confidence)
            total_equity = self.cash + sum(self.portfolio[sym]["position"] * data[sym]['close'].iloc[i] for sym in self.symbols)
            equity_curve.append(total_equity)
        return pd.Series(equity_curve, index=index[50:])

    def execute_signal(self, symbol, signal, price, confidence):
        pos = self.portfolio.get(symbol, {"position": 0, "avg_entry": 0})
        size = int(calculate_position_size(self.cash, confidence, volatility=0.01) * self.cash / price)
        commission = size * price * self.commission_pct
        if self.live and self.broker:
            # Live trading branch
            self.broker.execute_order(symbol, signal, size)
        else:
            # Backtest simulation
            if signal == 'BUY' and size > 0:
                if self.cash >= size * price + commission:
                    prev_pos = pos["position"]
                    prev_avg = pos["avg_entry"]
                    new_pos = prev_pos + size
                    new_avg = (prev_pos * prev_avg + size * price) / max(new_pos, 1)
                    self.portfolio[symbol] = {"position": new_pos, "avg_entry": new_avg}
                    self.cash -= size * price + commission
                    self.trade_log.append((datetime.now(), symbol, 'BUY', size, price, commission))
            elif signal == 'SELL':
                if pos["position"] > 0:
                    sell_size = pos["position"]
                    self.cash += sell_size * price - sell_size * price * self.commission_pct
                    profit = (price - pos["avg_entry"]) * sell_size - sell_size * price * self.commission_pct
                    self.trade_returns.append(profit / (pos["avg_entry"] * sell_size))
                    self.performance_metrics['total_trades'] += 1
                    if profit > 0:
                        self.performance_metrics['winning_trades'] += 1
                    self.performance_metrics['total_return'] += profit
                    self.trade_log.append((datetime.now(), symbol, 'SELL', sell_size, price, sell_size * price * self.commission_pct))
                    self.portfolio[symbol] = {"position": 0, "avg_entry": 0}

    def live_trading_loop(self, poll_interval=60):
        if not self.broker:
            print("No broker configured for live trading!")
            return
        print("Starting live trading loop...")
        while True:
            for symbol in self.symbols:
                bars = self.broker.get_recent_bars(symbol, N=100)
                dfi = self._calculate_comprehensive_indicators(bars)
                latest = dfi.iloc[-1]
                prev = dfi.iloc[-2] if len(dfi) > 1 else latest
                mlp = (0.5, 0.5)  # Replace with actual ML prediction
                signal, confidence, reasons = self.signal_engine.generate_enhanced_signal(
                    dfi, latest, prev, "LIVE", mlp, None, None
                )
                price = latest['close']
                # Execute live order
                self.execute_signal(symbol, signal, price, confidence)
            time.sleep(poll_interval)

    def plot_equity_curve(self, equity_curve):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        equity_curve.plot(label='Equity Curve')
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.title("Backtest Equity Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run_backtest(self):
        symbols = self.symbols if self.symbols else ["AAPL", "MSFT", "TSLA"]
        self.symbols = symbols
        data = {symbol: self.fetch_data(symbol, period="3mo", interval="1h") for symbol in symbols}
        # Example: dummy ML predictions
        preds = {symbol: [(0.5, 0.5)]*len(data[symbol]) for symbol in symbols}
        equity_curve = self.backtest_multiasset(data, ml_pred=preds)
        self.plot_equity_curve(equity_curve)
        print("Final Portfolio Value:", equity_curve.iloc[-1])

if __name__ == "__main__":
    alpaca_cfg = {
        'key': 'YOUR_ALPACA_API_KEY',
        'secret': 'YOUR_ALPACA_SECRET_KEY',
        'endpoint': 'https://paper-api.alpaca.markets'  # Use this for paper trading
    }
    bot = UltraAITradingBot(
        symbols=["AAPL", "MSFT"],  # Add your desired symbols here
        live=True,
        broker_cfg=alpaca_cfg
    )
    bot.live_trading_loop(poll_interval=60)  # Checks every 60 seconds