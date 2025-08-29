import pandas as pd
import alpaca_trade_api as tradeapi

class BrokerAPI:
    def __init__(self, config):
        # config: dict with keys 'key', 'secret', 'endpoint'
        self.api = tradeapi.REST(
            config['key'],
            config['secret'],
            config.get('endpoint', 'https://paper-api.alpaca.markets')
        )

    def get_historical_bars(self, symbol, period, interval):
        # period (e.g. '1mo'), interval ('1h', '1d')
        # Convert to Alpaca's timeframes
        tf_map = {'1h': '1Hour', '1d': '1Day'}
        tf = tf_map.get(interval, '1Hour')
        # Alpaca expects bars limit, so estimate
        limit = 720 if interval=='1h' else 30
        bars = self.api.get_bars(symbol, tf, limit=limit).df
        bars = bars.reset_index()
        bars = bars.rename(columns={
            'timestamp': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        bars = bars.set_index('datetime')
        return bars[['open', 'high', 'low', 'close', 'volume']]

    def get_recent_bars(self, symbol, N=100):
        bars = self.api.get_bars(symbol, '1Min', limit=N).df
        bars = bars.reset_index()
        bars = bars.rename(columns={
            'timestamp': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        bars = bars.set_index('datetime')
        return bars[['open', 'high', 'low', 'close', 'volume']]

    def execute_order(self, symbol, side, size):
        # side: 'BUY' or 'SELL'
        if size < 1:
            print(f"Order size < 1 for {symbol}, not placing order.")
            return
        try:
            side = 'buy' if side.upper() == 'BUY' else 'sell'
            self.api.submit_order(
                symbol=symbol,
                qty=size,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            print(f"Placed {side.upper()} order for {symbol}, qty={size}")
        except Exception as e:
            print(f"Order error: {e}")