from data_loader import load_all_stock_data

data = load_all_stock_data('all_stocks_5yr.csv')
symbol = 'AAPL'
df = data[symbol]

# Example: Simple Moving Average Crossover
fast = df['close'].rolling(window=10).mean()
slow = df['close'].rolling(window=50).mean()
signals = (fast > slow).astype(int)  # 1 if fast > slow

print(signals.tail())