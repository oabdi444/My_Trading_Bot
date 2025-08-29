import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (headers are present)
df = pd.read_csv('all_stocks_5yr.csv')

# Use the correct column name 'Name'
stock = 'ZION'  # Change to any symbol you like

# Filter for the selected stock
df_stock = df[df['Name'] == stock].copy()
df_stock['date'] = pd.to_datetime(df_stock['date'])
df_stock = df_stock.set_index('date')
df_stock = df_stock.drop(columns=['Name'])

# Show summary and first rows
print(f"First 5 rows for {stock}:")
print(df_stock.head())
print("\nSummary statistics:")
print(df_stock.describe())

# Plot Close Price
plt.figure(figsize=(12,6))
plt.plot(df_stock.index, df_stock['close'], label=f'{stock} Close Price')
plt.title(f'{stock} Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# Plot Volume
plt.figure(figsize=(12,4))
plt.bar(df_stock.index, df_stock['volume'], width=1.0, color='gray', label='Volume')
plt.title(f'{stock} Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Plot Open/High/Low/Close on same plot
plt.figure(figsize=(12,6))
plt.plot(df_stock.index, df_stock['open'], label='Open', alpha=0.7)
plt.plot(df_stock.index, df_stock['high'], label='High', alpha=0.7)
plt.plot(df_stock.index, df_stock['low'], label='Low', alpha=0.7)
plt.plot(df_stock.index, df_stock['close'], label='Close', alpha=0.7)
plt.title(f'{stock} OHLC Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()