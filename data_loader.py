import pandas as pd

def load_all_stock_data(csv_path='all_stocks_5yr.csv'):
    """Loads and returns a dictionary of DataFrames, one for each symbol."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    data = {}
    for symbol in df['Name'].unique():
        subdf = df[df['Name'] == symbol].copy()
        subdf = subdf.set_index('date')
        subdf = subdf.drop(columns=['Name'])
        data[symbol] = subdf
    return data