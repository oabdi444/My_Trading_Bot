import numpy as np
import pandas as pd

def calculate_rsi(close, window):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window - 1, adjust=False).mean()
    ema_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic(df, k_window, d_window):
    low_min = df['low'].rolling(window=k_window).min()
    high_max = df['high'].rolling(window=k_window).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_window).mean()
    return k, d

def calculate_williams_r(df, window):
    high_max = df['high'].rolling(window).max()
    low_min = df['low'].rolling(window).min()
    wr = -100 * (high_max - df['close']) / (high_max - low_min)
    return wr

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def calculate_ad_line(df):
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * df['volume']
    adl = mfv.cumsum()
    return adl

def calculate_cmf(df, window):
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * df['volume']
    cmf = mfv.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
    return cmf

def calculate_atr(df, window):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_support_resistance(df):
    window = 20
    support = df['low'].rolling(window).min()
    resistance = df['high'].rolling(window).max()
    return support, resistance