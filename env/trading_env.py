from gym_trading_env.environments import TradingEnv
import pandas as pd
import pandas_ta as ta
import numpy as np

def create_trading_env(csv_path, use_indicators=True):

    try:
        df = pd.read_csv(csv_path)
    except:
        df = pd.read_csv(csv_path, skiprows=[0, 1])

    column_mapping = {
        'Close': 'close',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume',
        'Adj Close': 'adj_close'
    }
    
    df.columns = df.columns.str.strip()  # Remove any whitespace
    df = df.rename(columns=lambda x: column_mapping.get(x, x.lower()))

    numeric_columns = ['close', 'open', 'high', 'low', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=numeric_columns)

    df['close_feature'] = df['close']
    df['volume_feature'] = df['volume']
    df['high_feature'] = df['high']
    df['low_feature'] = df['low']
    df['daily_return'] = df['close'].pct_change(fill_method=None).fillna(0)
    df['volatility'] = df['daily_return'].rolling(window=20, min_periods=1).std().fillna(0)

    if use_indicators:
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['MACD'] = ta.macd(df['close'])['MACD_12_26_9']
        df['SMA'] = ta.sma(df['close'], length=10)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['OBV'] = ta.obv(df['close'], df['volume'])

        df['returns_5d'] = df['close'].pct_change(periods=5, fill_method=None).fillna(0)
        df['returns_10d'] = df['close'].pct_change(periods=10, fill_method=None).fillna(0)
        
        df = df.dropna().reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    initial_price = df['close'].iloc[0]
    df['market_return'] = (df['close'] / initial_price - 1) * 100

    return TradingEnv(
        df=df,
        positions=[-1, 0, 1],  # short, hold, long
        trading_fees=0.001,    # 0.1% trading fee
        borrow_interest_rate=0.0001  # 0.01% borrow interest rate
    )
