from gym_trading_env.environments import TradingEnv
import pandas as pd
import pandas_ta as ta

def create_trading_env(csv_path, use_indicators=True):

    df = pd.read_csv(csv_path, skiprows=[1, 2])
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)

    # Clean up column names
    column_mapping = {
        'Close': 'close',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    }
    df = df.rename(columns=column_mapping)

    df['close_feature'] = df['close']
    df['volume_feature'] = df['volume']
    df['high_feature'] = df['high']
    df['low_feature'] = df['low']

    if use_indicators:
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['MACD'] = ta.macd(df['close'])['MACD_12_26_9']
        df['SMA'] = ta.sma(df['close'], length=10)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['OBV'] = ta.obv(df['close'], df['volume'])


        df = df.dropna().reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return TradingEnv(
        df=df,
        positions=[-1, 0, 1],
        trading_fees=0.001,
        borrow_interest_rate=0.0001
    )
