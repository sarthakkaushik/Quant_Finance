# !pip install yfinance ta

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import ta

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetches stock data for a given symbol between start and end dates.

    Args:
        symbol (str): The stock symbol.
        start_date (datetime.datetime): The start date.
        end_date (datetime.datetime): The end date.

    Returns:
        pandas.DataFrame: The stock data.
    """
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df.reset_index()


def add_technical_indicators(df):
    """
    Adds technical indicators to a given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to which the indicators are added.

    Returns:
        pandas.DataFrame: The DataFrame with the added technical indicators.
    """
    # Define technical indicator parameters
    windows = [20, 50, 100, 200]
    ema_windows = [20, 50, 63, 100, 200]

    # Trend Indicators
    df['SMA'] = df['Close'].rolling(window=windows[0]).mean()
    df['EMA'] = df['Close'].ewm(span=ema_windows[0], adjust=False).mean()

    for window in windows[1:]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

    for window in ema_windows[1:]:
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

    df['MACD'] = df['Close'].ewm(span=63, adjust=False).mean() - df['Close'].ewm(span=126, adjust=False).mean()
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

    # Momentum Indicators
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['Stoch_Osc'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])

    # Volatility Indicators
    df['BBlow'], df['BBmid'], df['BBupp'] = ta.volatility.bollinger_hband_indicator(df['Close']), ta.volatility.bollinger_mavg(df['Close']), ta.volatility.bollinger_lband_indicator(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

    # Volume Indicators
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])

    return df


def add_derived_features(df):
    """
    Adds derived features to a given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to which the features are added.

    Returns:
        pandas.DataFrame: The DataFrame with the added derived features.

    Description:
        This function adds several derived features to the input DataFrame. The features include:
        - Price change and percentage change: The difference and percentage change in the 'Close' column.
        - Lagged features: Lagged versions of the 'Close' and 'Volume' columns.
        - Rolling statistics: Rolling mean and standard deviation of the 'Close' and 'Volume' columns.
        - Relative volume: The ratio of the 'Volume' column to the rolling mean of the 'Volume' column.
        - Day of week and is month end: The day of the week and a boolean indicating if it is the end of the month.
        - VWAP and high volume with significant price move: The volume-weighted average price, a boolean indicating if the volume is above a certain threshold, and a boolean indicating if there is a significant price move.

        The function returns the modified DataFrame with the added features.
    """
    """
    Adds derived features to a given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to which the features are added.

    Returns:
        pandas.DataFrame: The DataFrame with the added derived features.

    Description:
        This function adds several derived features to the input DataFrame. The features include:
        - Price change and percentage change: The difference and percentage change in the 'Close' column.
        - Lagged features: Lagged versions of the 'Close' and 'Volume' columns.
        - Rolling statistics: Rolling mean and standard deviation of the 'Close' and 'Volume' columns.
        - Relative volume: The ratio of the 'Volume' column to the rolling mean of the 'Volume' column.
        - Day of week and is month end: The day of the week and a boolean indicating if it is the end of the month.
        - VWAP and high volume with significant price move: The volume-weighted average price, a boolean indicating if the volume is above a certain threshold, and a boolean indicating if there is a significant price move.

        The function returns the modified DataFrame with the added features.
    """
    # Price change and percentage change
    df['Price_Change'] = df['Close'].diff()
    df['Pct_Change'] = df['Close'].pct_change()

    # Lagged features
    lags = range(1, 11)
    for lag in lags:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

    # Rolling statistics
    windows = [5, 10, 20]
    for window in windows:
        for column in ['Close', 'Volume']:
            df[f'{column}_Roll_Mean_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_Roll_Std_{window}'] = df[column].rolling(window=window).std()

    # Relative volume
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # Day of week and is month end
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)

    # VWAP and high volume with significant price move
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    volume_stats = df['Volume'].rolling(window=20)
    df['High_Volume'] = (df['Volume'] > (volume_stats.mean() + 2 * volume_stats.std())).astype(int)
    returns = df['Close'].pct_change()
    returns_stats = returns.rolling(window=20)
    df['Significant_Price_Move'] = (abs(returns - returns_stats.mean()) > (2 * returns_stats.std())).astype(int)
    df['Volume_Spike_With_Price_Move'] = (df['High_Volume'] & df['Significant_Price_Move']).astype(int)

    return df

if __name__ == "__main__":
    symbol = "RELIANCE.NS"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)

    df = fetch_stock_data(symbol, start_date, end_date)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    df = add_technical_indicators(df)
    df = add_derived_features(df)
    print(df.head())