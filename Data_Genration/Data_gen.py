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
    # Trend Indicators
    # SMA: Simple Moving Average, helps identify trend direction
    # ML use: Can be used to create buy/sell signals or as a feature for trend prediction
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_100'] = ta.trend.sma_indicator(df['Close'], window=100)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    
    # EMA: Exponential Moving Average, gives more weight to recent prices
    # ML use: Similar to SMA, but may react faster to recent price changes
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA_63'] = ta.trend.ema_indicator(df['Close'], window=63)
    df['EMA_100'] = ta.trend.ema_indicator(df['Close'], window=100)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
    
    # MACD: Moving Average Convergence Divergence, helps identify trend changes
    # ML use: Can be used to predict trend reversals or as a feature for buy/sell decisions
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    
    # ADX: Average Directional Index, measures trend strength
    # ML use: Can help in identifying strong trends, useful for trend-following strategies
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

    # Momentum Indicators
    # RSI: Relative Strength Index, measures the speed and change of price movements
    # ML use: Can help predict overbought or oversold conditions
    df['RSI'] = ta.momentum.rsi(df['Close'])
    
    # Stochastic Oscillator: Compares a closing price to its price range over time
    # ML use: Another indicator for overbought/oversold conditions and potential reversals
    df['Stoch_Osc'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    
    # Williams %R: Measures overbought and oversold levels
    # ML use: Similar to RSI and Stochastic, can be used to predict potential price reversals
    df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])

    # Volatility Indicators
    # Bollinger Bands: Measure market volatility and overbought/oversold conditions
    # ML use: Can be used to predict potential breakouts or mean reversion
    df['BBlow'], df['BBmid'], df['BBupp'] = ta.volatility.bollinger_hband_indicator(df['Close']), ta.volatility.bollinger_mavg(df['Close']), ta.volatility.bollinger_lband_indicator(df['Close'])
    
    # ATR: Average True Range, measures market volatility
    # ML use: Can be used to set stop-loss levels or as a feature for volatility prediction
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

    # Volume Indicators
    # OBV: On-Balance Volume, relates volume to price change
    # ML use: Can be used to confirm price trends or predict potential reversals
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    # CMF: Chaikin Money Flow, measures buying and selling pressure
    # ML use: Can help in predicting potential trend reversals or continuation
    df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])

    print("Hello TECHNICAL")

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
    # Price change: Absolute change in price
    # ML use: Direct indicator of price movement, useful for regression models
    df['Price_Change'] = df['Close'].diff()
    
    # Percentage change: Relative change in price
    # ML use: Normalized price change, useful for comparing across different price scales
    df['Pct_Change'] = df['Close'].pct_change()

    # Lagged features: Past values of close price and volume
    # ML use: Allows the model to capture time-dependent patterns and trends
    for i in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)

    # Rolling statistics: Moving averages and standard deviations
    # ML use: Captures recent trends and volatility, can help in predicting future movements
    for window in [5, 10, 20]:
        df[f'Close_Roll_Mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Close_Roll_Std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Volume_Roll_Mean_{window}'] = df['Volume'].rolling(window=window).mean()

    # Relative volume: Current volume compared to recent average
    # ML use: Identifies unusual trading activity, which might precede significant price moves
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # Day of week: Captures potential day-of-week effects
    # ML use: Some stocks might have patterns related to the day of the week
    df['Day_of_Week'] = df['Date'].dt.dayofweek

    # Is month end: Captures potential end-of-month effects
    # ML use: Some stocks might have patterns related to the end of the month (e.g., due to rebalancing)
    df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)

    # VWAP: Volume-Weighted Average Price
    # ML use: Provides a benchmark for intraday trades, can be used to identify trend strength
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # High Volume: Identifies days with unusually high trading volume
    # ML use: Can indicate important events or significant market interest
    volume_mean = df['Volume'].rolling(window=20).mean()
    volume_std = df['Volume'].rolling(window=20).std()
    df['High_Volume'] = (df['Volume'] > (volume_mean + 2 * volume_std)).astype(int)

    # Significant Price Move: Identifies days with unusually large price changes
    # ML use: Can indicate important events or significant shifts in market sentiment
    returns = df['Close'].pct_change()
    returns_mean = returns.rolling(window=20).mean()
    returns_std = returns.rolling(window=20).std()
    df['Significant_Price_Move'] = (abs(returns - returns_mean) > (2 * returns_std)).astype(int)

    # Volume Spike with Price Move: Identifies high volume days with significant price changes
    # ML use: Can indicate particularly important market events or major shifts in supply/demand
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