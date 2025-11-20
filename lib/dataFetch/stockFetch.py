import yfinance as yf
import pandas as pd

def stockFetch(tickers, startDate='1970-01-01', interval='1d'):
    """
    Fetch historical stock data for the given ticker symbol(s).
    
    Args:
        tickers (list): List of stock ticker symbols (e.g., ['NVDA', 'TSLA']).
        startDate (str): Starting date for historical data in 'YYYY-MM-DD' format (default: '1970-01-01').
        interval (str): Time interval between data points (default: '1d').
                       Options: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', 
                                '1d', '5d', '1wk', '1mo', '3mo'.
                       Note: Minute intervals ('1m'-'90m') are only available for the last 7-60 days.
    
    Returns:
        pandas.DataFrame: Historical stock data with columns like Open, High, Low, Close, Volume, etc.
                         If multiple tickers provided, columns will be MultiIndex with (metric, ticker).
    
    Raises:
        ValueError: If tickers is not a list, is empty, or if date format is invalid
        Exception: If data download fails
    """
    # Validate input
    if not isinstance(tickers, list):
        raise ValueError("tickers must be a list of ticker symbols")
    if len(tickers) == 0:
        raise ValueError("tickers list cannot be empty")
    
    # Validate date format
    try:
        from datetime import datetime
        datetime.strptime(startDate, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"startDate must be in 'YYYY-MM-DD' format, got: {startDate}")
    
    # Validate interval
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    if interval not in valid_intervals:
        raise ValueError(f"interval must be one of {valid_intervals}, got: {interval}")
    
    try:
        # Download historical stock data for one or more tickers using yfinance
        # Returns a DataFrame with price/volume columns, and MultiIndex columns if multiple tickers
        stock_data = yf.download(tickers, start=startDate, interval=interval)
        
        # Check if data was successfully downloaded
        if stock_data.empty:
            raise Exception(f"No data returned for tickers: {tickers}. Check if ticker symbols are valid.")

        # Return the DataFrame for further analysis or modeling
        return stock_data
    
    except Exception as e:
        # Catch download errors or network issues
        raise Exception(f"Error fetching stock data: {str(e)}")
