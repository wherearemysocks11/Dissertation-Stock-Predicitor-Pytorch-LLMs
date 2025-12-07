import yfinance as yf
import pandas as pd
from datetime import datetime


class StockFetcher:
    """
    A class to fetch historical stock data using yfinance.
    """
    
    def __init__(self):
        """Initialize the StockFetcher."""
        self.valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', 
                               '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    def fetch(self, ticker, startDate='1970-01-01', interval='1d'):
        """
        Fetch historical stock data for the given ticker symbol.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'NVDA', 'TSLA').
            startDate (str): Starting date for historical data in 'YYYY-MM-DD' format (default: '1970-01-01', api handles the rest if the data starts later).
            interval (str): Time interval between data points (default: '1d').
                           Options: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', 
                                    '1d', '5d', '1wk', '1mo', '3mo'.
                           Note: Minute intervals ('1m'-'90m') are only available for the last 7-60 days.
        
        Returns:
            pandas.DataFrame: Historical stock data with columns: Open, High, Low, Close, Volume, etc.
        
        Raises:
            ValueError: If ticker is not a string, is empty, or if date format is invalid
            Exception: If data download fails
        """
        # Validate input
        self._validate_ticker(ticker)
        self._validate_date(startDate)
        self._validate_interval(interval)
        
        try:
            # Download historical stock data using yfinance
            stock_data = yf.download(ticker, start=startDate, interval=interval)
            
            # Check if data was successfully downloaded
            if stock_data.empty:
                raise Exception(f"No data returned for ticker: {ticker}. Check if ticker symbol is valid.")

            # Return the DataFrame for further analysis or modeling
            return stock_data
        
        except Exception as e:
            # Catch download errors or network issues
            raise Exception(f"Error fetching stock data: {str(e)}")
    
    def _validate_ticker(self, ticker):
        """Validate ticker parameter."""
        if not isinstance(ticker, str):
            raise ValueError("ticker must be a string")
        if len(ticker) == 0:
            raise ValueError("ticker cannot be empty")
    
    def _validate_date(self, startDate):
        """Validate date format."""
        try:
            datetime.strptime(startDate, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"startDate must be in 'YYYY-MM-DD' format, got: {startDate}")
    
    def _validate_interval(self, interval):
        """Validate interval parameter."""
        if interval not in self.valid_intervals:
            raise ValueError(f"interval must be one of {self.valid_intervals}, got: {interval}")
