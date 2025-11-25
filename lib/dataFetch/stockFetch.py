import yfinance as yf
import pandas as pd
from datetime import datetime


class StockFetcher:
    """
    A class to fetch comprehensive stock data using yfinance.
    """
    
    def __init__(self):
        """Initialize the StockFetcher."""
        self.valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', 
                               '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    def fetch(self, tickers, startDate='1970-01-01', interval='1d', include_all=True):
        """
        Fetch historical stock data for the given ticker symbol(s).
        
        Args:
            tickers (list): List of stock ticker symbols (e.g., ['NVDA', 'TSLA']).
            startDate (str): Starting date for historical data in 'YYYY-MM-DD' format (default: '1970-01-01').
            interval (str): Time interval between data points (default: '1d').
                           Options: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', 
                                    '1d', '5d', '1wk', '1mo', '3mo'.
                           Note: Minute intervals ('1m'-'90m') are only available for the last 7-60 days.
            include_all (bool): If True, returns a dict with all available data types (default: True).
                               If False, returns only the historical price DataFrame.
        
        Returns:
            If include_all=True (default):
                dict: Dictionary containing all available data:
                    - 'history': Historical price data (DataFrame)
                    - 'dividends': Dividend history per ticker (dict of DataFrames)
                    - 'splits': Stock split history per ticker (dict of DataFrames)
                    - 'financials': Annual financial statements per ticker (dict of DataFrames)
                    - 'quarterly_financials': Quarterly financial statements per ticker (dict of DataFrames)
                    - 'balance_sheet': Annual balance sheets per ticker (dict of DataFrames)
                    - 'quarterly_balance_sheet': Quarterly balance sheets per ticker (dict of DataFrames)
                    - 'cashflow': Annual cash flow statements per ticker (dict of DataFrames)
                    - 'quarterly_cashflow': Quarterly cash flow statements per ticker (dict of DataFrames)
                    - 'info': Company info/fundamentals per ticker (dict of dicts)
                    - 'recommendations': Analyst recommendations per ticker (dict of DataFrames)
                    - 'earnings': Earnings data per ticker (dict of DataFrames)
                    - 'quarterly_earnings': Quarterly earnings per ticker (dict of DataFrames)
                    - 'options': Available option expiry dates per ticker (dict of lists)
        
        Raises:
            ValueError: If tickers is not a list, is empty, or if date format is invalid
            Exception: If data download fails
        """
        # Validate input
        self._validate_tickers(tickers)
        self._validate_date(startDate)
        self._validate_interval(interval)
        
        try:
            # Download historical stock data for one or more tickers using yfinance
            # Returns a DataFrame with price/volume columns, and MultiIndex columns if multiple tickers
            stock_data = yf.download(tickers, start=startDate, interval=interval)
            
            # Check if data was successfully downloaded
            if stock_data.empty:
                raise Exception(f"No data returned for tickers: {tickers}. Check if ticker symbols are valid.")

            # If only historical data is needed, return it directly
            if not include_all:
                return stock_data
            
            # Otherwise, fetch all available data for each ticker
            all_data = {
                'history': stock_data,
                'dividends': {},
                'splits': {},
                'financials': {},
                'quarterly_financials': {},
                'balance_sheet': {},
                'quarterly_balance_sheet': {},
                'cashflow': {},
                'quarterly_cashflow': {},
                'info': {},
                'recommendations': {},
                'earnings': {},
                'quarterly_earnings': {},
                'options': {}
            }
            
            # Fetch detailed data for each ticker
            for ticker_symbol in tickers:
                ticker = yf.Ticker(ticker_symbol)
                
                try:
                    # Corporate actions
                    all_data['dividends'][ticker_symbol] = ticker.dividends
                    all_data['splits'][ticker_symbol] = ticker.splits
                    
                    # Financial statements
                    all_data['financials'][ticker_symbol] = ticker.financials
                    all_data['quarterly_financials'][ticker_symbol] = ticker.quarterly_financials
                    all_data['balance_sheet'][ticker_symbol] = ticker.balance_sheet
                    all_data['quarterly_balance_sheet'][ticker_symbol] = ticker.quarterly_balance_sheet
                    all_data['cashflow'][ticker_symbol] = ticker.cashflow
                    all_data['quarterly_cashflow'][ticker_symbol] = ticker.quarterly_cashflow
                    
                    # Fundamentals and info
                    all_data['info'][ticker_symbol] = ticker.info
                    
                    # Analyst data
                    all_data['recommendations'][ticker_symbol] = ticker.recommendations
                    all_data['earnings'][ticker_symbol] = ticker.earnings
                    all_data['quarterly_earnings'][ticker_symbol] = ticker.quarterly_earnings
                    
                    # Options data
                    all_data['options'][ticker_symbol] = list(ticker.options)
                    
                except Exception as e:
                    print(f"Warning: Could not fetch some data for {ticker_symbol}: {str(e)}")
                    continue
            
            return all_data
        
        except Exception as e:
            # Catch download errors or network issues
            raise Exception(f"Error fetching stock data: {str(e)}")
    
    def _validate_tickers(self, tickers):
        """Validate tickers parameter."""
        if not isinstance(tickers, list):
            raise ValueError("tickers must be a list of ticker symbols")
        if len(tickers) == 0:
            raise ValueError("tickers list cannot be empty")
    
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
