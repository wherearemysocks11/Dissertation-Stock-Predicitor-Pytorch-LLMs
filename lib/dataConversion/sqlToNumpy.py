import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path


class SQLToNumpy:
    """
    A class to convert SQL database data into NumPy arrays for PyTorch training.
    """
    
    def __init__(self, db_path='data/stock_data.db'):
        """
        Initialize the SQLToNumpy converter.
        
        Args:
            db_path (str): Path to SQLite database file (default: 'data/stock_data.db').
        """
        self.db_path = db_path
        
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.conn = sqlite3.connect(db_path)
    
    def get_price_history(self, tickers=None, start_date=None, end_date=None, 
                          columns=None, normalize=False):
        """
        Extract price history as NumPy array.
        
        Args:
            tickers (list, optional): List of tickers to filter. If None, gets all.
            start_date (str, optional): Start date filter 'YYYY-MM-DD'.
            end_date (str, optional): End date filter 'YYYY-MM-DD'.
            columns (list, optional): Columns to extract (default: ['Open', 'High', 'Low', 'Close', 'Volume']).
            normalize (bool): Whether to normalize values to [0, 1] range (default: False).
        
        Returns:
            tuple: (numpy.ndarray, list, pandas.DatetimeIndex)
                   - Array of shape (n_samples, n_features)
                   - List of column names
                   - DatetimeIndex of dates
        """
        # Build query
        query = "SELECT * FROM stock_history"
        conditions = []
        
        if start_date:
            conditions.append(f"\"('Date', '')\" >= '{start_date}'")
        if end_date:
            conditions.append(f"\"('Date', '')\" <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY \"('Date', '')\""
        
        # Load data
        df = pd.read_sql_query(query, self.conn)
        
        if df.empty:
            return np.array([]), [], pd.DatetimeIndex([])
        
        # The date column has weird tuple format from MultiIndex
        date_col = [col for col in df.columns if 'Date' in str(col)][0]
        df['Date'] = pd.to_datetime(df[date_col])
        dates = df['Date']
        
        # Filter columns by ticker if specified
        numeric_cols = []
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in df.columns:
            col_str = str(col)
            if tickers:
                # Check if any of the target metrics and tickers are in this column
                for metric in columns:
                    for ticker in tickers:
                        if metric in col_str and ticker in col_str:
                            numeric_cols.append(col)
                            break
            else:
                # Include all metric columns
                for metric in columns:
                    if metric in col_str and 'Date' not in col_str:
                        numeric_cols.append(col)
                        break
        
        # Extract data
        if not numeric_cols:
            return np.array([]), [], dates
        
        data = df[numeric_cols].values.astype(np.float32)
        
        # Create readable column names
        readable_cols = [str(col).replace("('", "").replace("')", "").replace("', '", "_") for col in numeric_cols]
        
        # Normalize if requested
        if normalize:
            data = self._normalize(data)
        
        return data, readable_cols, dates
    
    def get_dividends(self, tickers=None, start_date=None, end_date=None):
        """
        Extract dividend history as NumPy array.
        
        Args:
            tickers (list, optional): List of tickers to filter.
            start_date (str, optional): Start date filter 'YYYY-MM-DD'.
            end_date (str, optional): End date filter 'YYYY-MM-DD'.
        
        Returns:
            tuple: (numpy.ndarray, pandas.DatetimeIndex, list)
                   - Array of shape (n_samples, 1) containing dividend amounts
                   - DatetimeIndex of dividend dates
                   - List of corresponding tickers
        """
        query = "SELECT * FROM stock_dividends"
        conditions = []
        
        if tickers:
            ticker_list = "', '".join(tickers)
            conditions.append(f"ticker IN ('{ticker_list}')")
        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, self.conn)
        
        if df.empty:
            return np.array([]), pd.DatetimeIndex([]), []
        
        df['date'] = pd.to_datetime(df['date'])
        return df['dividend'].values.astype(np.float32), df['date'], df['ticker'].tolist()
    
    def get_financials(self, tickers=None, metrics=None, annual=True, normalize=False):
        """
        Extract financial statement data as NumPy array.
        
        Args:
            tickers (list, optional): List of tickers to filter.
            metrics (list, optional): List of metric names to extract. If None, gets all.
            annual (bool): If True, uses annual financials; if False, uses quarterly (default: True).
            normalize (bool): Whether to normalize values (default: False).
        
        Returns:
            tuple: (numpy.ndarray, list, list, list)
                   - Array of shape (n_samples, n_metrics)
                   - List of metric names
                   - List of dates
                   - List of corresponding tickers
        """
        table = 'stock_financials' if annual else 'stock_quarterly_financials'
        query = f"SELECT * FROM {table}"
        conditions = []
        
        if tickers:
            ticker_list = "', '".join(tickers)
            conditions.append(f"ticker IN ('{ticker_list}')")
        if metrics:
            metric_list = "', '".join(metrics)
            conditions.append(f"metric IN ('{metric_list}')")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        df = pd.read_sql_query(query, self.conn)
        
        if df.empty:
            return np.array([]), [], [], []
        
        # Pivot to wide format: rows are (ticker, date), columns are metrics
        pivot = df.pivot_table(index=['ticker', 'date'], columns='metric', values='value')
        pivot = pivot.reset_index()
        
        metric_cols = [col for col in pivot.columns if col not in ['ticker', 'date']]
        data = pivot[metric_cols].values.astype(np.float32)
        
        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)
        
        if normalize:
            data = self._normalize(data)
        
        return data, metric_cols, pivot['date'].tolist(), pivot['ticker'].tolist()
    
    def get_balance_sheet(self, tickers=None, metrics=None, annual=True, normalize=False):
        """
        Extract balance sheet data as NumPy array.
        
        Args:
            tickers (list, optional): List of tickers to filter.
            metrics (list, optional): List of metric names to extract.
            annual (bool): If True, uses annual data; if False, uses quarterly (default: True).
            normalize (bool): Whether to normalize values (default: False).
        
        Returns:
            tuple: (numpy.ndarray, list, list, list)
                   - Array of shape (n_samples, n_metrics)
                   - List of metric names
                   - List of dates
                   - List of tickers
        """
        table = 'stock_balance_sheet' if annual else 'stock_quarterly_balance_sheet'
        return self._get_financial_table(table, tickers, metrics, normalize)
    
    def get_cashflow(self, tickers=None, metrics=None, annual=True, normalize=False):
        """
        Extract cash flow data as NumPy array.
        
        Args:
            tickers (list, optional): List of tickers to filter.
            metrics (list, optional): List of metric names to extract.
            annual (bool): If True, uses annual data; if False, uses quarterly (default: True).
            normalize (bool): Whether to normalize values (default: False).
        
        Returns:
            tuple: (numpy.ndarray, list, list, list)
                   - Array of shape (n_samples, n_metrics)
                   - List of metric names
                   - List of dates
                   - List of tickers
        """
        table = 'stock_cashflow' if annual else 'stock_quarterly_cashflow'
        return self._get_financial_table(table, tickers, metrics, normalize)
    
    def get_company_info(self, tickers=None, keys=None):
        """
        Extract company fundamental info as NumPy array.
        
        Args:
            tickers (list, optional): List of tickers to filter.
            keys (list, optional): List of info keys to extract (e.g., ['marketCap', 'beta', 'forwardPE']).
        
        Returns:
            tuple: (numpy.ndarray, list, list)
                   - Array of shape (n_tickers, n_keys) with numeric values
                   - List of key names
                   - List of tickers
        """
        query = "SELECT * FROM stock_info"
        conditions = []
        
        if tickers:
            ticker_list = "', '".join(tickers)
            conditions.append(f"ticker IN ('{ticker_list}')")
        if keys:
            key_list = "', '".join(keys)
            conditions.append(f"key IN ('{key_list}')")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        df = pd.read_sql_query(query, self.conn)
        
        if df.empty:
            return np.array([]), [], []
        
        # Pivot to wide format
        pivot = df.pivot_table(index='ticker', columns='key', values='value', aggfunc='first')
        pivot = pivot.reset_index()
        
        key_cols = [col for col in pivot.columns if col != 'ticker']
        
        # Convert to numeric where possible
        data_numeric = []
        for col in key_cols:
            try:
                data_numeric.append(pd.to_numeric(pivot[col], errors='coerce').values)
            except:
                data_numeric.append(np.zeros(len(pivot)))
        
        data = np.column_stack(data_numeric).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        
        return data, key_cols, pivot['ticker'].tolist()
    
    def get_interest_rates(self, country_codes=None, start_year=None, end_year=None):
        """
        Extract interest rate data as NumPy array.
        
        Args:
            country_codes (list, optional): List of country codes to filter.
            start_year (int, optional): Start year filter.
            end_year (int, optional): End year filter.
        
        Returns:
            tuple: (numpy.ndarray, list, list)
                   - Array of shape (n_samples, 1) containing interest rates
                   - List of years
                   - List of country codes
        """
        query = "SELECT * FROM interest_rates"
        conditions = []
        
        if country_codes:
            country_list = "', '".join(country_codes)
            conditions.append(f"country_code IN ('{country_list}')")
        if start_year:
            conditions.append(f"year >= {start_year}")
        if end_year:
            conditions.append(f"year <= {end_year}")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY year"
        
        df = pd.read_sql_query(query, self.conn)
        
        if df.empty:
            return np.array([]), [], []
        
        return df['value'].values.astype(np.float32), df['year'].tolist(), df['country_code'].tolist()
    
    def _get_financial_table(self, table, tickers, metrics, normalize):
        """Helper method to extract data from financial statement tables."""
        query = f"SELECT * FROM {table}"
        conditions = []
        
        if tickers:
            ticker_list = "', '".join(tickers)
            conditions.append(f"ticker IN ('{ticker_list}')")
        if metrics:
            metric_list = "', '".join(metrics)
            conditions.append(f"metric IN ('{metric_list}')")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        df = pd.read_sql_query(query, self.conn)
        
        if df.empty:
            return np.array([]), [], [], []
        
        # Pivot to wide format
        pivot = df.pivot_table(index=['ticker', 'date'], columns='metric', values='value')
        pivot = pivot.reset_index()
        
        metric_cols = [col for col in pivot.columns if col not in ['ticker', 'date']]
        data = pivot[metric_cols].values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        
        if normalize:
            data = self._normalize(data)
        
        return data, metric_cols, pivot['date'].tolist(), pivot['ticker'].tolist()
    
    def _normalize(self, data):
        """Normalize data to [0, 1] range using min-max scaling."""
        data_min = data.min(axis=0, keepdims=True)
        data_max = data.max(axis=0, keepdims=True)
        # Avoid division by zero
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        return (data - data_min) / data_range
    
    def _to_timezone_naive(self, dates):
        """Convert dates to timezone-naive DatetimeIndex."""
        if isinstance(dates, pd.DatetimeIndex):
            return dates.tz_localize(None) if dates.tz is not None else dates
        # Convert list/array to DatetimeIndex with UTC handling
        dt_index = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
        return dt_index.tz_localize(None)
    
    def get_full_dataset(self, tickers, start_date=None, end_date=None, normalize=True):
        """
        Get a complete dataset combining multiple data sources for model training.
        
        Args:
            tickers (list): List of tickers to extract.
            start_date (str, optional): Start date 'YYYY-MM-DD'.
            end_date (str, optional): End date 'YYYY-MM-DD'.
            normalize (bool): Whether to normalize numeric features (default: True).
        
        Returns:
            dict: Dictionary containing:
                - 'price_history': (data, columns, dates)
                - 'dividends': (data, dates, tickers)
                - 'financials': (data, metrics, dates, tickers)
                - 'balance_sheet': (data, metrics, dates, tickers)
                - 'cashflow': (data, metrics, dates, tickers)
                - 'company_info': (data, keys, tickers)
                - 'interest_rates': (data, years, countries) - gets all countries
        """
        dataset = {}
        
        # Price history
        dataset['price_history'] = self.get_price_history(
            tickers=tickers, 
            start_date=start_date, 
            end_date=end_date,
            normalize=normalize
        )
        
        # Dividends
        dataset['dividends'] = self.get_dividends(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        # Financial statements
        dataset['financials'] = self.get_financials(
            tickers=tickers,
            annual=True,
            normalize=normalize
        )
        
        dataset['balance_sheet'] = self.get_balance_sheet(
            tickers=tickers,
            annual=True,
            normalize=normalize
        )
        
        dataset['cashflow'] = self.get_cashflow(
            tickers=tickers,
            annual=True,
            normalize=normalize
        )
        
        # Company info
        dataset['company_info'] = self.get_company_info(tickers=tickers)
        
        # Interest rates (all countries)
        if start_date and end_date:
            start_year = int(start_date[:4])
            end_year = int(end_date[:4])
        else:
            start_year = end_year = None
        
        dataset['interest_rates'] = self.get_interest_rates(
            start_year=start_year,
            end_year=end_year
        )
        
        return dataset
    
    def get_unified_array(self, ticker, start_date=None, end_date=None, normalize=True):
        """
        Create a single unified time-aligned NumPy array combining all data sources.
        
        This method:
        1. Uses daily price data as the base timeline
        2. Forward-fills financial statements (repeats last known quarterly/annual values)
        3. Joins dividends on their dates (0 for non-dividend days)
        4. Joins interest rates by year
        5. Adds static company info features (repeated for all days)
        
        Args:
            ticker (str): Single ticker symbol to extract (unified arrays work best with one ticker).
            start_date (str, optional): Start date 'YYYY-MM-DD'.
            end_date (str, optional): End date 'YYYY-MM-DD'.
            normalize (bool): Whether to normalize numeric features (default: True).
        
        Returns:
            tuple: (numpy.ndarray, list, pandas.DatetimeIndex)
                   - Unified array of shape (n_days, total_features)
                   - List of feature names in order
                   - DatetimeIndex of dates
        """
        # 1. Get daily price history as base
        price_data, price_cols, dates = self.get_price_history(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            normalize=False  # We'll normalize at the end
        )
        
        if len(price_data) == 0:
            return np.array([]), [], pd.DatetimeIndex([])
        
        # Create DataFrame with dates as index
        df = pd.DataFrame(price_data, columns=price_cols, index=dates)
        feature_names = price_cols.copy()
        
        # 2. Add dividends (0 for non-dividend days)
        div_data, div_dates, div_tickers = self.get_dividends(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date
        )
        if len(div_data) > 0:
            div_dates_clean = self._to_timezone_naive(div_dates)
            div_series = pd.Series(div_data, index=div_dates_clean)
            df['dividend'] = div_series.reindex(dates, fill_value=0.0)
            feature_names.append('dividend')
        else:
            df['dividend'] = 0.0
            feature_names.append('dividend')
        
        # 3. Add quarterly financials (forward-fill)
        fin_data, fin_metrics, fin_dates, fin_tickers = self.get_financials(
            tickers=[ticker],
            annual=False,  # Use quarterly for more frequent updates
            normalize=False
        )
        if len(fin_data) > 0:
            # Create a mapping of date -> financial metrics
            fin_dates_dt = self._to_timezone_naive(fin_dates)
            for i, metric in enumerate(fin_metrics):
                # Create series with quarterly dates
                metric_series = pd.Series(fin_data[:, i], index=fin_dates_dt)
                # Reindex to daily and forward-fill
                df[f'fin_{metric}'] = metric_series.reindex(dates, method='ffill').fillna(0.0)
                feature_names.append(f'fin_{metric}')
        
        # 4. Add quarterly balance sheet (forward-fill)
        bs_data, bs_metrics, bs_dates, bs_tickers = self.get_balance_sheet(
            tickers=[ticker],
            annual=False,
            normalize=False
        )
        if len(bs_data) > 0:
            bs_dates_dt = self._to_timezone_naive(bs_dates)
            for i, metric in enumerate(bs_metrics):
                metric_series = pd.Series(bs_data[:, i], index=bs_dates_dt)
                df[f'bs_{metric}'] = metric_series.reindex(dates, method='ffill').fillna(0.0)
                feature_names.append(f'bs_{metric}')
        
        # 5. Add quarterly cash flow (forward-fill)
        cf_data, cf_metrics, cf_dates, cf_tickers = self.get_cashflow(
            tickers=[ticker],
            annual=False,
            normalize=False
        )
        if len(cf_data) > 0:
            cf_dates_dt = self._to_timezone_naive(cf_dates)
            for i, metric in enumerate(cf_metrics):
                metric_series = pd.Series(cf_data[:, i], index=cf_dates_dt)
                df[f'cf_{metric}'] = metric_series.reindex(dates, method='ffill').fillna(0.0)
                feature_names.append(f'cf_{metric}')
        
        # 6. Add company info (static, repeated for all days)
        info_data, info_keys, info_tickers = self.get_company_info(tickers=[ticker])
        if len(info_data) > 0:
            for i, key in enumerate(info_keys):
                df[f'info_{key}'] = info_data[0, i]  # Same value for all rows
                feature_names.append(f'info_{key}')
        
        # 7. Add interest rates (by year, forward-fill)
        ir_data, ir_years, ir_countries = self.get_interest_rates()
        if len(ir_data) > 0:
            # Group by country
            ir_df = pd.DataFrame({
                'year': ir_years,
                'country': ir_countries,
                'rate': ir_data
            })
            
            # For each country, map to daily dates
            for country in ir_df['country'].unique():
                country_data = ir_df[ir_df['country'] == country]
                # Create series with year-start dates
                year_dates = pd.to_datetime(country_data['year'].astype(str) + '-01-01')
                rate_series = pd.Series(country_data['rate'].values, index=year_dates)
                # Reindex to daily and forward-fill
                df[f'interest_rate_{country}'] = rate_series.reindex(dates, method='ffill').fillna(0.0)
                feature_names.append(f'interest_rate_{country}')
        
        # Convert to NumPy array
        unified_array = df.values.astype(np.float32)
        
        # Handle any remaining NaN values
        unified_array = np.nan_to_num(unified_array, nan=0.0)
        
        # Normalize if requested
        if normalize:
            unified_array = self._normalize(unified_array)
        
        return unified_array, feature_names, dates
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (auto-close connection)."""
        self.close()
