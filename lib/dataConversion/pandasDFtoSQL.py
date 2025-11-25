import sqlite3
import pandas as pd
from pathlib import Path


class DataToSQL:
    """
    A class to convert fetched data (stocks, news, interest rates) to SQLite database.
    """
    
    def __init__(self, db_path='data/stock_data.db'):
        """
        Initialize the DataToSQL converter.
        
        Args:
            db_path (str): Path to SQLite database file (default: 'data/stock_data.db').
        """
        self.db_path = db_path
        
        # Create parent directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create connection (will create file if it doesn't exist)
        self.conn = sqlite3.connect(db_path)
    
    def _convert_datetime_columns(self, df):
        """Convert all datetime columns in a DataFrame to strings for SQLite compatibility."""
        df_copy = df.copy()
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(str)
        return df_copy
    
    def store_stock_data(self, stock_data, if_exists='replace'):
        """
        Store stock data from StockFetcher into SQLite database.
        
        Args:
            stock_data (dict or DataFrame): Output from StockFetcher.fetch().
                                           If include_all=True, expects dict with multiple data types.
                                           If include_all=False, expects DataFrame (price history only).
            if_exists (str): What to do if table exists: 'replace', 'append', or 'fail' (default: 'replace').
        
        Returns:
            dict: Summary of tables created/updated with row counts.
        """
        summary = {}
        
        # Handle simple DataFrame (include_all=False case)
        if isinstance(stock_data, pd.DataFrame):
            return self._store_history_only(stock_data, if_exists)
        
        # Handle comprehensive dict (include_all=True case)
        if not isinstance(stock_data, dict):
            raise ValueError("stock_data must be either a DataFrame or dict from StockFetcher.fetch()")
        
        # 1. Store price history
        if 'history' in stock_data and not stock_data['history'].empty:
            df = stock_data['history'].reset_index()
            df = self._convert_datetime_columns(df)
            df.to_sql('stock_history', self.conn, if_exists=if_exists, index=False)
            summary['stock_history'] = len(df)
        
        # 2. Store dividends (per ticker)
        if 'dividends' in stock_data:
            all_divs = []
            for ticker, div_series in stock_data['dividends'].items():
                if not div_series.empty:
                    df = div_series.reset_index()
                    df.columns = ['date', 'dividend']
                    df['ticker'] = ticker
                    df = self._convert_datetime_columns(df)
                    all_divs.append(df)
            if all_divs:
                combined = pd.concat(all_divs, ignore_index=True)
                combined.to_sql('stock_dividends', self.conn, if_exists=if_exists, index=False)
                summary['stock_dividends'] = len(combined)
        
        # 3. Store splits (per ticker)
        if 'splits' in stock_data:
            all_splits = []
            for ticker, split_series in stock_data['splits'].items():
                if not split_series.empty:
                    df = split_series.reset_index()
                    df.columns = ['date', 'split_ratio']
                    df['ticker'] = ticker
                    df = self._convert_datetime_columns(df)
                    all_splits.append(df)
            if all_splits:
                combined = pd.concat(all_splits, ignore_index=True)
                combined.to_sql('stock_splits', self.conn, if_exists=if_exists, index=False)
                summary['stock_splits'] = len(combined)
        
        # 4-8. Store financial statements (convert wide to long format for flexibility)
        financial_tables = {
            'financials': 'stock_financials',
            'quarterly_financials': 'stock_quarterly_financials',
            'balance_sheet': 'stock_balance_sheet',
            'quarterly_balance_sheet': 'stock_quarterly_balance_sheet',
            'cashflow': 'stock_cashflow',
            'quarterly_cashflow': 'stock_quarterly_cashflow'
        }
        
        for data_key, table_name in financial_tables.items():
            if data_key in stock_data:
                all_records = []
                for ticker, df in stock_data[data_key].items():
                    if not df.empty:
                        # Convert wide format (metrics as rows, dates as cols) to long format
                        df_long = df.reset_index().melt(id_vars='index', var_name='date', value_name='value')
                        df_long.rename(columns={'index': 'metric'}, inplace=True)
                        df_long['ticker'] = ticker
                        # Convert date column explicitly (may be Timestamp from melt)
                        if 'date' in df_long.columns:
                            df_long['date'] = df_long['date'].astype(str)
                        all_records.append(df_long)
                if all_records:
                    combined = pd.concat(all_records, ignore_index=True)
                    combined.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
                    summary[table_name] = len(combined)
        
        # 9. Store company info/fundamentals (as key-value pairs)
        if 'info' in stock_data:
            all_info = []
            for ticker, info_dict in stock_data['info'].items():
                for key, value in info_dict.items():
                    all_info.append({
                        'ticker': ticker,
                        'key': key,
                        'value': str(value)  # Convert all to string for simplicity
                    })
            if all_info:
                df_info = pd.DataFrame(all_info)
                df_info.to_sql('stock_info', self.conn, if_exists=if_exists, index=False)
                summary['stock_info'] = len(df_info)
        
        # 10. Store analyst recommendations
        if 'recommendations' in stock_data:
            all_recs = []
            for ticker, df in stock_data['recommendations'].items():
                if df is not None and not df.empty:
                    df_copy = df.reset_index()
                    df_copy['ticker'] = ticker
                    df_copy = self._convert_datetime_columns(df_copy)
                    all_recs.append(df_copy)
            if all_recs:
                combined = pd.concat(all_recs, ignore_index=True)
                combined.to_sql('stock_recommendations', self.conn, if_exists=if_exists, index=False)
                summary['stock_recommendations'] = len(combined)
        
        # 11-12. Store earnings data
        earnings_tables = {
            'earnings': 'stock_earnings',
            'quarterly_earnings': 'stock_quarterly_earnings'
        }
        
        for data_key, table_name in earnings_tables.items():
            if data_key in stock_data:
                all_earnings = []
                for ticker, df in stock_data[data_key].items():
                    if df is not None and not df.empty:
                        df_copy = df.reset_index()
                        df_copy['ticker'] = ticker
                        df_copy = self._convert_datetime_columns(df_copy)
                        all_earnings.append(df_copy)
                if all_earnings:
                    combined = pd.concat(all_earnings, ignore_index=True)
                    combined.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
                    summary[table_name] = len(combined)
        
        self.conn.commit()
        return summary
    
    def store_news_data(self, news_df, if_exists='replace'):
        """
        Store news data from NewsFetcher into SQLite database.
        
        Args:
            news_df (DataFrame): Output from NewsFetcher.fetch().
            if_exists (str): What to do if table exists: 'replace', 'append', or 'fail' (default: 'replace').
        
        Returns:
            dict: Summary with row count.
        """
        if news_df.empty:
            return {'news_articles': 0}
        
        news_df.to_sql('news_articles', self.conn, if_exists=if_exists, index=False)
        self.conn.commit()
        return {'news_articles': len(news_df)}
    
    def store_interest_rate_data(self, interest_df, if_exists='replace'):
        """
        Store interest rate data from InterestRateFetcher into SQLite database.
        
        Args:
            interest_df (DataFrame): Output from InterestRateFetcher.fetch().
            if_exists (str): What to do if table exists: 'replace', 'append', or 'fail' (default: 'replace').
        
        Returns:
            dict: Summary with row count.
        """
        if interest_df.empty:
            return {'interest_rates': 0}
        
        interest_df.to_sql('interest_rates', self.conn, if_exists=if_exists, index=False)
        self.conn.commit()
        return {'interest_rates': len(interest_df)}
    
    def _store_history_only(self, history_df, if_exists='replace'):
        """Helper to store price history DataFrame only."""
        df = history_df.reset_index()
        df = self._convert_datetime_columns(df)
        df.to_sql('stock_history', self.conn, if_exists=if_exists, index=False)
        self.conn.commit()
        return {'stock_history': len(df)}
    
    def query(self, sql):
        """
        Execute a SQL query and return results as DataFrame.
        
        Args:
            sql (str): SQL query string.
        
        Returns:
            pandas.DataFrame: Query results.
        """
        return pd.read_sql_query(sql, self.conn)
    
    def list_tables(self):
        """
        List all tables in the database.
        
        Returns:
            list: Table names.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (auto-close connection)."""
        self.close()
