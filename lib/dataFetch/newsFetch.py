import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
import pandas as pd


class NewsFetcher:
    """
    A class to fetch news articles using NewsAPI.org.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the NewsFetcher.
        
        Args:
            api_key (str, optional): NewsAPI key. If not provided, will load from environment.
        """
        # Load environment variables from .env file (for API key security)
        load_dotenv()
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('NEWSAPI_KEY')
        
        if not self.api_key:
            raise ValueError("NEWSAPI_KEY not found in environment variables. Check your .env file.")
        
        # Initialize the News API client
        self.newsapi = NewsApiClient(api_key=self.api_key)
    
    def fetch(self, tickers, page_size=100, sort_by='relevancy'):
        """
        Fetch recent news articles related to the given stock ticker(s).
        
        Args:
            tickers (list): List of stock ticker symbols.
            page_size (int): Number of articles to fetch per ticker (default: 100, max: 100).
            sort_by (str): Sort order - 'relevancy', 'popularity', or 'publishedAt' (default: 'relevancy').
            
        Returns:
            pandas.DataFrame: A DataFrame of news articles with columns:
                             'ticker', 'title', 'source', 'author', 'description', 
                             'url', 'publishedAt', 'content'
        
        Raises:
            ValueError: If tickers is not a list or is empty
            Exception: If API request fails
        """
        # Validate input
        self._validate_tickers(tickers)
        
        try:
            # List to store all articles for all tickers
            all_data = []

            # Loop through each ticker symbol in the provided list
            for ticker in tickers:
                try:
                    # Query NewsAPI for articles related to the ticker
                    all_articles = self.newsapi.get_everything(
                        q=ticker,
                        sort_by=sort_by,
                        page_size=page_size  # 100 is maximum allowed for free tier
                    )

                    # For each article, add a record with ticker identifier and relevant fields
                    for article in all_articles.get('articles', []):
                        all_data.append({
                            'ticker': ticker,  # Stock ticker symbol
                            'title': article.get('title'),
                            'source': article.get('source', {}).get('name'),
                            'author': article.get('author'),
                            'description': article.get('description'),
                            'url': article.get('url'),
                            'publishedAt': article.get('publishedAt'),
                            'content': article.get('content')
                        })
                except Exception as e:
                    # Log error for this ticker but continue with others
                    print(f"Warning: Failed to fetch news for {ticker}: {str(e)}")
                    continue

            # Convert the list of article records to a pandas DataFrame for analysis
            return pd.DataFrame(all_data)
        
        except ValueError as e:
            # Re-raise ValueError for missing API key or validation errors
            raise
        except Exception as e:
            # Catch any other unexpected errors
            raise Exception(f"Error fetching news data: {str(e)}")
    
    def _validate_tickers(self, tickers):
        """Validate tickers parameter."""
        if not isinstance(tickers, list):
            raise ValueError("tickers must be a list of ticker symbols")
        if len(tickers) == 0:
            raise ValueError("tickers list cannot be empty")
