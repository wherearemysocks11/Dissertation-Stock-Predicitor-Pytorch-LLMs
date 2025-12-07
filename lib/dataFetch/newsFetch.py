import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
import pandas as pd

# Load environment variables from .env file at module import time
load_dotenv()


class NewsFetcher:
    """
    A class to fetch news articles using NewsAPI.org.
    """
    
    def __init__(self, api_key=os.getenv('NEWSAPI_KEY')):
        """
        Initialize the NewsFetcher.
        
        Args:
            api_key : (default: os.getenv('NEWSAPI_KEY')).
        """
        # Get API key from parameter
        self.api_key = api_key
        
        if not self.api_key:
            raise ValueError("NEWSAPI_KEY not found in environment variables. Check your .env file.")
        
        # Initialize the News API client
        self.newsapi = NewsApiClient(api_key=self.api_key)
    
    def fetch(self, ticker, page_size=100, sort_by='relevancy'):
        """
        Fetch recent news articles related to the given stock ticker.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA').
            page_size (int): Number of articles to fetch (default: 100, max: 100 on free plan).
            sort_by (str): Sort order - 'relevancy', 'popularity', or 'publishedAt' (default: 'relevancy').
            
        Returns:
            pandas.DataFrame: A DataFrame of news articles with columns:
                             'ticker', 'title', 'source', 'author', 
                             'url', 'publishedAt', 'content'
        
        Raises:
            ValueError: If ticker is not a string or is empty
            Exception: If API request fails
        """
        # Validate input
        self._validate_ticker(ticker)
        
        try:
            # List to store all articles
            all_data = []

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
                        'url': article.get('url'),
                        'publishedAt': article.get('publishedAt'),
                        'content': article.get('content')
                    })
            except Exception as e:
                raise Exception(f"Failed to fetch news for {ticker}: {str(e)}")

            # Convert the list of article records to a pandas DataFrame for analysis
            return pd.DataFrame(all_data)
        
        except ValueError as e:
            # Re-raise ValueError for missing API key or validation errors
            raise
        except Exception as e:
            # Catch any other unexpected errors
            raise Exception(f"Error fetching news data: {str(e)}")
    
    def _validate_ticker(self, ticker):
        """Validate ticker parameter."""
        if not isinstance(ticker, str):
            raise ValueError("ticker must be a string")
        if len(ticker) == 0:
            raise ValueError("ticker cannot be empty")
