import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
import pandas as pd

def newsFetch(tickers):
    """
    Fetch recent news articles related to the given stock ticker(s).
    
    Args:
        tickers (list): List of stock ticker symbols.
        
    Returns:
        pandas.DataFrame: A DataFrame of news articles with columns:
                         'ticker', 'title', 'source', 'author', 'description', 
                         'url', 'publishedAt', 'content'
    
    Raises:
        ValueError: If tickers is not a list or is empty
        Exception: If API key is not found or API request fails
    """
    # Validate input
    if not isinstance(tickers, list):
        raise ValueError("tickers must be a list of ticker symbols")
    if len(tickers) == 0:
        raise ValueError("tickers list cannot be empty")
    
    try:
        # Load environment variables from .env file (for API key security)
        load_dotenv()

        # Initialize the News API client using the API key from environment
        api_key = os.getenv('NEWSAPI_KEY')
        if not api_key:
            raise ValueError("NEWSAPI_KEY not found in environment variables. Check your .env file.")
        
        newsapi = NewsApiClient(api_key=api_key)

        # List to store all articles for all tickers
        all_data = []

        # Loop through each ticker symbol in the provided list
        for ticker in tickers:
            try:
                # Query NewsAPI for articles related to the ticker
                all_articles = newsapi.get_everything(
                    q=ticker,
                    sort_by='relevancy',
                    page_size=100  # 100 is maximum allowed for free tier
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