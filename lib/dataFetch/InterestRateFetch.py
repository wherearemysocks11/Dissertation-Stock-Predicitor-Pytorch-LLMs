import requests
import pandas as pd
from datetime import datetime


class InterestRateFetcher:
    """
    A class to fetch interest rate data from the World Bank API.
    """
    
    def __init__(self, indicator='FR.INR.RINR', timeout=30):
        """
        Initialize the InterestRateFetcher.
        
        Args:
            indicator (str): World Bank indicator code (default: 'FR.INR.RINR' - Real interest rate).
                           Options: 'FR.INR.RINR', 'FR.INR.LEND', 'FR.INR.DPST'
            timeout (int): Request timeout in seconds (default: 30).
        """
        self.indicator = indicator
        self.timeout = timeout
        self.base_url = "https://api.worldbank.org/v2/country"
    
    def fetch(self, country_codes=['USA', 'GBR', 'DEU', 'JPN', 'CHN'], startDate='1970-01-01'):
        """
        Fetch interest rate data for multiple countries from the World Bank API.
        
        Args:
            country_codes (list): List of ISO 3-letter country codes (default: ['USA', 'GBR', 'DEU', 'JPN', 'CHN']).
                                 Examples: ['USA', 'GBR', 'EUR', 'IND']
            startDate (str): Starting date for data in 'YYYY-MM-DD' format (default: '1970-01-01').
        
        Returns:
            dict: Dictionary where keys are country codes and values are pandas DataFrames.
                  Each DataFrame contains interest rate data for that country with columns:
                  'year', 'value', 'country', 'indicator', 'country_code'
                  Returns an empty dictionary if no data is retrieved.
        
        Raises:
            ValueError: If country_codes is not a list, is empty, or date format is invalid
            Exception: If API request fails
        """
        # Validate input
        self._validate_country_codes(country_codes)
        self._validate_date(startDate)
        
        try:
            # Extract the year from the startDate string (YYYY-MM-DD)
            start_year = startDate.split('-')[0]
            # Get the current year for the end of the range
            end_year = datetime.now().year

            # Dictionary to store DataFrames for each country
            country_data = {}

            # Loop through each country code in the provided list
            for country_code in country_codes:
                try:
                    # Construct the World Bank API URL for the specific country and indicator
                    url = f"{self.base_url}/{country_code}/indicator/{self.indicator}"

                    # Set query parameters for the API request
                    params = {
                        'date': f'{start_year}:{end_year}',  # Date range for data
                        'format': 'json'                      # Return data in JSON format
                    }

                    # Make GET request to World Bank API
                    response = requests.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()  # Raise exception for HTTP errors
                    data = response.json()

                    # World Bank API returns a list with [metadata, actual_data]
                    # Check if actual data exists (index 1) and is not empty
                    if len(data) < 2 or not data[1]:
                        print(f"Warning: No data available for country code: {country_code}")
                        continue  # Skip to next country if no data found

                    # Extract and structure the data from the API response
                    records = []
                    for item in data[1]:
                        # Only include records where the value is not null
                        if item['value'] is not None:
                            records.append({
                                'year': int(item['date']),              # Year of the interest rate value
                                'value': float(item['value']),          # Interest rate value
                                'country': item['country']['value'],    # Full country name
                                'indicator': item['indicator']['value'], # Indicator description
                                'country_code': country_code            # ISO country code
                            })

                    # If we have valid records, create a DataFrame and add to dictionary
                    if records:
                        df = pd.DataFrame(records)
                        df = df.sort_values('year').reset_index(drop=True)  # Sort by year
                        country_data[country_code] = df
                        
                except requests.exceptions.Timeout:
                    print(f"Warning: Request timeout for country code: {country_code}")
                    continue
                except requests.exceptions.RequestException as e:
                    print(f"Warning: Failed to fetch data for {country_code}: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Warning: Unexpected error for {country_code}: {str(e)}")
                    continue

            # Return dictionary of DataFrames (one per country)
            if country_data:
                return country_data
            else:
                # Return empty dictionary if no data found
                print("Warning: No data retrieved for any country")
                return {}
        
        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Catch any other unexpected errors
            raise Exception(f"Error fetching interest rate data: {str(e)}")
    
    def _validate_country_codes(self, country_codes):
        """Validate country_codes parameter."""
        if not isinstance(country_codes, list):
            raise ValueError("country_codes must be a list of country codes")
        if len(country_codes) == 0:
            raise ValueError("country_codes list cannot be empty")
    
    def _validate_date(self, startDate):
        """Validate date format."""
        try:
            datetime.strptime(startDate, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"startDate must be in 'YYYY-MM-DD' format, got: {startDate}")