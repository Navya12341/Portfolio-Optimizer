"""
Data Acquisition Module for NSE Stock Data
Fetches and caches historical stock prices from Yahoo Finance
"""

import pandas as pd
import yfinance as yf
from datetime import datetime

class NSEDataFetcher:
    def __init__(self):
        """Initialize NSE Data Fetcher"""
        self.suffix = ".NS"  # NSE suffix for Yahoo Finance
    
    def fetch_stock_data(self, tickers, start_date, end_date=None):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            tickers (list): List of NSE stock symbols
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format (optional)
        """
        try:
            # Add .NS suffix to tickers
            nse_tickers = [f"{ticker}{self.suffix}" for ticker in tickers]
            
            # Fetch data
            data = yf.download(
                nse_tickers,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            # Handle single stock case
            if len(tickers) == 1:
                data.columns = pd.MultiIndex.from_product([['Close'], [tickers[0]]])
            
            # Select only Close prices and clean column names
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            else:
                prices = data['Close']
                
            # Clean column names (remove .NS suffix)
            prices.columns = [col.replace(self.suffix, '') for col in prices.columns]
            
            return prices
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def compute_returns(self, prices):
        """
        Compute daily returns from prices
        
        Args:
            prices (pd.DataFrame): DataFrame of stock prices
        """
        try:
            # Compute daily returns
            returns = prices.pct_change()
            
            # Drop first row (NaN values)
            returns = returns.dropna()
            
            return returns
            
        except Exception as e:
            print(f"Error computing returns: {str(e)}")
            return pd.DataFrame()
    
    def fetch_benchmark(self, start_date, end_date=None):
        """Fetch NIFTY 50 index data"""
        try:
            self.logger.info("Fetching NIFTY 50 benchmark data...")
            nifty_data = yf.download(
                '^NSEI',  # NIFTY 50 symbol
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if nifty_data.empty:
                raise ValueError("No benchmark data received")
                
            prices = nifty_data['Adj Close'] if 'Adj Close' in nifty_data else nifty_data['Close']
            return prices
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to fetch benchmark: {str(e)}")
            return None

# Default NSE large-cap stocks
DEFAULT_NSE_TICKERS = [
    'RELIANCE',   # Reliance Industries
    'TCS',        # Tata Consultancy Services
    'HDFCBANK',   # HDFC Bank
    'INFY',       # Infosys
    'HINDUNILVR', # Hindustan Unilever
    'ICICIBANK',  # ICICI Bank
    'BHARTIARTL', # Bharti Airtel
    'ITC',        # ITC Limited
    'KOTAKBANK',  # Kotak Mahindra Bank
    'LT'          # Larsen & Toubro
]


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = NSEDataFetcher()
    prices = fetcher.fetch_stock_data(DEFAULT_NSE_TICKERS)
    print("\nStock Prices Shape:", prices.shape)
    print("\nFirst few rows:")
    print(prices.head())
    print("\nLast few rows:")
    print(prices.tail())
    
    returns = fetcher.compute_returns(prices)
    print("\nReturns Shape:", returns.shape)
    print("\nReturns Summary:")
    print(returns.describe())