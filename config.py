"""
Configuration Module
Central location for all project settings and parameters
"""

import os
from datetime import datetime, timedelta
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# ============ DATA SETTINGS ============

# Default NSE Large-Cap Stocks
DEFAULT_TICKERS = [
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

# Alternative stock sets for different sectors
BANKING_STOCKS = ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK']
IT_STOCKS = ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM']
PHARMA_STOCKS = ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON']

# Date Range
DEFAULT_START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 years
DEFAULT_END_DATE = None  # Current date

# Data cache directory
DATA_DIR = 'data'
RESULTS_DIR = 'results'

# ============ PORTFOLIO SETTINGS ============

# Risk-free rate (annual)
RISK_FREE_RATE = 0.06  # 6% typical for India

# Trading days per year
TRADING_DAYS_PER_YEAR = 252

# Monte Carlo simulation parameters
NUM_SIMULATIONS = 10000
RANDOM_SEED = 42

# Optimization constraints
ALLOW_SHORT_SELLING = False  # Set to True to allow short positions
MIN_WEIGHT = 0.0  # Minimum weight per asset (0 = no short selling)
MAX_WEIGHT = 0.5  # Maximum weight per asset (0.5 = 50% max allocation)

# ============ MACHINE LEARNING SETTINGS ============

# Feature engineering
LOOKBACK_PERIOD = 60  # Days of historical data for features
PREDICTION_HORIZON = 20  # Days ahead to predict (monthly)

# Model parameters
ML_MODEL_TYPE = 'random_forest'  # Options: 'random_forest', 'gradient_boosting'
ML_N_ESTIMATORS = 100
ML_MAX_DEPTH = 10
ML_MIN_SAMPLES_SPLIT = 10
ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42

# Feature lags
FEATURE_LAGS = [1, 5, 10, 20]
FEATURE_WINDOWS = [5, 10, 20, 60]

# ============ VISUALIZATION SETTINGS ============

# Figure sizes
FIGURE_SIZE_LARGE = (14, 8)
FIGURE_SIZE_MEDIUM = (12, 6)
FIGURE_SIZE_SMALL = (10, 6)

# DPI for saved figures
FIGURE_DPI = 300

# Color schemes
COLORMAP_RETURNS = 'viridis'
COLORMAP_CORRELATION = 'coolwarm'
COLORMAP_ALLOCATION = 'Set3'

# Plot styles
PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # Use seaborn style

# ============ OUTPUT SETTINGS ============

# File naming
CACHE_FILE_PATTERN = 'nse_stocks_{start}_{end}.csv'
EFFICIENT_FRONTIER_FILE = 'efficient_frontier.png'
ALLOCATION_MAX_SHARPE_FILE = 'allocation_max_sharpe.png'
ALLOCATION_MIN_RISK_FILE = 'allocation_min_risk.png'
ALLOCATION_ML_FILE = 'allocation_ml_optimized.png'
ML_PREDICTIONS_FILE = 'ml_predictions.png'
PORTFOLIO_COMPARISON_FILE = 'portfolio_comparison.png'
COMPARISON_CSV_FILE = 'portfolio_comparison.csv'

# ============ DISPLAY SETTINGS ============

# Console output
PRINT_WIDTH = 80
DECIMAL_PLACES = 4

# Table formatting
PERCENTAGE_FORMAT = '{:.2%}'
FLOAT_FORMAT = '{:.4f}'
CURRENCY_FORMAT = '₹{:,.2f}'

# ============ VALIDATION SETTINGS ============

# Data quality checks
MIN_DATA_POINTS = 1000  # Minimum trading days required
MAX_MISSING_RATIO = 0.05  # Maximum 5% missing data allowed

# Optimization bounds
MIN_RETURN = -1.0  # -100% minimum return
MAX_RETURN = 5.0   # 500% maximum return
MIN_SHARPE = -5.0  # Minimum Sharpe ratio
MAX_SHARPE = 10.0  # Maximum Sharpe ratio

# ============ FEATURE FLAGS ============

# Enable/disable features
ENABLE_CACHING = True
FORCE_DATA_REFRESH = False
ENABLE_PROGRESS_BARS = True
VERBOSE_OUTPUT = True
SAVE_INTERMEDIATE_RESULTS = True

# ============ HELPER FUNCTIONS ============

def get_data_path(filename):
    """Get full path for data file"""
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, filename)

def get_results_path(filename):
    """Get full path for results file"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, filename)

def format_percentage(value):
    """Format value as percentage"""
    return PERCENTAGE_FORMAT.format(value)

def format_float(value):
    """Format float value"""
    return FLOAT_FORMAT.format(value)

def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "=" * PRINT_WIDTH)
    print(title.center(PRINT_WIDTH))
    print("=" * PRINT_WIDTH)

def print_subsection_header(title):
    """Print formatted subsection header"""
    print("\n" + title)
    print("-" * len(title))


# ============ ENVIRONMENT SETUP ============

def setup_environment():
    """Setup project environment"""
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Set matplotlib style
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass
    
    # Set random seeds for reproducibility
    import numpy as np
    import random
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print("Environment setup complete!")


# Portfolio configuration dictionary
PORTFOLIO_CONFIG = {
    'risk_free_rate': RISK_FREE_RATE,
    'simulation_count': NUM_SIMULATIONS,
    'lookback_period': LOOKBACK_PERIOD,
    'start_date': DEFAULT_START_DATE,
    'plot_dpi': FIGURE_DPI,
    'figure_size': FIGURE_SIZE_LARGE
}


if __name__ == "__main__":
    print("Configuration module loaded successfully")
    print(f"\nDefault tickers: {', '.join(DEFAULT_TICKERS)}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Risk-free rate: {format_percentage(RISK_FREE_RATE)}")
    print(f"Number of simulations: {NUM_SIMULATIONS:,}")

def plot_portfolio_comparison(historical_port, ml_port, returns, save_path='results/portfolio_comparison.png'):
    """Compare historical and ML-based portfolio performance"""
    try:
        plt.figure(figsize=(14, 8))
        
        # Calculate cumulative returns
        hist_weights = np.array(historical_port['weights'])
        ml_weights = np.array(ml_port['weights'])
        
        hist_returns = returns.dot(hist_weights)
        ml_returns = returns.dot(ml_weights)
        
        hist_cumulative = (1 + hist_returns).cumprod()
        ml_cumulative = (1 + ml_returns).cumprod()
        
        # Plot cumulative returns
        plt.plot(hist_cumulative.index, hist_cumulative, 
                label='Historical Portfolio', linewidth=2)
        plt.plot(ml_cumulative.index, ml_cumulative, 
                label='ML-Based Portfolio', linewidth=2)
        
        plt.title('Portfolio Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
    finally:
        plt.close()  # Ensure figure is closed even if error occurs

def create_comparison_table(historical_port, ml_port, min_risk_port, tickers):
    # Use more memory-efficient dtypes
    comparison = pd.DataFrame({
        'Stock': tickers,
        'Max Sharpe (%)': historical_port['weights'] * 100,
        'ML-Based (%)': ml_port['weights'] * 100,
        'Min Risk (%)': min_risk_port['weights'] * 100
    }, dtype={
        'Stock': 'category',
        'Max Sharpe (%)': 'float32',
        'ML-Based (%)': 'float32',
        'Min Risk (%)': 'float32'
    })
    
    # ...existing code...

@lru_cache(maxsize=32)
def simulate_portfolios(num_portfolios=10000):
    """Simulate random portfolios using Monte Carlo simulation"""
    results = np.zeros((num_portfolios, 3))  # Return, Risk, Sharpe
    weights_array = np.zeros((num_portfolios, self.n_assets))
    
    for i in range(num_portfolios):
        weights = np.random.random(self.n_assets)
        weights = weights / np.sum(weights)
        weights_array[i] = weights
        
        portfolio_return = np.sum(weights * self.exp_returns) * 252
        portfolio_risk = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
        )
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        results[i] = [portfolio_return, portfolio_risk, sharpe_ratio]
        
    return pd.DataFrame(
        results,
        columns=['Return', 'Risk', 'Sharpe']
    ), weights_array

def save_results(file_path, data):
    """Safe file saving with path validation"""
    try:
        # Convert to absolute path
        file_path = os.path.abspath(file_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save file based on extension
        if file_path.endswith('.csv'):
            data.to_csv(file_path, index=False)
        elif file_path.endswith('.png'):
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            
    except Exception as e:
        print(f"Error saving file {file_path}: {str(e)}")

class MLReturnPredictor:
    def __init__(self, returns, lookback_period=60):
        """Initialize ML predictor with error checking"""
        self.returns = returns.copy()
        self.lookback_period = lookback_period
        self.models = {}
        self.predictions = {}
        
        # Fill any missing values in returns
        self.returns = self.returns.fillna(method='ffill').fillna(method='bfill')
    
    def create_features(self, data):
        """Create features with proper handling of NaN values"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Rolling statistics
            features['rolling_mean'] = data.rolling(window=self.lookback_period, min_periods=1).mean()
            features['rolling_std'] = data.rolling(window=self.lookback_period, min_periods=1).std()
            features['momentum'] = data.pct_change(periods=self.lookback_period).fillna(0)
            
            # Volatility features
            features['volatility'] = data.rolling(window=self.lookback_period, min_periods=1).std()
            features['high_vol'] = (features['volatility'] > features['volatility'].mean()).astype(int)
            
            # Price level features
            features['ma_ratio'] = (data / features['rolling_mean']).fillna(1)
            
            # Clean up any remaining NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            print(f"Error creating features: {str(e)}")
            return pd.DataFrame()
    
    def train_all_models(self, model_type='random_forest'):
        """Train ML models with robust error handling"""
        training_results = {}
        
        for ticker in self.returns.columns:
            try:
                # Prepare data
                data = self.returns[ticker]
                features = self.create_features(data)
                
                if features.empty:
                    print(f"Skipping {ticker} - failed to create features")
                    continue
                
                # Create target (next day's return)
                target = data.shift(-1)
                
                # Align data and remove NaN
                features = features[:-1]  # Remove last row as we don't have target for it
                target = target[features.index]  # Align with features
                
                # Remove any remaining NaN
                valid_idx = ~(features.isna().any(axis=1) | target.isna())
                features = features[valid_idx]
                target = target[valid_idx]
                
                if len(features) < self.lookback_period:
                    print(f"Insufficient data for {ticker}")
                    continue
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=3,
                    min_samples_leaf=50,
                    random_state=42
                )
                
                model.fit(features, target)
                self.models[ticker] = model
                
                # Store results
                training_results[ticker] = {
                    'r2_score': model.score(features, target),
                    'feature_importance': dict(zip(features.columns, 
                                                 model.feature_importances_))
                }
                
                print(f"Successfully trained model for {ticker}")
                
            except Exception as e:
                print(f"Error training model for {ticker}: {str(e)}")
                continue
        
        return training_results
    
    def predict_next_period(self):
        """Predict next period returns with error handling"""
        try:
            predictions = {}
            
            for ticker in self.returns.columns:
                if ticker not in self.models:
                    print(f"No model available for {ticker}")
                    predictions[ticker] = 0.0
                    continue
                    
                try:
                    features = self.create_features(self.returns[ticker])
                    if not features.empty:
                        latest_features = features.iloc[[-1]]
                        pred = self.models[ticker].predict(latest_features)[0]
                        predictions[ticker] = pred
                    else:
                        predictions[ticker] = 0.0
                        
                except Exception as e:
                    print(f"Error predicting for {ticker}: {str(e)}")
                    predictions[ticker] = 0.0
            
            self.predictions = predictions
            return predictions
            
        except Exception as e:
            print(f"Error in prediction pipeline: {str(e)}")
            return {ticker: 0.0 for ticker in self.returns.columns}
    
    def plot_predictions_vs_actual(self, save_path='results/ml_predictions.png'):
        """Plot predicted vs actual returns"""
        # Prepare data
        actual_returns = self.returns.iloc[-252:].mean() * 252  # Annualized
        predicted_returns = pd.Series(self.predictions)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Actual': actual_returns,
            'Predicted': predicted_returns
        })
        
        # Plot
        plt.figure(figsize=(10, 6))
        comparison.plot(kind='bar')
        plt.title('ML Predicted vs Actual Returns')
        plt.xlabel('Assets')
        plt.ylabel('Annual Return')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison
    
    def create_ml_optimized_portfolio(self):
        """Create portfolio weights based on ML predictions"""
        predictions = pd.Series(self.predictions)
        
        # Simple ranking-based allocation
        ranked_assets = predictions.sort_values(ascending=False)
        n_assets = len(ranked_assets)
        
        # Linear decay weights
        weights = np.linspace(0.2, 0.05, n_assets)
        weights = weights / weights.sum()  # Normalize to sum to 1
        
        # Create portfolio weights Series
        portfolio_weights = pd.Series(weights, index=ranked_assets.index)
        portfolio_weights = portfolio_weights.reindex(self.returns.columns)
        
        return portfolio_weights.values