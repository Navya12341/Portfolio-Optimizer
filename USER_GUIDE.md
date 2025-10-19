# NSE Portfolio Optimization - User Guide

## 📖 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation Guide](#installation-guide)
3. [Module Documentation](#module-documentation)
4. [Usage Examples](#usage-examples)
5. [Customization Guide](#customization-guide)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [FAQ](#faq)

---

## Quick Start

### 2. eda.py

**Purpose**: Exploratory data analysis and visualization

**Key Class**: `StockEDA`

**Methods**:
- `plot_price_history()` - Normalized price chart
- `plot_cumulative_returns()` - Cumulative return visualization
- `plot_correlation_heatmap()` - Correlation matrix
- `plot_risk_return_scatter()` - Risk-return profile
- `generate_summary_statistics()` - Statistical metrics

**Example**:
```python
from eda import StockEDA

eda = StockEDA(prices, returns)
eda.print_summary()
eda.plot_correlation_heatmap()
```

### 3. portfolio.py

**Purpose**: Portfolio optimization using Modern Portfolio Theory

**Key Class**: `PortfolioOptimizer`

**Methods**:
- `simulate_portfolios(num_portfolios)` - Monte Carlo simulation
- `optimize_max_sharpe()` - Maximum Sharpe ratio portfolio
- `optimize_min_risk()` - Minimum variance portfolio
- `plot_efficient_frontier()` - Visualize efficient frontier
- `portfolio_performance(weights)` - Calculate metrics

**Example**:
```python
from portfolio import PortfolioOptimizer

optimizer = PortfolioOptimizer(returns, risk_free_rate=0.06)
max_sharpe = optimizer.optimize_max_sharpe()
print(f"Optimal Sharpe: {max_sharpe['sharpe']:.4f}")
```

### 4. ml_predict.py

**Purpose**: Machine learning-based return prediction

**Key Class**: `MLReturnPredictor`

**Methods**:
- `train_all_models()` - Train models for all stocks
- `predict_next_period()` - Predict future returns
- `create_ml_optimized_portfolio()` - Generate ML-based weights
- `plot_predictions_vs_actual()` - Compare predictions

**Example**:
```python
from ml_predict import MLReturnPredictor

predictor = MLReturnPredictor(returns)
predictor.train_all_models(model_type='random_forest')
predictions = predictor.predict_next_period()
```

### 5. utils.py

**Purpose**: Additional performance metrics and utilities

**Key Functions**:
- `calculate_max_drawdown()` - Maximum drawdown calculation
- `calculate_sortino_ratio()` - Sortino ratio
- `calculate_var_cvar()` - Value at Risk and CVaR
- `portfolio_metrics_summary()` - Comprehensive metrics
- `create_tear_sheet()` - Full performance report

**Example**:
```python
from utils import portfolio_metrics_summary, create_tear_sheet

metrics = portfolio_metrics_summary(portfolio_returns)
create_tear_sheet(returns, weights, tickers)
```

### 6. config.py

**Purpose**: Central configuration management

**Key Settings**:
- `DEFAULT_TICKERS` - Stock universe
- `RISK_FREE_RATE` - Risk-free rate
- `NUM_SIMULATIONS` - Monte Carlo iterations
- `ML_MODEL_TYPE` - Machine learning algorithm

**Modify settings**:
```python
import config

config.RISK_FREE_RATE = 0.07  # Change to 7%
config.NUM_SIMULATIONS = 20000  # More simulations
```

---

## Usage Examples

### Example 1: Basic Portfolio Optimization

```python
from data_fetch import NSEDataFetcher
from portfolio import PortfolioOptimizer

# Fetch data
fetcher = NSEDataFetcher()
prices = fetcher.fetch_stock_data(['RELIANCE', 'TCS', 'HDFCBANK'])
returns = fetcher.compute_returns(prices)

# Optimize
optimizer = PortfolioOptimizer(returns)
result = optimizer.optimize_max_sharpe()

print(f"Expected Return: {result['return']:.2%}")
print(f"Risk: {result['risk']:.2%}")
print(f"Sharpe Ratio: {result['sharpe']:.4f}")
```

### Example 2: Custom Stock Selection

```python
from data_fetch import NSEDataFetcher

# IT sector stocks
IT_STOCKS = ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM']

fetcher = NSEDataFetcher()
prices = fetcher.fetch_stock_data(IT_STOCKS, start_date='2021-01-01')
returns = fetcher.compute_returns(prices)

# Continue with analysis...
```

### Example 3: ML Prediction Pipeline

```python
from data_fetch import NSEDataFetcher
from ml_predict import MLReturnPredictor
from portfolio import PortfolioOptimizer

# Get data
fetcher = NSEDataFetcher()
prices = fetcher.fetch_stock_data(['RELIANCE', 'TCS', 'INFY'])
returns = fetcher.compute_returns(prices)

# Train ML model
predictor = MLReturnPredictor(returns)
predictor.train_all_models()
predictions = predictor.predict_next_period()

# Create ML portfolio
ml_weights = predictor.create_ml_optimized_portfolio()

# Evaluate
optimizer = PortfolioOptimizer(returns)
perf = optimizer.portfolio_performance(ml_weights)
print(f"ML Portfolio Sharpe: {perf[2]:.4f}")
```

### Example 4: Sector Comparison

```python
from data_fetch import NSEDataFetcher
from portfolio import PortfolioOptimizer

sectors = {
    'IT': ['TCS', 'INFY', 'WIPRO'],
    'Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK'],
    'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA']
}

fetcher = NSEDataFetcher()
results = {}

for sector, stocks in sectors.items():
    prices = fetcher.fetch_stock_data(stocks)
    returns = fetcher.compute_returns(prices)
    optimizer = PortfolioOptimizer(returns)
    result = optimizer.optimize_max_sharpe()
    results[sector] = result['sharpe']

# Compare
for sector, sharpe in results.items():
    print(f"{sector:10s} Sharpe: {sharpe:.4f}")
```

### Example 5: Custom Date Range Analysis

```python
from data_fetch import NSEDataFetcher
from eda import StockEDA

# Pre-COVID vs Post-COVID
periods = {
    'Pre-COVID': ('2018-01-01', '2020-02-01'),
    'Post-COVID': ('2020-06-01', '2023-12-31')
}

fetcher = NSEDataFetcher()
tickers = ['RELIANCE', 'TCS', 'HDFCBANK']

for period_name, (start, end) in periods.items():
    prices = fetcher.fetch_stock_data(tickers, start, end)
    returns = fetcher.compute_returns(prices)
    
    print(f"\n{period_name}:")
    print(f"Avg Return: {returns.mean().mean() * 252:.2%}")
    print(f"Avg Volatility: {returns.std().mean() * np.sqrt(252):.2%}")
```

---

## Customization Guide

### Change Stock Universe

**Method 1: Edit config.py**
```python
# In config.py
DEFAULT_TICKERS = [
    'RELIANCE',
    'TCS',
    'YOUR_STOCK_1',
    'YOUR_STOCK_2'
]
```

**Method 2: Pass custom list**
```python
my_stocks = ['TATASTEEL', 'JSWSTEEL', 'HINDALCO']
prices = fetcher.fetch_stock_data(my_stocks)
```

### Adjust Risk Parameters

```python
# In main.py or your script
optimizer = PortfolioOptimizer(
    returns,
    risk_free_rate=0.07  # Change from 6% to 7%
)
```

### Modify ML Model

```python
# In ml_predict.py, change model parameters
model = RandomForestRegressor(
    n_estimators=200,      # More trees
    max_depth=15,          # Deeper trees
    min_samples_split=5,   # Smaller split size
    random_state=42
)
```

### Add Custom Constraints

```python
# In portfolio.py, modify optimization
def optimize_with_sector_constraints(self, sector_limits):
    weights = cp.Variable(self.num_assets)
    
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        # Add sector constraints
        cp.sum(weights[sector_indices]) <= sector_limit
    ]
    # Continue with optimization...
```

### Custom Visualization

```python
import matplotlib.pyplot as plt

# Create custom plot
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns)
plt.title('My Custom Portfolio Analysis')
plt.savefig('results/my_custom_plot.png', dpi=300)
```

---

## Troubleshooting

### Common Issues

#### 1. Data Download Fails

**Problem**: `ConnectionError` or timeout

**Solution**:
```python
# Try with force refresh
fetcher = NSEDataFetcher()
prices = fetcher.fetch_stock_data(tickers, force_refresh=True)

# Or check internet connection
# Some corporate networks block Yahoo Finance
```

#### 2. Invalid Ticker Symbol

**Problem**: `KeyError` or empty DataFrame

**Solution**:
```python
# NSE stocks need .NS suffix
# Correct: 'RELIANCE.NS' or just 'RELIANCE' (auto-added)
# Wrong: 'RELIANCE.BSE'

# Verify ticker exists
import yfinance as yf
stock = yf.Ticker('RELIANCE.NS')
print(stock.info['longName'])  # Should print company name
```

#### 3. Optimization Fails

**Problem**: `SolverError` from CVXPY

**Solution**:
```python
# Install alternative solver
pip install cvxopt

# Or increase solver tolerance
problem.solve(solver=cp.SCS, max_iters=10000)
```

#### 4. Memory Error

**Problem**: Out of memory with large simulations

**Solution**:
```python
# Reduce simulations
config.NUM_SIMULATIONS = 5000  # Instead of 10000

# Or process in batches
for i in range(10):
    results, weights = optimizer.simulate_portfolios(1000)
```

#### 5. ML Training Slow

**Problem**: Model training takes too long

**Solution**:
```python
# Reduce model complexity
predictor = MLReturnPredictor(returns, lookback_period=30)  # Reduce from 60

# Or use fewer estimators
config.ML_N_ESTIMATORS = 50  # Instead of 100
```

### Error Messages

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` | Missing package | `pip install <package>` |
| `FileNotFoundError` | Missing directory | Script auto-creates, check permissions |
| `KeyError` | Invalid ticker | Verify ticker symbol is correct |
| `ValueError: weights` | Optimization failed | Check data quality, increase iterations |
| `ConvergenceWarning` | ML model not converging | Increase iterations or change parameters |

---

## Best Practices

### 1. Data Quality

```python
# Always check data quality
print(f"Missing values: {prices.isnull().sum().sum()}")
print(f"Data points: {len(prices)}")
print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

# Ensure minimum data
assert len(prices) >= 1000, "Insufficient data points"
```

### 2. Reproducibility

```python
# Set random seeds at the start
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Save configuration
with open('results/config.txt', 'w') as f:
    f.write(f"Tickers: {tickers}\n")
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Risk-free rate: {risk_free_rate}\n")
```

### 3. Validation

```python
# Validate portfolio weights
assert np.abs(weights.sum() - 1.0) < 1e-6, "Weights don't sum to 1"
assert np.all(weights >= 0), "Negative weights found"

# Validate returns
assert not returns.isnull().any().any(), "NaN in returns"
assert np.isfinite(returns).all().all(), "Infinite values in returns"
```

### 4. Documentation

```python
# Document your analysis
def my_analysis():
    """
    Custom portfolio analysis for XYZ strategy
    
    Parameters:
    - Uses 5-year historical data
    - Optimizes for Sharpe ratio
    - Assumes 6% risk-free rate
    
    Returns:
    - Dictionary with optimal weights and metrics
    """
    # Your code here
```

### 5. Version Control

```bash
# Initialize git repository
git init
git add *.py requirements.txt README.md
git commit -m "Initial commit: Portfolio optimization project"

# Add .gitignore
echo "data/
results/
__pycache__/
*.pyc
venv/" > .gitignore
```

---

## FAQ

### General Questions

**Q: How much historical data do I need?**
A: Minimum 2-3 years (500+ trading days). 5 years is recommended for robust results.

**Q: Can I use this for intraday trading?**
A: No, this is designed for long-term portfolio allocation. Intraday requires different methods.

**Q: Is this production-ready for real money?**
A: This is an educational/research tool. For real investments, consult a financial advisor.

**Q: Can I use BSE stocks instead of NSE?**
A: Yes, use `.BO` suffix instead of `.NS` (e.g., 'RELIANCE.BO').

### Technical Questions

**Q: Why is my Sharpe ratio negative?**
A: Your portfolio return is below the risk-free rate. Consider different stocks or time period.

**Q: ML predictions seem random. Is this normal?**
A: Stock returns are inherently noisy. R² of 0.15-0.30 is typical. Use ensemble predictions.

**Q: Can I add transaction costs?**
A: Yes, modify the optimization to include:
```python
transaction_cost = 0.001  # 0.1%
adjusted_return = returns - transaction_cost * np.abs(weights - old_weights)
```

**Q: How do I compare with a benchmark (e.g., NIFTY)?**
A: Download NIFTY data and use the comparison functions:
```python
nifty_prices = fetcher.fetch_stock_data(['^NSEI'])  # NIFTY 50 index
nifty_returns = fetcher.compute_returns(nifty_prices)
```

**Q: Can I short-sell stocks?**
A: Modify constraints in `portfolio.py`:
```python
constraints = [
    cp.sum(weights) == 1,
    weights >= -0.3,  # Allow up to 30% short
    weights <= 1.0
]
```

### Performance Questions

**Q: How long does the full pipeline take?**
A: Typically 1-2 minutes on modern hardware for 10 stocks with 5 years of data.

**Q: Can I run this on Jupyter Notebook?**
A: Yes! All modules work in Jupyter. Just import and use:
```python
%matplotlib inline
from data_fetch import NSEDataFetcher
# Continue as normal...
```

**Q: How do I update data daily?**
A: Set up a scheduled task:
```bash
# Linux cron job
0 18 * * 1-5 cd /path/to/project && python main.py
```

---

## Additional Resources

### Learning Materials
- **Book**: "Python for Finance" by Yves Hilpisch
- **Course**: Coursera - "Investment Management with Python"
- **Tutorial**: QuantStart Portfolio Optimization Guide

### Documentation Links
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [CVXPY Documentation](https://www.cvxpy.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)

### Support
- **GitHub Issues**: Report bugs and request features
- **Email**: contact@example.com
- **Discussion Forum**: [Link to forum]

---

**Last Updated**: October 2025
**Version**: 1.0.0Minimum Setup (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python main.py

# 3. Check results
ls results/
```

That's it! The pipeline will:
- Download 5 years of NSE data
- Perform complete analysis
- Generate all visualizations
- Save results to `results/` folder

---

## Installation Guide

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for data download)
- ~200MB free disk space

### Step-by-Step Installation

#### Windows

```bash
# 1. Create project directory
mkdir nse_portfolio
cd nse_portfolio

# 2. Download project files
# (Copy all .py files to this directory)

# 3. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the project
python main.py
```

#### Mac/Linux

```bash
# 1. Create project directory
mkdir nse_portfolio
cd nse_portfolio

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python main.py
```

### Verification

Test your installation:

```python
python -c "import yfinance, pandas, numpy, cvxpy, sklearn; print('All packages installed successfully!')"
```

---

## Module Documentation

### 1. data_fetch.py

**Purpose**: Fetch and cache NSE stock data

**Key Class**: `NSEDataFetcher`

**Methods**:
- `fetch_stock_data(tickers, start_date, end_date)` - Download stock prices
- `compute_returns(prices)` - Calculate daily returns
- `get_stock_info(ticker)` - Get stock metadata

**Example**:
```python
from data_fetch import NSEDataFetcher

fetcher = NSEDataFetcher()
prices = fetcher.fetch_stock_data(['RELIANCE', 'TCS'], '2020-01-01')
returns = fetcher.compute_returns(prices)
```

###