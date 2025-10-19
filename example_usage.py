"""
Example Usage Script
Demonstrates various use cases and customizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import project modules
from data_fetch import NSEDataFetcher
from eda import StockEDA
from portfolio_optimizer_v2 import PortfolioOptimizer
from ml_predict import MLReturnPredictor
from utils_enhanced import portfolio_metrics_summary, create_tear_sheet

print("NSE Portfolio Optimization - Example Usage")
print("=" * 80)

# ============================================================================
# EXAMPLE 1: Quick Portfolio Optimization
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 1: Quick Portfolio Optimization for 3 Stocks")
print("=" * 80)

# Simple 3-stock portfolio
simple_stocks = ['RELIANCE', 'TCS', 'HDFCBANK']

fetcher = NSEDataFetcher()
prices = fetcher.fetch_stock_data(simple_stocks, start_date='2022-01-01')
returns = fetcher.compute_returns(prices)

# Optimize
optimizer = PortfolioOptimizer(returns, risk_free_rate=0.06)
max_sharpe = optimizer.optimize_max_sharpe()

print(f"\nOptimal Portfolio (Max Sharpe Ratio):")
print(f"  Expected Return: {max_sharpe['return']:.2%}")
print(f"  Risk (Std Dev):  {max_sharpe['risk']:.2%}")
print(f"  Sharpe Ratio:    {max_sharpe['sharpe']:.4f}")
print(f"\nAllocations:")
for stock, weight in zip(simple_stocks, max_sharpe['weights']):
    if weight > 0.01:
        print(f"  {stock:15s} {weight:6.2%}")

# ============================================================================
# EXAMPLE 2: Sector-Specific Portfolio
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 2: IT Sector Portfolio Analysis")
print("=" * 80)

it_stocks = ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM']

prices_it = fetcher.fetch_stock_data(it_stocks, start_date='2020-01-01')
returns_it = fetcher.compute_returns(prices_it)

# Run EDA
eda_it = StockEDA(prices_it, returns_it)
summary_it = eda_it.generate_summary_statistics()

print("\nIT Sector Summary:")
print(summary_it[['Annual Return', 'Annual Volatility', 'Sharpe Ratio']].to_string())

# Optimize
optimizer_it = PortfolioOptimizer(returns_it)
it_portfolio = optimizer_it.optimize_max_sharpe()

print(f"\nIT Sector Optimal Portfolio:")
print(f"  Sharpe Ratio: {it_portfolio['sharpe']:.4f}")
print(f"  Top Holdings:")
sorted_weights = sorted(zip(it_stocks, it_portfolio['weights']), 
                       key=lambda x: x[1], reverse=True)
for stock, weight in sorted_weights[:3]:
    print(f"    {stock}: {weight:.2%}")

# ============================================================================
# EXAMPLE 3: Risk-Based Portfolio Comparison
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 3: Comparing Different Risk Strategies")
print("=" * 80)

# Use default stock set
default_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC']
prices_def = fetcher.fetch_stock_data(default_stocks, start_date='2021-01-01')
returns_def = fetcher.compute_returns(prices_def)

optimizer_def = PortfolioOptimizer(returns_def)

# Strategy 1: Maximum Sharpe
max_sharpe_strat = optimizer_def.optimize_max_sharpe()

# Strategy 2: Minimum Risk
min_risk_strat = optimizer_def.optimize_min_risk()

# Strategy 3: Equal Weight
equal_weights = np.ones(len(default_stocks)) / len(default_stocks)
eq_return, eq_risk, eq_sharpe = optimizer_def.portfolio_performance(equal_weights)
equal_weight_strat = {
    'weights': equal_weights,
    'return': eq_return,
    'risk': eq_risk,
    'sharpe': eq_sharpe
}

# Compare
comparison = pd.DataFrame({
    'Strategy': ['Max Sharpe', 'Min Risk', 'Equal Weight'],
    'Return': [max_sharpe_strat['return'], min_risk_strat['return'], 
              equal_weight_strat['return']],
    'Risk': [max_sharpe_strat['risk'], min_risk_strat['risk'], 
            equal_weight_strat['risk']],
    'Sharpe': [max_sharpe_strat['sharpe'], min_risk_strat['sharpe'], 
              equal_weight_strat['sharpe']]
})

print("\nStrategy Comparison:")
print(comparison.to_string(index=False))

# ============================================================================
# EXAMPLE 4: Time Period Comparison
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 4: Bull Market vs Bear Market Performance")
print("=" * 80)

stocks_comp = ['RELIANCE', 'TCS', 'HDFCBANK']

# Bull period (example: 2020-2021 recovery)
bull_start, bull_end = '2020-04-01', '2021-12-31'
prices_bull = fetcher.fetch_stock_data(stocks_comp, bull_start, bull_end)
returns_bull = fetcher.compute_returns(prices_bull)

# Recent period (2023-2024)
recent_start, recent_end = '2023-01-01', '2024-12-31'
prices_recent = fetcher.fetch_stock_data(stocks_comp, recent_start, recent_end)
returns_recent = fetcher.compute_returns(prices_recent)

print("\nBull Period (Apr 2020 - Dec 2021):")
print(f"  Average Return: {returns_bull.mean().mean() * 252:.2%}")
print(f"  Average Volatility: {returns_bull.std().mean() * np.sqrt(252):.2%}")

print("\nRecent Period (2023-2024):")
print(f"  Average Return: {returns_recent.mean().mean() * 252:.2%}")
print(f"  Average Volatility: {returns_recent.std().mean() * np.sqrt(252):.2%}")

# ============================================================================
# EXAMPLE 5: ML Prediction and Comparison
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 5: ML-Based Portfolio Optimization")
print("=" * 80)

# Use 5-year data for ML
ml_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY']
prices_ml = fetcher.fetch_stock_data(ml_stocks, start_date='2020-01-01')
returns_ml = fetcher.compute_returns(prices_ml)

# Train ML models
print("\nTraining ML models (this may take a minute)...")
predictor = MLReturnPredictor(returns_ml, lookback_period=60)
training_results = predictor.train_all_models(model_type='random_forest')

# Get predictions
predictions = predictor.predict_next_period()

print("\nPredicted Annual Returns:")
for stock, pred in predictions.items():
    historical = returns_ml[stock].mean() * 252
    print(f"  {stock:15s} Historical: {historical:7.2%}  |  Predicted: {pred:7.2%}")

# Create ML portfolio
ml_weights = predictor.create_ml_optimized_portfolio()

# Compare with traditional optimization
optimizer_ml = PortfolioOptimizer(returns_ml)
traditional = optimizer_ml.optimize_max_sharpe()

print("\nPortfolio Comparison:")
print(f"\n  Traditional (Max Sharpe):")
print(f"    Sharpe Ratio: {traditional['sharpe']:.4f}")
print(f"    Top Holdings:")
for stock, weight in zip(ml_stocks, traditional['weights']):
    if weight > 0.05:
        print(f"      {stock}: {weight:.2%}")

ml_perf = optimizer_ml.portfolio_performance(ml_weights)
print(f"\n  ML-Based Portfolio:")
print(f"    Sharpe Ratio: {ml_perf[2]:.4f}")
print(f"    Top Holdings:")
for stock, weight in zip(ml_stocks, ml_weights):
    if weight > 0.05:
        print(f"      {stock}: {weight:.2%}")

# ============================================================================
# EXAMPLE 6: Detailed Performance Metrics
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 6: Comprehensive Performance Metrics")
print("=" * 80)

# Use maximum Sharpe portfolio
portfolio_returns = (returns_ml * traditional['weights']).sum(axis=1)

# Calculate all metrics
from utils_enhanced import (calculate_max_drawdown, calculate_sortino_ratio,
                  calculate_var_cvar, calculate_calmar_ratio)

max_dd, dd_series = calculate_max_drawdown(portfolio_returns)
sortino = calculate_sortino_ratio(portfolio_returns, risk_free_rate=0.06)
var_95, cvar_95 = calculate_var_cvar(portfolio_returns, confidence_level=0.95)
calmar = calculate_calmar_ratio(portfolio_returns, risk_free_rate=0.06)

print("\nAdvanced Risk Metrics:")
print(f"  Maximum Drawdown:     {max_dd:.2%}")
print(f"  Sortino Ratio:        {sortino:.4f}")
print(f"  Calmar Ratio:         {calmar:.4f}")
print(f"  Value at Risk (95%):  {var_95:.2%}")
print(f"  CVaR (95%):           {cvar_95:.2%}")

# Distribution metrics
print("\nReturn Distribution:")
print(f"  Skewness:             {portfolio_returns.skew():.4f}")
print(f"  Kurtosis:             {portfolio_returns.kurtosis():.4f}")
print(f"  Best Day:             {portfolio_returns.max():.2%}")
print(f"  Worst Day:            {portfolio_returns.min():.2%}")

# ============================================================================
# EXAMPLE 7: Custom Visualization
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 7: Creating Custom Visualizations")
print("=" * 80)

# Create a custom 3-panel visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Panel 1: Cumulative returns comparison
ax1 = axes[0]
for stock in ml_stocks:
    cumulative = (1 + returns_ml[stock]).cumprod()
    ax1.plot(cumulative.index, cumulative, label=stock, linewidth=2)
ax1.set_title('Cumulative Returns - Individual Stocks', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative Return')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Panel 2: Rolling volatility
ax2 = axes[1]
for stock in ml_stocks:
    rolling_vol = returns_ml[stock].rolling(60).std() * np.sqrt(252)
    ax2.plot(rolling_vol.index, rolling_vol, label=stock, linewidth=2)
ax2.set_title('60-Day Rolling Volatility', fontsize=14, fontweight='bold')
ax2.set_ylabel('Annualized Volatility')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Panel 3: Portfolio value over time
ax3 = axes[2]
portfolio_cumulative = (1 + portfolio_returns).cumprod()
equal_portfolio_returns = returns_ml.mean(axis=1)
equal_cumulative = (1 + equal_portfolio_returns).cumprod()

ax3.plot(portfolio_cumulative.index, portfolio_cumulative, 
        label='Optimized Portfolio', linewidth=2, color='blue')
ax3.plot(equal_cumulative.index, equal_cumulative,
        label='Equal-Weight Portfolio', linewidth=2, color='gray', 
        linestyle='--', alpha=0.7)
ax3.set_title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Portfolio Value (Base = 1)')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/custom_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nCustom visualization saved to: results/custom_analysis.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLES COMPLETE")
print("=" * 80)
print("\nAll example outputs have been generated successfully!")
print("Check the 'results/' directory for saved visualizations.")
print("\nKey Takeaways:")
print("  1. Portfolio optimization can significantly improve risk-adjusted returns")
print("  2. Diversification reduces portfolio volatility")
print("  3. ML predictions show promise but require careful validation")
print("  4. Different market periods require different strategies")
print("  5. Regular rebalancing is essential for maintaining optimal allocation")
print("\n" + "=" * 80)