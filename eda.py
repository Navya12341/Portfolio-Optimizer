"""
Exploratory Data Analysis Module
Visualization and statistical analysis of stock data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class StockEDA:
    """Exploratory data analysis for stock data"""
    
    def __init__(self, prices, returns):
        """
        Initialize EDA module
        
        Args:
            prices (pd.DataFrame): Price data
            returns (pd.DataFrame): Returns data
        """
        self.prices = prices
        self.returns = returns
        self.tickers = prices.columns.tolist()
    
    def plot_price_history(self, save_path='results/price_history.png'):
        """
        Plot normalized price history
        
        Args:
            save_path (str): Path to save plot
        """
        # Normalize prices to start at 100
        normalized_prices = self.prices / self.prices.iloc[0] * 100
        
        plt.figure(figsize=(14, 7))
        for col in normalized_prices.columns:
            plt.plot(normalized_prices.index, normalized_prices[col], label=col, linewidth=2)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Price (Base = 100)', fontsize=12)
        plt.title('Historical Price Performance (Normalized)', fontsize=14, fontweight='bold')
        plt.legend(loc='best', ncol=2, fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cumulative returns saved to {save_path}")
    
    def plot_correlation_heatmap(self, save_path='results/correlation_heatmap.png'):
        """
        Plot correlation heatmap of returns
        
        Args:
            save_path (str): Path to save plot
        """
        correlation_matrix = self.returns.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    linewidths=1,
                    cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Correlation Matrix of Stock Returns', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation heatmap saved to {save_path}")
    
    def plot_risk_return_scatter(self, save_path='results/risk_return_scatter.png'):
        """
        Plot risk-return scatter for individual stocks
        
        Args:
            save_path (str): Path to save plot
        """
        # Calculate annualized metrics
        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        
        plt.figure(figsize=(12, 8))
        
        for i, ticker in enumerate(self.tickers):
            plt.scatter(annual_volatility[ticker], 
                       annual_returns[ticker],
                       s=200,
                       alpha=0.7,
                       label=ticker)
            
            # Add labels
            plt.annotate(ticker,
                        (annual_volatility[ticker], annual_returns[ticker]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9)
        
        plt.xlabel('Annual Volatility (Risk)', fontsize=12)
        plt.ylabel('Annual Return', fontsize=12)
        plt.title('Risk-Return Profile of Individual Stocks', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Format axes as percentage
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Risk-return scatter saved to {save_path}")
    
    def plot_return_distribution(self, save_path='results/return_distribution.png'):
        """
        Plot distribution of returns for all stocks
        
        Args:
            save_path (str): Path to save plot
        """
        n_stocks = len(self.tickers)
        n_cols = 3
        n_rows = (n_stocks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_stocks > 1 else [axes]
        
        for i, ticker in enumerate(self.tickers):
            ax = axes[i]
            returns_data = self.returns[ticker].dropna()
            
            # Histogram
            ax.hist(returns_data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Add normal distribution overlay
            mu, sigma = returns_data.mean(), returns_data.std()
            x = np.linspace(returns_data.min(), returns_data.max(), 100)
            ax2 = ax.twinx()
            ax2.plot(x, stats.norm.pdf(x, mu, sigma) * len(returns_data) * 
                    (returns_data.max() - returns_data.min()) / 50, 
                    'r-', linewidth=2, label='Normal Dist')
            ax2.set_ylabel('Probability Density', fontsize=8)
            ax2.tick_params(labelsize=8)
            
            ax.set_title(f'{ticker} Daily Returns', fontsize=10, fontweight='bold')
            ax.set_xlabel('Return', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Hide empty subplots
        for i in range(n_stocks, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Return distribution saved to {save_path}")
    
    def plot_cumulative_returns(self):
        """Plot cumulative returns for all stocks"""
        try:
            # Calculate cumulative returns
            cumulative_returns = (1 + self.returns).cumprod()
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot each stock's cumulative returns
            for column in cumulative_returns.columns:
                plt.plot(cumulative_returns.index, 
                        cumulative_returns[column], 
                        label=column, 
                        linewidth=1.5)
            
            # Customize plot
            plt.title('Cumulative Returns Over Time', 
                     fontsize=14, 
                     fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return (1 + return)', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), 
                      loc='upper left', 
                      borderaxespad=0.)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            save_path = 'results/cumulative_returns.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Cumulative returns saved to {save_path}")
            
        except Exception as e:
            print(f"Error plotting cumulative returns: {str(e)}")
        finally:
            plt.close()  # Ensure figure is closed even if error occurs
    
    def generate_summary_statistics(self):
        """
        Generate summary statistics table
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        # Calculate metrics
        annual_return = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility
        
        # Skewness and Kurtosis
        skewness = self.returns.skew()
        kurtosis = self.returns.kurtosis()
        
        # Max drawdown
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (95%)
        var_95 = self.returns.quantile(0.05)
        
        # Create summary DataFrame
        summary = pd.DataFrame({
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'VaR (95%)': var_95
        })
        
        return summary
    
    def print_summary(self):
        """Print summary statistics to console"""
        summary = self.generate_summary_statistics()
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(summary.to_string())
        print("\nNote: All returns and volatilities are annualized")
        print("Sharpe Ratio assumes 6% risk-free rate (typical for India)")
        print("="*80)
        
        return summary


if __name__ == "__main__":
    print("EDA module loaded successfully")
    plt.savefig('eda_plot.png', bbox_inches='tight')