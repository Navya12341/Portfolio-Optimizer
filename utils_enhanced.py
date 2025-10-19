"""
Enhanced Utilities Module for Portfolio Analysis
"""

import os
from venv import logger
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Union, List
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class RiskMetrics:
    """Comprehensive risk metric calculations"""
    
    

    @staticmethod
    def calculate_var_cvar(returns: pd.Series, 
                          confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Find the index for VaR
        index = int((1 - confidence_level) * len(sorted_returns))
        
        # Calculate VaR
        var = -sorted_returns[index]
        
        # Calculate CVaR
        cvar = -sorted_returns[:index].mean()
        
        return var, cvar
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Series, Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Calculate maximum drawdown and drawdown series
        
        Args:
            returns: Series of returns
            
        Returns:
            Tuple of (max_drawdown, drawdown_series, (peak_date, valley_date))
        """
        # Calculate cumulative returns
        cum_rets = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_rets.expanding().max()
        
        # Calculate drawdown series
        drawdown = cum_rets / running_max - 1
        
        # Find maximum drawdown and its corresponding timestamps
        max_dd = drawdown.min()
        valley_idx = drawdown.idxmin()
        
        # Find the peak before the valley
        peak_idx = running_max.loc[:valley_idx].idxmax()
        
        return max_dd, drawdown, (peak_idx, valley_idx)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """
        Calculate Sortino ratio (return / downside deviation)
        
        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
        """
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
            
        downside_std = np.sqrt(np.mean(downside_returns**2))
        expected_return = returns.mean() * 252
        
        return (expected_return - risk_free_rate) / (downside_std * np.sqrt(252))

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """
        Calculate Calmar ratio (return / max drawdown)
        """
        max_dd = RiskMetrics.calculate_max_drawdown(returns)[0]
        if max_dd == 0:
            return np.inf
            
        expected_return = returns.mean() * 252
        return (expected_return - risk_free_rate) / abs(max_dd)
    
    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio (probability-weighted ratio of gains vs. losses)
        
        Args:
            returns: Return series
            threshold: Return threshold (default 0)
        """
        excess_returns = returns - threshold/252
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns < 0].sum())
        
        if negative_returns == 0:
            return np.inf
            
        return positive_returns / negative_returns
    
    @staticmethod
    def calculate_tail_ratio(returns: pd.Series) -> float:
        """
        Calculate tail ratio (95th percentile / 5th percentile)
        Measures asymmetry of return distribution
        """
        if returns.empty:
            return np.nan
        
        p95 = returns.quantile(0.95)
        p5 = returns.quantile(0.05)
        
        if p5 == 0:
            return np.inf
        
        return abs(p95 / p5)
    
    @staticmethod
    def calculate_skewness_kurtosis(returns: pd.Series) -> Tuple[float, float]:
        """Calculate skewness and excess kurtosis"""
        if returns.empty or len(returns) < 3:
            return np.nan, np.nan
        
        skew = returns.skew()
        kurt = returns.kurtosis()  # Excess kurtosis (Fisher's definition)
        
        return skew, kurt
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio (active return / tracking error)"""
        if portfolio_returns.empty or benchmark_returns.empty:
            return np.nan
        
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        
        if tracking_error == 0:
            return np.inf
        
        return (active_returns.mean() * 252) / tracking_error
    
    @staticmethod
    def calculate_beta_alpha(portfolio_returns: pd.Series,
                            market_returns: pd.Series,
                            risk_free_rate: float = 0.06) -> Tuple[float, float]:
        """Calculate portfolio beta and Jensen's alpha"""
        if portfolio_returns.empty or market_returns.empty:
            return np.nan, np.nan
        
        # Calculate beta (covariance / market variance)
        covariance = portfolio_returns.cov(market_returns)
        market_variance = market_returns.var()
        
        if market_variance == 0:
            return np.nan, np.nan
        
        beta = covariance / market_variance
        
        # Calculate alpha
        portfolio_return = portfolio_returns.mean() * 252
        market_return = market_returns.mean() * 252
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        alpha = portfolio_return - expected_return
        
        return beta, alpha


class PerformanceAnalytics:
    """Advanced performance analytics"""
    
    @staticmethod
    def rolling_metrics(returns: pd.Series, 
                       window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics
        
        Args:
            returns: Return series
            window: Rolling window size (default 252 days)
            
        Returns:
            DataFrame with rolling metrics
        """
        metrics = pd.DataFrame(index=returns.index)
        
        # Rolling return
        metrics['rolling_return'] = returns.rolling(window).mean() * 252
        
        # Rolling volatility
        metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe (assuming 6% risk-free rate)
        risk_free_rate = 0.06
        metrics['rolling_sharpe'] = (
            (metrics['rolling_return'] - risk_free_rate) / 
            metrics['rolling_volatility']
        )
        
        # Rolling max drawdown
        def rolling_max_dd(x):
            cum_ret = (1 + x).cumprod()
            running_max = cum_ret.expanding().max()
            dd = (cum_ret - running_max) / running_max
            return dd.min()
        
        metrics['rolling_max_dd'] = returns.rolling(window).apply(rolling_max_dd)
        
        return metrics
    
    @staticmethod
    def performance_attribution(portfolio_returns: pd.Series,
                               asset_returns: pd.DataFrame,
                               weights: np.ndarray) -> pd.DataFrame:
        """
        Calculate contribution of each asset to portfolio performance
        
        Returns:
            DataFrame with attribution analysis
        """
        # Calculate contribution to total return
        total_return = portfolio_returns.sum()
        asset_contributions = (asset_returns * weights).sum()
        
        attribution = pd.DataFrame({
            'Weight': weights,
            'Asset Return': asset_returns.sum(),
            'Contribution': asset_contributions,
            'Contribution %': (asset_contributions / total_return * 100)
        }, index=asset_returns.columns)
        
        return attribution.sort_values('Contribution', ascending=False)
    
    @staticmethod
    def statistical_tests(returns: pd.Series) -> Dict[str, Dict]:
        """
        Perform statistical tests on returns
        
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Normality test (Jarque-Bera)
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
            results['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_pvalue,
                'is_normal': jb_pvalue > 0.05
            }
        except:
            results['jarque_bera'] = {'error': 'Test failed'}
        
        # Autocorrelation test (Ljung-Box)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(returns.dropna(), lags=[10], return_df=True)
            results['ljung_box'] = {
                'statistic': lb_result['lb_stat'].values[0],
                'p_value': lb_result['lb_pvalue'].values[0],
                'has_autocorrelation': lb_result['lb_pvalue'].values[0] < 0.05
            }
        except:
            results['ljung_box'] = {'note': 'statsmodels not available'}
        
        # Stationarity test (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(returns.dropna())
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            }
        except:
            results['adf'] = {'note': 'statsmodels not available'}
        
        return results

def check_portfolio_diversification(weights: np.ndarray, 
                                   asset_names: List[str], 
                                   logger):
    """Check and warn about concentration risk"""
    logger.info("\n" + "=" * 80)
    logger.info("DIVERSIFICATION ANALYSIS")
    logger.info("=" * 80)

    # Concentration metrics
    max_weight = weights.max()
    max_asset = asset_names[weights.argmax()]
    top_3_weight = np.sort(weights)[-3:].sum()

    # Herfindahl-Hirschman Index (HHI)
    hhi = (weights ** 2).sum()

    logger.info(f"\n  Largest Position:     {max_weight:.2%} ({max_asset})")
    logger.info(f"  Top 3 Positions:      {top_3_weight:.2%}")
    logger.info(f"  HHI Score:            {hhi:.4f}")

    # Effective number of assets
    n_effective = 1 / hhi if hhi > 0 else 0
    logger.info(f"  Effective # Assets:   {n_effective:.1f} (out of {len(weights)})")

    # Warnings
    warnings_issued = False

    if max_weight > 0.40:
        logger.warning(f"\n⚠️  HIGH CONCENTRATION RISK!")
        logger.warning(f"  Largest position ({max_asset}) is {max_weight:.2%}")
        logger.warning(f"  Recommendation: Consider capping at 30-35%")
        warnings_issued = True

    if top_3_weight > 0.70:
        logger.warning(f"\n⚠️  Top 3 positions represent {top_3_weight:.2%} of portfolio")
        logger.warning(f"  Consider spreading risk across more assets")
        warnings_issued = True

    if n_effective < len(weights) / 2:
        logger.warning(f"\n⚠️  Low effective diversification ({n_effective:.1f} assets)")
        logger.warning(f"  Many assets have negligible weights")
        warnings_issued = True

    # Good diversification
    if not warnings_issued:
        if max_weight <= 0.30 and n_effective >= len(weights) * 0.6:
            logger.info(f"\n✅ Good diversification across {n_effective:.1f} effective assets")
        else:
            logger.info(f"\n✓ Acceptable diversification level")

    logger.info("\n" + "=" * 80)
    
def portfolio_metrics_summary(returns: Union[pd.Series, pd.DataFrame],
                             weights: Optional[np.ndarray] = None,
                             benchmark_returns: Optional[pd.Series] = None,
                             risk_free_rate: float = 0.06) -> Dict:
    """
    Calculate comprehensive portfolio metrics
    
    Args:
        returns: Portfolio returns (Series) or asset returns (DataFrame)
        weights: Portfolio weights (if returns is DataFrame)
        benchmark_returns: Benchmark returns for comparison
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of portfolio metrics
    """
    # Calculate portfolio returns if needed
    if isinstance(returns, pd.DataFrame):
        if weights is None:
            raise ValueError("Weights required when returns is DataFrame")
        portfolio_returns = (returns * weights).sum(axis=1)
    else:
        portfolio_returns = returns
    
    # Basic metrics
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
    
    # Risk metrics
    var_95, cvar_95 = RiskMetrics.calculate_var_cvar(portfolio_returns, 0.95)
    var_99, cvar_99 = RiskMetrics.calculate_var_cvar(portfolio_returns, 0.99)
    max_dd, _, _ = RiskMetrics.calculate_max_drawdown(portfolio_returns)
    sortino = RiskMetrics.calculate_sortino_ratio(portfolio_returns, risk_free_rate)
    calmar = RiskMetrics.calculate_calmar_ratio(portfolio_returns, risk_free_rate)
    omega = RiskMetrics.calculate_omega_ratio(portfolio_returns)
    tail_ratio = RiskMetrics.calculate_tail_ratio(portfolio_returns)
    
    # Distribution metrics
    skewness, kurtosis = RiskMetrics.calculate_skewness_kurtosis(portfolio_returns)
    
    # Compile metrics
    metrics = {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Omega Ratio': omega,
        'Max Drawdown': max_dd,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'VaR (99%)': var_99,
        'CVaR (99%)': cvar_99,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Tail Ratio': tail_ratio,
        'Best Day': portfolio_returns.max(),
        'Worst Day': portfolio_returns.min(),
        'Positive Days %': (portfolio_returns > 0).mean() * 100
    }
    
    # Add benchmark comparison if provided
    if benchmark_returns is not None:
        beta, alpha = RiskMetrics.calculate_beta_alpha(
            portfolio_returns, benchmark_returns, risk_free_rate
        )
        info_ratio = RiskMetrics.calculate_information_ratio(
            portfolio_returns, benchmark_returns
        )
        
        metrics['Beta'] = beta
        metrics['Alpha'] = alpha
        metrics['Information Ratio'] = info_ratio
    
    return metrics


def print_metrics_table(metrics_dict: Dict, title: str = "Portfolio Metrics"):
    """Print formatted metrics table"""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    for metric, value in metrics_dict.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            # Format based on metric type
            if any(x in metric for x in ['Return', 'Volatility', 'Drawdown', 
                                         'VaR', 'CVaR', 'Alpha']):
                print(f"{metric:30s} {value:>12.2%}")
            elif 'Days %' in metric:
                print(f"{metric:30s} {value:>12.1f}%")
            elif 'Ratio' in metric and np.isinf(value):
                print(f"{metric:30s} {'    Infinite':>12}")
            else:
                print(f"{metric:30s} {value:>12.4f}")
        else:
            print(f"{metric:30s} {str(value):>12}")
    
    print(f"{'='*70}\n")


def create_tear_sheet(returns: pd.DataFrame, 
                     weights: np.ndarray, 
                     tickers: List[str],
                     save_path: str = 'results/tear_sheet.png'):
    """Create a comprehensive portfolio tear sheet"""
    # Calculate portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Calculate drawdown
    max_dd, drawdown_series, (peak_date, valley_date) = RiskMetrics.calculate_max_drawdown(portfolio_returns)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2)
    
    # Plot cumulative returns
    ax1 = fig.add_subplot(gs[0, :])
    cum_returns = (1 + portfolio_returns).cumprod()
    ax1.plot(cum_returns.index, cum_returns, 'b-')
    ax1.set_title('Cumulative Returns')
    ax1.grid(True)
    
    # Plot drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(drawdown_series.index, drawdown_series.values, 0, color='red', alpha=0.3)
    ax2.set_title(f'Drawdown (Max: {max_dd:.1%})')
    ax2.grid(True)
    
    # Plot return distribution
    ax3 = fig.add_subplot(gs[1, 1])
    portfolio_returns.hist(bins=50, ax=ax3, density=True)
    ax3.set_title('Return Distribution')
    
    # Plot allocation
    ax4 = fig.add_subplot(gs[2, :])
    colors = plt.cm.viridis(np.linspace(0, 1, len(tickers)))
    ax4.bar(tickers, weights * 100, color=colors)
    ax4.set_title('Portfolio Allocation')
    plt.xticks(rotation=45)
    
    # Add performance metrics
    metrics = {
        'Annual Return': portfolio_returns.mean() * 252,
        'Annual Volatility': portfolio_returns.std() * np.sqrt(252),
        'Sharpe Ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
        'Max Drawdown': max_dd,
        'Skewness': portfolio_returns.skew(),
        'Kurtosis': portfolio_returns.kurtosis()
    }
    
    # Add metrics text box
    metrics_text = '\n'.join([f"{k}: {v:.2%}" if k != 'Sharpe Ratio' else f"{k}: {v:.2f}"
                             for k, v in metrics.items()])
    fig.text(0.02, 0.02, metrics_text, fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics


def validate_portfolio_data(returns: pd.DataFrame, 
                           weights: Optional[np.ndarray] = None) -> Tuple[bool, List[str]]:
    """
    Validate portfolio data for common issues
    
    Returns:
        Tuple of (is_valid, list of warnings/errors)
    """
    issues = []
    is_valid = True
    
    # Check for empty data
    if returns.empty:
        issues.append("ERROR: Returns DataFrame is empty")
        return False, issues
    
    # Check for NaN values
    if returns.isnull().any().any():
        nan_cols = returns.columns[returns.isnull().any()].tolist()
        issues.append(f"WARNING: NaN values found in columns: {nan_cols}")
    
    # Check for infinite values
    if np.isinf(returns.values).any():
        issues.append("WARNING: Infinite values detected in returns")
    
    # Check for sufficient data
    if len(returns) < 100:
        issues.append(f"WARNING: Only {len(returns)} data points. Recommend at least 100.")
    
    # Check for extreme values (potential data errors)
    extreme_threshold = 0.5  # 50% daily return
    extreme_returns = (returns.abs() > extreme_threshold).any(axis=1)
    if extreme_returns.any():
        n_extreme = extreme_returns.sum()
        issues.append(f"WARNING: {n_extreme} days with extreme returns (>50%). Check data quality.")
    
    # Validate weights if provided
    if weights is not None:
        if len(weights) != len(returns.columns):
            issues.append(f"ERROR: Weights length ({len(weights)}) != number of assets ({len(returns.columns)})")
            is_valid = False
        
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            issues.append(f"WARNING: Weights sum to {weights.sum():.6f}, not 1.0")
        
        if (weights < 0).any():
            issues.append("WARNING: Negative weights detected (short positions)")
        
        if (weights > 1).any():
            issues.append("WARNING: Weights > 100% detected")
    
    # Check for constant returns (no variation)
    constant_cols = returns.columns[returns.std() < 1e-10].tolist()
    if constant_cols:
        issues.append(f"WARNING: Constant returns (no variation) in: {constant_cols}")
    
    return is_valid and len([i for i in issues if i.startswith("ERROR")]) == 0, issues


def compare_portfolios(portfolios: Dict[str, Dict],
                      returns: pd.DataFrame,
                      save_path: str = 'results/portfolio_comparison.png'):
    """
    Compare multiple portfolios side by side
    
    Args:
        portfolios: Dict of {name: {'weights': array, 'return': float, 'risk': float, 'sharpe': float}}
        returns: Asset returns DataFrame
        save_path: Path to save comparison plot
    """
    # Calculate portfolio returns for each strategy
    portfolio_returns = {}
    for name, portfolio in portfolios.items():
        weights = portfolio['weights']
        portfolio_returns[name] = (returns * weights).sum(axis=1)
    
    returns_df = pd.DataFrame(portfolio_returns)
    cumulative_returns = (1 + returns_df).cumprod()
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    for col in cumulative_returns.columns:
        ax1.plot(cumulative_returns.index, cumulative_returns[col], 
                linewidth=2, label=col)
    ax1.set_title('Cumulative Returns Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Risk-Return Scatter
    ax2 = axes[0, 1]
    for name, portfolio in portfolios.items():
        ax2.scatter(portfolio['risk'], portfolio['return'], 
                   s=200, alpha=0.7, label=name)
        ax2.annotate(name, (portfolio['risk'], portfolio['return']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax2.set_title('Risk-Return Profile', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Annual Volatility', fontsize=11)
    ax2.set_ylabel('Annual Return', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))
    
    # 3. Rolling Sharpe Comparison
    ax3 = axes[1, 0]
    for name, ret_series in portfolio_returns.items():
        rolling_sharpe = (
            (ret_series.rolling(252).mean() * 252 - 0.06) /
            (ret_series.rolling(252).std() * np.sqrt(252))
        )
        ax3.plot(rolling_sharpe.index, rolling_sharpe, 
                linewidth=2, label=name, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_title('Rolling 1-Year Sharpe Ratio', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics Comparison Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create comparison table
    comparison_data = []
    for name, portfolio in portfolios.items():
        comparison_data.append([
            name,
            f"{portfolio['return']:.2%}",
            f"{portfolio['risk']:.2%}",
            f"{portfolio['sharpe']:.3f}"
        ])
    
    table = ax4.table(
        cellText=comparison_data,
        colLabels=['Strategy', 'Return', 'Risk', 'Sharpe'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0.3, 1, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Performance Metrics Comparison', 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('Multi-Portfolio Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Portfolio comparison saved to {save_path}")


def print_data_quality_report(returns: pd.DataFrame, logger: logging.Logger):
    """Print comprehensive data quality report"""
    logger.info("\n" + "╔" + "═" * 78 + "╗")
    logger.info("║" + "DATA QUALITY REPORT".center(78) + "║")
    logger.info("╠" + "═" * 78 + "╣")
    
    # Basic stats
    logger.info("║ Dataset Overview" + " " * 63 + "║")
    logger.info("║" + "─" * 78 + "║")
    logger.info(f"║  Time Period     : {returns.index[0].date()} to {returns.index[-1].date()}" + " " * 25 + "║")
    logger.info(f"║  Trading Days    : {len(returns):,d}" + " " * 52 + "║")
    logger.info(f"║  Number of Assets: {len(returns.columns)}" + " " * 52 + "║")
    logger.info(f"║  Total Points    : {returns.size:,d}" + " " * 52 + "║")
    
    # Missing data analysis
    missing = returns.isnull().sum()
    logger.info("╟" + "─" * 78 + "╢")
    logger.info("║ Missing Data Analysis" + " " * 58 + "║")
    logger.info("║" + "─" * 78 + "║")
    
    if missing.any():
        for col in missing[missing > 0].index:
            pct = missing[col] / len(returns) * 100
            logger.info(f"║  ⚠ {col:12s}: {missing[col]:>4} days ({pct:>5.1f}%)" + " " * 42 + "║")
    else:
        logger.info("║  ✓ No missing values detected" + " " * 51 + "║")
    
    # Extreme values analysis
    extreme_threshold = 0.15  # 15% daily move
    extreme = (returns.abs() > extreme_threshold).sum()
    logger.info("╟" + "─" * 78 + "╢")
    logger.info("║ Extreme Returns Analysis" + " " * 56 + "║")
    logger.info("║" + "─" * 78 + "║")
    
    if extreme.any():
        for col in extreme[extreme > 0].index:
            max_move = returns[col].abs().max() * 100
            logger.info(f"║  ! {col:12s}: {extreme[col]:>4} days, max move: {max_move:>6.1f}%" + " " * 35 + "║")
    else:
        logger.info("║  ✓ No extreme returns detected" + " " * 50 + "║")
    
    # Correlation analysis
    corr_matrix = returns.corr()
    high_corr_threshold = 0.8
    high_corr = (corr_matrix > high_corr_threshold) & (corr_matrix < 1.0)
    
    logger.info("╟" + "─" * 78 + "╢")
    logger.info("║ Correlation Analysis" + " " * 60 + "║")
    logger.info("║" + "─" * 78 + "║")
    
    if high_corr.any().any():
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if high_corr.iloc[i, j]:
                    corr = corr_matrix.iloc[i, j]
                    asset1 = corr_matrix.index[i]
                    asset2 = corr_matrix.columns[j]
                    logger.info(f"║  • {asset1:12s} - {asset2:12s}: {corr:>6.3f}" + " " * 35 + "║")
    else:
        logger.info("║  ✓ No high correlations detected" + " " * 48 + "║")
    
    # Data quality score
    quality_score = 100
    if missing.any():
        quality_score -= 20 * (missing.sum() / returns.size)
    if extreme.any():
        quality_score -= 10 * (extreme.sum() / returns.size)
    if high_corr.any().any():
        quality_score -= 5
    
    logger.info("╟" + "─" * 78 + "╢")
    logger.info("║ Overall Data Quality Score" + " " * 55 + "║")
    logger.info("║" + "─" * 78 + "║")
    logger.info(f"║  Quality Score: {quality_score:>5.1f}/100" + " " * 54 + "║")
    
    logger.info("╚" + "═" * 78 + "╝")

# Quick helper functions
def annualize_returns(returns: Union[pd.Series, float], 
                     periods_per_year: int = 252) -> Union[pd.Series, float]:
    """Annualize returns"""
    if isinstance(returns, pd.Series):
        return returns.mean() * periods_per_year
    return returns * periods_per_year


def annualize_volatility(returns: pd.Series, 
                        periods_per_year: int = 252) -> float:
    """Annualize volatility"""
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.06) -> float:
    """Calculate Sharpe ratio"""
    annual_return = annualize_returns(returns)
    annual_vol = annualize_volatility(returns)
    
    if annual_vol == 0:
        return np.inf
    
    return (annual_return - risk_free_rate) / annual_vol


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate the maximum drawdown from a series of returns
    
    Args:
        returns: Series of returns
        
    Returns:
        float: Maximum drawdown value as a positive number
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.expanding().max()
    
    # Calculate drawdown
    drawdowns = cum_returns / running_max - 1
    
    # Get maximum drawdown (make positive)
    max_drawdown = abs(drawdowns.min())
    
    return max_drawdown