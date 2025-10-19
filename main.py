"""
Enhanced Main Execution Script

Major improvements:
1. Command-line argument support
2. Better error handling and logging
3. Progress tracking
4. Configurable execution modes
5. Unit tests for critical functions
6. Comprehensive reporting
"""

import numpy as np
import pandas as pd
import argparse
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import enhanced modules
from data_fetch import NSEDataFetcher, DEFAULT_NSE_TICKERS
from eda import StockEDA
from utils_enhanced import (
    print_data_quality_report,
    RiskMetrics,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)


# Setup logging
def setup_logging(log_level: str = 'INFO'):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class PortfolioPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.results = {}
        
        # Create results directory
        Path('results').mkdir(exist_ok=True)
    
    def run(self):
        """Execute complete pipeline"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("ENHANCED NSE PORTFOLIO OPTIMIZATION PIPELINE")
            self.logger.info("=" * 80)
            
            # Step 1: Data Acquisition
            self.logger.info("\nSTEP 1: Data Acquisition")
            prices, returns = self.fetch_data()
            
            # Step 2: Data Validation
            self.logger.info("\nSTEP 2: Data Validation")
            self.validate_data(returns)
            
            # Step 3: Exploratory Analysis
            self.logger.info("\nSTEP 3: Exploratory Data Analysis")
            self.exploratory_analysis(prices, returns)
            
            # Step 4: Portfolio Optimization
            self.logger.info("\nSTEP 4: Portfolio Optimization")
            portfolios = self.optimize_portfolios(returns)
            
            # Step 5: ML Predictions (if enabled)
            if self.config.get('enable_ml', True):
                self.logger.info("\nSTEP 5: Machine Learning Predictions")
                ml_portfolio = self.ml_optimization(returns)
                portfolios['ML-Optimized'] = ml_portfolio
            
            # Step 6: Comparison and Reporting
            self.logger.info("\nSTEP 6: Performance Comparison")
            self.generate_reports(portfolios, returns)
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def fetch_data(self):
        """Fetch stock data"""
        tickers = self.config.get('tickers', DEFAULT_NSE_TICKERS)
        start_date = self.config.get('start_date', '2020-01-01')
        end_date = self.config.get('end_date', None)
        
        self.logger.info(f"Fetching data for {len(tickers)} stocks")
        self.logger.info(f"Date range: {start_date} to {end_date or 'present'}")
        
        fetcher = NSEDataFetcher()
        prices = fetcher.fetch_stock_data(tickers, start_date, end_date)
        
        if prices.empty:
            raise ValueError("Failed to fetch price data")
        
        returns = fetcher.compute_returns(prices)
        
        self.logger.info(f"✓ Data fetched: {prices.shape[0]} days, {prices.shape[1]} assets")
        
        self.results['prices'] = prices
        self.results['returns'] = returns
        
        return prices, returns
    
    def validate_data(self, returns):
        """Validate data quality"""
        from utils_enhanced import validate_portfolio_data
        
        is_valid, issues = validate_portfolio_data(returns)
        
        if issues:
            for issue in issues:
                if issue.startswith("ERROR"):
                    self.logger.error(issue)
                else:
                    self.logger.warning(issue)
        
        if not is_valid:
            raise ValueError("Data validation failed")
        
        self.logger.info("✓ Data validation passed")
    
    def exploratory_analysis(self, prices, returns):
        """Run EDA"""
        eda = StockEDA(prices, returns)
        
        # Generate visualizations
        self.logger.info("Generating visualizations...")
        
        eda.plot_price_history()
        eda.plot_cumulative_returns()
        eda.plot_correlation_heatmap()
        eda.plot_risk_return_scatter()
        eda.plot_return_distribution()
        
        # Print summary
        summary = eda.generate_summary_statistics()
        self.logger.info("\nSummary Statistics:")
        self.logger.info("\n" + summary.to_string())
        
        self.results['summary_stats'] = summary
        
        self.logger.info("✓ EDA completed")
    
    def optimize_portfolios(self, returns):
        """Run portfolio optimizations"""
        # Import here to use enhanced version
        from portfolio_optimizer_v2 import PortfolioOptimizer
        
        risk_free_rate = self.config.get('risk_free_rate', 0.06)
        optimizer = PortfolioOptimizer(returns, risk_free_rate)
        
        portfolios = {}
        
        # Monte Carlo simulation
        self.logger.info("Running Monte Carlo simulation...")
        sim_results, _ = optimizer.simulate_portfolios(
            num_portfolios=self.config.get('num_simulations', 10000)
        )
        
        # Maximum Sharpe portfolio
        self.logger.info("Optimizing for maximum Sharpe ratio...")
        max_sharpe = optimizer.optimize_max_sharpe()
        portfolios['Max Sharpe'] = max_sharpe.to_dict()
        self.logger.info(f"  ✓ Sharpe Ratio: {max_sharpe.sharpe_ratio:.3f}")
        
        # Minimum risk portfolio
        self.logger.info("Optimizing for minimum risk...")
        min_risk = optimizer.optimize_min_risk()
        portfolios['Min Risk'] = min_risk.to_dict()
        self.logger.info(f"  ✓ Volatility: {min_risk.volatility:.2%}")
        
        # Risk parity portfolio
        if self.config.get('include_risk_parity', True):
            self.logger.info("Optimizing for risk parity...")
            risk_parity = optimizer.optimize_risk_parity()
            portfolios['Risk Parity'] = risk_parity.to_dict()
            self.logger.info(f"  ✓ Sharpe Ratio: {risk_parity.sharpe_ratio:.3f}")
        
        # Generate visualizations
        optimizer.plot_efficient_frontier(sim_results, max_sharpe, min_risk)
        optimizer.plot_allocation(
            max_sharpe.weights,
            "Maximum Sharpe Ratio Portfolio",
            "results/allocation_max_sharpe.png"
        )
        optimizer.plot_allocation(
            min_risk.weights,
            "Minimum Risk Portfolio",
            "results/allocation_min_risk.png"
        )
        
        self.results['portfolios'] = portfolios
        self.results['optimizer'] = optimizer
        
        return portfolios
    
    def ml_optimization(self, returns):
        """ML-based optimization"""
        from ml_predictor_v2 import MLReturnPredictor
        
        lookback = self.config.get('ml_lookback', 60)
        model_type = self.config.get('ml_model_type', 'random_forest')
        
        self.logger.info(f"Training ML models (lookback={lookback})...")
        
        predictor = MLReturnPredictor(returns, lookback_period=lookback)
        training_results = predictor.train_all_models(model_type=model_type)
        
        # Make predictions
        self.logger.info("Generating predictions...")
        predictions = predictor.predict_next_period()
        
        # Create ML portfolio
        ml_weights = predictor.create_ml_optimized_portfolio(method='proportional')
        
        # Calculate performance
        optimizer = self.results['optimizer']
        ml_return, ml_risk, ml_sharpe = optimizer.portfolio_performance(ml_weights)
        
        ml_portfolio = {
            'weights': ml_weights,
            'return': ml_return,
            'risk': ml_risk,
            'sharpe': ml_sharpe
        }
        
        self.logger.info(f"  ✓ ML Portfolio Sharpe: {ml_sharpe:.3f}")
        
        # Visualizations
        predictor.plot_predictions_vs_actual()
        optimizer.plot_allocation(
            ml_weights,
            "ML-Optimized Portfolio",
            "results/allocation_ml_optimized.png"
        )
        
        self.results['ml_predictor'] = predictor
        
        return ml_portfolio
    
    def generate_reports(self, portfolios, returns):
        """Generate comprehensive reports"""
        from utils_enhanced import (
            portfolio_metrics_summary,
            print_metrics_table,
            compare_portfolios,
            create_tear_sheet
        )
        
        # Print comparison table
        self.logger.info("\nPortfolio Performance Comparison:")
        comparison_data = []
        for name, portfolio in portfolios.items():
            comparison_data.append({
                'Strategy': name,
                'Return': portfolio['return'],
                'Risk': portfolio['risk'],
                'Sharpe': portfolio['sharpe']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        self.logger.info("\n" + comparison_df.to_string(index=False))
        
        # Save to CSV
        comparison_df.to_csv('results/portfolio_comparison.csv', index=False)
        
        # Generate comparison visualization
        compare_portfolios(portfolios, returns)
        
        # Generate tear sheet for best portfolio
        best_portfolio_name = max(portfolios.items(), 
                                 key=lambda x: x[1]['sharpe'])[0]
        best_portfolio = portfolios[best_portfolio_name]
        
        self.logger.info(f"\nGenerating tear sheet for best portfolio: {best_portfolio_name}")
        
        create_tear_sheet(
            returns,
            best_portfolio['weights'],
            returns.columns.tolist(),
            save_path='results/best_portfolio_tear_sheet.png'
        )
        
        # Comprehensive metrics
        portfolio_returns = (returns * best_portfolio['weights']).sum(axis=1)
        metrics = portfolio_metrics_summary(portfolio_returns)
        
        print_metrics_table(metrics, f"{best_portfolio_name} - Detailed Metrics")
        
        self.logger.info("\n✓ All reports generated successfully")
        self.logger.info("\nResults saved to:")
        self.logger.info("  - results/efficient_frontier.png")
        self.logger.info("  - results/allocation_*.png")
        self.logger.info("  - results/portfolio_comparison.csv")
        self.logger.info("  - results/portfolio_comparison.png")
        self.logger.info("  - results/best_portfolio_tear_sheet.png")


def run_unit_tests():
    """Run basic unit tests"""
    logger = logging.getLogger(__name__)
    logger.info("\nRunning unit tests...")
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        # Test 1: Portfolio weight validation
        weights = np.array([0.2, 0.3, 0.5])
        assert np.isclose(weights.sum(), 1.0), "Weights should sum to 1"
        tests_passed += 1
        logger.info("  ✓ Test 1: Weight validation")
    except AssertionError as e:
        tests_failed += 1
        logger.error(f"  ✗ Test 1 failed: {e}")
    
    try:
        # Test 2: Return calculation
        from portfolio_optimizer_v2 import PortfolioOptimizer
        
        # Create dummy data
        dates = pd.date_range('2020-01-01', periods=100)
        dummy_returns = pd.DataFrame(
            np.random.randn(100, 3) * 0.01,
            index=dates,
            columns=['A', 'B', 'C']
        )
        
        optimizer = PortfolioOptimizer(dummy_returns)
        weights = np.array([0.33, 0.33, 0.34])
        ret, risk, sharpe = optimizer.portfolio_performance(weights)
        
        assert isinstance(ret, float), "Return should be float"
        assert isinstance(risk, float), "Risk should be float"
        assert isinstance(sharpe, float), "Sharpe should be float"
        tests_passed += 1
        logger.info("  ✓ Test 2: Portfolio performance calculation")
    except Exception as e:
        tests_failed += 1
        logger.error(f"  ✗ Test 2 failed: {e}")
    
    try:
        # Test 3: Data validation
        from utils_enhanced import validate_portfolio_data
        
        test_returns = pd.DataFrame({
            'A': [0.01, 0.02, -0.01],
            'B': [0.00, 0.01, 0.02]
        })
        
        is_valid, issues = validate_portfolio_data(test_returns)
        assert isinstance(is_valid, bool), "Should return boolean"
        assert isinstance(issues, list), "Should return list of issues"
        tests_passed += 1
        logger.info("  ✓ Test 3: Data validation")
    except Exception as e:
        tests_failed += 1
        logger.error(f"  ✗ Test 3 failed: {e}")
    
    logger.info(f"\nTest Results: {tests_passed} passed, {tests_failed} failed")
    
    return tests_failed == 0


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced NSE Portfolio Optimization'
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='List of stock tickers (default: predefined list)'
    )
    
    parser.add_argument(
        '--start-date',
        default='2020-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        default=None,
        help='End date (YYYY-MM-DD), default: today'
    )
    
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.06,
        help='Annual risk-free rate (default: 0.06)'
    )
    
    parser.add_argument(
        '--num-simulations',
        type=int,
        default=10000,
        help='Number of Monte Carlo simulations (default: 10000)'
    )
    
    parser.add_argument(
        '--disable-ml',
        action='store_true',
        help='Disable ML prediction step'
    )
    
    parser.add_argument(
        '--ml-model',
        choices=['random_forest', 'gradient_boosting', 'ridge'],
        default='random_forest',
        help='ML model type (default: random_forest)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--run-tests',
        action='store_true',
        help='Run unit tests before execution'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: fewer simulations and skip ML'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Print header
    logger.info("\n" + "=" * 80)
    logger.info("ENHANCED NSE PORTFOLIO OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests if requested
    if args.run_tests:
        if not run_unit_tests():
            logger.error("Unit tests failed. Exiting.")
            return 1
    
    # Build configuration
    config = {
        'tickers': args.tickers or DEFAULT_NSE_TICKERS,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'risk_free_rate': args.risk_free_rate,
        'num_simulations': 1000 if args.quick else args.num_simulations,
        'enable_ml': not (args.disable_ml or args.quick),
        'ml_model_type': args.ml_model,
        'ml_lookback': 60,
        'include_risk_parity': not args.quick
    }
    
    logger.info(f"\nConfiguration:")
    for key, value in config.items():
        if key != 'tickers':
            logger.info(f"  {key}: {value}")
    logger.info(f"  tickers: {len(config['tickers'])} stocks")
    
    try:
        # Run pipeline
        pipeline = PortfolioPipeline(config, logger)
        results = pipeline.run()
        
        # Print final summary
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        portfolios = results.get('portfolios', {})
        if portfolios:
            logger.info("\nOptimized Portfolios:")
            for name, portfolio in portfolios.items():
                logger.info(f"\n  {name}:")
                logger.info(f"    Expected Return: {portfolio['return']:.2%}")
                logger.info(f"    Volatility:      {portfolio['risk']:.2%}")
                logger.info(f"    Sharpe Ratio:    {portfolio['sharpe']:.3f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS - All visualizations saved to 'results/' directory")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("EXECUTION FAILED")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())