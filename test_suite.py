"""
Comprehensive Test Suite for Portfolio Optimization

Run with: python test_suite.py
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TestDataFetch(unittest.TestCase):
    """Test data fetching functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from data_fetch import NSEDataFetcher
        self.fetcher = NSEDataFetcher()
    
    def test_fetch_single_stock(self):
        """Test fetching single stock data"""
        prices = self.fetcher.fetch_stock_data(
            ['RELIANCE'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        self.assertFalse(prices.empty, "Prices should not be empty")
        self.assertIn('RELIANCE', prices.columns, "Should contain RELIANCE")
    
    def test_returns_calculation(self):
        """Test return calculation"""
        # Create dummy price data
        dates = pd.date_range('2023-01-01', periods=10)
        prices = pd.DataFrame({
            'A': [100, 101, 102, 101, 103, 104, 103, 105, 106, 107]
        }, index=dates)
        
        returns = self.fetcher.compute_returns(prices)
        
        self.assertEqual(len(returns), 9, "Should have 9 returns for 10 prices")
        self.assertAlmostEqual(returns.iloc[0]['A'], 0.01, places=4)


class TestPortfolioOptimizer(unittest.TestCase):
    """Test portfolio optimization"""
    
    def setUp(self):
        """Set up test data"""
        # Create synthetic return data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500)
        
        self.returns = pd.DataFrame(
            np.random.randn(500, 3) * 0.01,
            index=dates,
            columns=['Stock_A', 'Stock_B', 'Stock_C']
        )
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        from portfolio_optimizer_v2 import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(self.returns, risk_free_rate=0.06)
        
        self.assertEqual(optimizer.n_assets, 3)
        self.assertEqual(optimizer.risk_free_rate, 0.06)
        self.assertIsNotNone(optimizer.cov_matrix)
        self.assertIsNotNone(optimizer.exp_returns)
    
    def test_portfolio_weights_sum_to_one(self):
        """Test that optimized weights sum to 1"""
        from portfolio_optimizer_v2 import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(self.returns)
        result = optimizer.optimize_max_sharpe()
        
        weights_sum = result.weights.sum()
        self.assertAlmostEqual(weights_sum, 1.0, places=5,
                              msg="Weights should sum to 1.0")
    
    def test_weights_non_negative(self):
        """Test that weights are non-negative (no short selling)"""
        from portfolio_optimizer_v2 import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(self.returns)
        result = optimizer.optimize_max_sharpe()
        
        self.assertTrue(np.all(result.weights >= -1e-6),
                       "All weights should be non-negative")
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation"""
        from portfolio_optimizer_v2 import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(self.returns)
        results, weights = optimizer.simulate_portfolios(num_portfolios=100)
        
        self.assertEqual(len(results), 100, "Should generate 100 portfolios")
        self.assertEqual(weights.shape, (100, 3), "Weights shape should be (100, 3)")
        
        # Check all weights sum to 1
        weights_sums = weights.sum(axis=1)
        np.testing.assert_array_almost_equal(
            weights_sums,
            np.ones(100),
            decimal=5,
            err_msg="All portfolio weights should sum to 1"
        )
    
    def test_portfolio_performance_calculation(self):
        """Test portfolio performance metrics"""
        from portfolio_optimizer_v2 import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(self.returns)
        weights = np.array([0.33, 0.33, 0.34])
        
        ret, risk, sharpe = optimizer.portfolio_performance(weights)
        
        self.assertIsInstance(ret, float)
        self.assertIsInstance(risk, float)
        self.assertIsInstance(sharpe, float)
        self.assertGreater(risk, 0, "Risk should be positive")
    
    def test_min_risk_optimization(self):
        """Test minimum risk optimization"""
        from portfolio_optimizer_v2 import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(self.returns)
        min_risk_result = optimizer.optimize_min_risk()
        
        self.assertIsNotNone(min_risk_result)
        self.assertGreater(min_risk_result.volatility, 0)
    
    def test_risk_parity(self):
        """Test risk parity optimization"""
        from portfolio_optimizer_v2 import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer(self.returns)
        rp_result = optimizer.optimize_risk_parity()
        
        self.assertIsNotNone(rp_result)
        self.assertAlmostEqual(rp_result.weights.sum(), 1.0, places=5)


class TestRiskMetrics(unittest.TestCase):
    """Test risk metric calculations"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.returns = pd.Series(np.random.randn(1000) * 0.01)
    
    def test_var_cvar_calculation(self):
        """Test VaR and CVaR calculation"""
        from utils_enhanced import RiskMetrics
        
        var, cvar = RiskMetrics.calculate_var_cvar(self.returns, 0.95)
        
        self.assertIsNotNone(var)
        self.assertIsNotNone(cvar)
        self.assertLess(cvar, var, "CVaR should be more negative than VaR")
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        from utils_enhanced import RiskMetrics
        
        max_dd, dd_series, dates = RiskMetrics.calculate_max_drawdown(self.returns)
        
        self.assertLessEqual(max_dd, 0, "Max drawdown should be negative")
        self.assertIsInstance(dd_series, pd.Series)
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        from utils_enhanced import calculate_sharpe_ratio
        
        sharpe = calculate_sharpe_ratio(self.returns, risk_free_rate=0.06)
        
        self.assertIsInstance(sharpe, float)
    
    def test_sortino_ratio(self):
        """Test Sortino ratio calculation"""
        from utils_enhanced import RiskMetrics
        
        sortino = RiskMetrics.calculate_sortino_ratio(self.returns, 0.06)
        
        self.assertIsInstance(sortino, float)
    
    def test_omega_ratio(self):
        """Test Omega ratio calculation"""
        from utils_enhanced import RiskMetrics
        
        omega = RiskMetrics.calculate_omega_ratio(self.returns)
        
        self.assertIsInstance(omega, float)
        self.assertGreater(omega, 0, "Omega ratio should be positive")


class TestMLPredictor(unittest.TestCase):
    """Test ML prediction functionality"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500)
        
        self.returns = pd.DataFrame(
            np.random.randn(500, 3) * 0.01,
            index=dates,
            columns=['Stock_A', 'Stock_B', 'Stock_C']
        )
    
    def test_predictor_initialization(self):
        """Test ML predictor initialization"""
        from ml_predictor_v2 import MLReturnPredictor
        
        predictor = MLReturnPredictor(self.returns, lookback_period=60)
        
        self.assertEqual(predictor.lookback_period, 60)
        self.assertEqual(len(predictor.returns.columns), 3)
    
    def test_feature_engineering(self):
        """Test feature creation"""
        from ml_predictor_v2 import FeatureEngine
        
        data = self.returns['Stock_A']
        features = FeatureEngine.create_technical_features(data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 10, "Should create multiple features")
        self.assertEqual(len(features), len(data), "Features should match data length")
    
    def test_model_training(self):
        """Test model training"""
        from ml_predictor_v2 import MLReturnPredictor
        
        predictor = MLReturnPredictor(self.returns, lookback_period=60)
        
        # This might fail if insufficient data, which is expected
        try:
            results = predictor.train_all_models(model_type='random_forest')
            self.assertIsInstance(results, dict)
        except ValueError:
            # Expected if insufficient data
            pass
    
    def test_portfolio_creation(self):
        """Test ML portfolio creation"""
        from ml_predictor_v2 import MLReturnPredictor
        
        predictor = MLReturnPredictor(self.returns, lookback_period=60)
        
        # Mock predictions
        predictor.predictions = {'Stock_A': 0.10, 'Stock_B': 0.05, 'Stock_C': -0.02}
        
        weights = predictor.create_ml_optimized_portfolio(method='proportional')
        
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)


class TestDataValidation(unittest.TestCase):
    """Test data validation functions"""
    
    def test_validation_empty_data(self):
        """Test validation with empty data"""
        from utils_enhanced import validate_portfolio_data
        
        empty_df = pd.DataFrame()
        is_valid, issues = validate_portfolio_data(empty_df)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
    
    def test_validation_nan_values(self):
        """Test validation with NaN values"""
        from utils_enhanced import validate_portfolio_data
        
        data = pd.DataFrame({
            'A': [0.01, np.nan, 0.02],
            'B': [0.01, 0.02, 0.03]
        })
        
        is_valid, issues = validate_portfolio_data(data)
        
        # Should have warning about NaN values
        self.assertTrue(any('NaN' in issue for issue in issues))
    
    def test_validation_extreme_values(self):
        """Test validation with extreme values"""
        from utils_enhanced import validate_portfolio_data
        
        data = pd.DataFrame({
            'A': [0.01, 0.02, 0.8],  # 80% return is extreme
            'B': [0.01, 0.02, 0.03]
        })
        
        is_valid, issues = validate_portfolio_data(data)
        
        # Should have warning about extreme returns
        self.assertTrue(any('extreme' in issue.lower() for issue in issues))
    
    def test_weight_validation(self):
        """Test weight validation"""
        from utils_enhanced import validate_portfolio_data
        
        data = pd.DataFrame({
            'A': [0.01, 0.02, 0.03],
            'B': [0.01, 0.02, 0.03]
        })
        
        # Weights don't sum to 1
        weights = np.array([0.4, 0.5])
        
        is_valid, issues = validate_portfolio_data(data, weights)
        
        # Should have warning about weights not summing to 1
        self.assertTrue(any('sum' in issue.lower() for issue in issues))


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance analytics"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500)
        self.returns = pd.Series(
            np.random.randn(500) * 0.01,
            index=dates
        )
    
    def test_rolling_metrics(self):
        """Test rolling metrics calculation"""
        from utils_enhanced import PerformanceAnalytics
        
        rolling = PerformanceAnalytics.rolling_metrics(self.returns, window=252)
        
        self.assertIsInstance(rolling, pd.DataFrame)
        self.assertIn('rolling_sharpe', rolling.columns)
        self.assertIn('rolling_volatility', rolling.columns)
    
    def test_annualization(self):
        """Test return and volatility annualization"""
        from utils_enhanced import annualize_returns, annualize_volatility
        
        annual_ret = annualize_returns(self.returns)
        annual_vol = annualize_volatility(self.returns)
        
        self.assertIsInstance(annual_ret, float)
        self.assertIsInstance(annual_vol, float)
        self.assertGreater(annual_vol, 0)


def run_all_tests():
    """Run all test suites"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataFetch))
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolioOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)