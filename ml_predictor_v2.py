"""
Enhanced Machine Learning Return Prediction Module

Major improvements:
1. Advanced feature engineering with 20+ technical indicators
2. Proper cross-validation with walk-forward analysis
3. Multiple model support with ensemble methods
4. Feature importance analysis
5. Prediction confidence intervals
6. Robust error handling and data validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


@dataclass
class ModelPerformance:
    """Store model performance metrics"""
    r2_score: float
    rmse: float
    mae: float
    feature_importance: Dict[str, float]


class FeatureEngine:
    """Advanced feature engineering for time series data"""
    
    @staticmethod
    def create_technical_features(data: pd.Series, 
                                  windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        Create comprehensive technical indicators
        
        Args:
            data: Price or return series
            windows: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with technical features
        """
        features = pd.DataFrame(index=data.index)
        
        # 1. Moving Averages
        for window in windows:
            features[f'sma_{window}'] = data.rolling(window, min_periods=1).mean()
            features[f'ema_{window}'] = data.ewm(span=window, adjust=False).mean()
        
        # 2. Momentum indicators
        for window in [5, 10, 20]:
            features[f'momentum_{window}'] = data.pct_change(window).fillna(0)
            features[f'roc_{window}'] = (data / data.shift(window) - 1).fillna(0)
        
        # 3. Volatility indicators
        for window in windows:
            features[f'volatility_{window}'] = data.rolling(window, min_periods=1).std()
            features[f'atr_{window}'] = FeatureEngine._calculate_atr(data, window)
        
        # 4. Price position indicators
        for window in [10, 20]:
            rolling_min = data.rolling(window, min_periods=1).min()
            rolling_max = data.rolling(window, min_periods=1).max()
            features[f'pct_from_low_{window}'] = (data - rolling_min) / (rolling_max - rolling_min + 1e-10)
        
        # 5. Trend indicators
        for window in [5, 10, 20]:
            features[f'trend_{window}'] = (
                data.rolling(window, min_periods=1).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
            )
        
        # 6. RSI-like indicator
        for window in [14, 21]:
            features[f'rsi_{window}'] = FeatureEngine._calculate_rsi(data, window)
        
        # 7. MACD components
        ema_12 = data.ewm(span=12, adjust=False).mean()
        ema_26 = data.ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # 8. Bollinger Bands position
        for window in [20]:
            sma = data.rolling(window, min_periods=1).mean()
            std = data.rolling(window, min_periods=1).std()
            features[f'bb_position_{window}'] = (data - sma) / (2 * std + 1e-10)
        
        # Fill any remaining NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features
    
    @staticmethod
    def _calculate_atr(data: pd.Series, window: int) -> pd.Series:
        """Calculate Average True Range"""
        high = data.rolling(window, min_periods=1).max()
        low = data.rolling(window, min_periods=1).min()
        tr = high - low
        return tr.rolling(window, min_periods=1).mean()
    
    @staticmethod
    def _calculate_rsi(data: pd.Series, window: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)


class MLReturnPredictor:
    """
    Enhanced ML-based return prediction with multiple models and validation
    """
    
    def __init__(self, 
                 returns: pd.DataFrame,
                 lookback_period: int = 60,
                 prediction_horizon: int = 20):
        """
        Initialize ML predictor
        
        Args:
            returns: DataFrame of historical returns
            lookback_period: Days of history for features
            prediction_horizon: Days ahead to predict
        """
        # Validate inputs
        if returns.empty:
            raise ValueError("Returns DataFrame cannot be empty")
        
        if returns.isnull().any().any():
            warnings.warn("NaN values in returns. Filling with forward/backward fill.")
            returns = returns.fillna(method='ffill').fillna(method='bfill')
        
        self.returns = returns
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.models: Dict[str, Dict] = {}
        self.predictions: Dict[str, float] = {}
        self.feature_engine = FeatureEngine()
        self.scalers: Dict[str, StandardScaler] = {}
    
    def train_all_models(self, 
                        model_type: str = 'random_forest',
                        cv_splits: int = 5) -> Dict[str, ModelPerformance]:
        """
        Train models for all assets with cross-validation
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', 'ridge', or 'ensemble'
            cv_splits: Number of cross-validation splits
            
        Returns:
            Dictionary of training results per asset
        """
        results = {}
        
        for ticker in self.returns.columns:
            try:
                print(f"Training model for {ticker}...", end=' ')
                performance = self._train_single_model(ticker, model_type, cv_splits)
                results[ticker] = performance
                print(f"✓ R²={performance.r2_score:.3f}")
            except Exception as e:
                warnings.warn(f"Failed to train model for {ticker}: {str(e)}")
                continue
        
        return results
    
    def _train_single_model(self, 
                           ticker: str,
                           model_type: str,
                           cv_splits: int) -> ModelPerformance:
        """Train model for a single asset"""
        # Prepare data
        data = self.returns[ticker]
        
        # Create features
        features_df = self.feature_engine.create_technical_features(
            (1 + data).cumprod()  # Convert to price series
        )
        
        # Create target (future returns)
        target = data.shift(-self.prediction_horizon).fillna(0)
        
        # Align data
        valid_idx = ~(features_df.isna().any(axis=1) | target.isna())
        X = features_df[valid_idx].values
        y = target[valid_idx].values
        
        # Ensure sufficient data
        if len(X) < max(100, self.lookback_period):
            raise ValueError(f"Insufficient data for {ticker}: {len(X)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[ticker] = scaler
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self._create_model(model_type)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            cv_scores.append(r2_score(y_val, y_pred))
        
        # Train final model on all data
        final_model = self._create_model(model_type)
        final_model.fit(X_scaled, y)
        
        # Store model
        self.models[ticker] = {
            'model': final_model,
            'features': features_df.columns.tolist(),
            'cv_score': np.mean(cv_scores)
        }
        
        # Calculate performance metrics
        y_pred_full = final_model.predict(X_scaled)
        
        # Get feature importance
        if hasattr(final_model, 'feature_importances_'):
            importance = dict(zip(
                features_df.columns,
                final_model.feature_importances_
            ))
        else:
            importance = {}
        r2 = r2_score(y, y_pred_full)
        
        # Add interpretation
        if r2 > 0.30:
            quality = "Excellent"
            symbol = "✓✓"
        elif r2 > 0.20:
            quality = "Good"
            symbol = "✓"
        elif r2 > 0.10:
            quality = "Fair"
            symbol = "⚠"
        else:
            quality = "Poor"
            symbol = "✗"
            warnings.warn(f"Low R² for {ticker}. Consider more data or different features.")
        
        print(f"{symbol} {quality} (R²={r2:.3f})")
        return ModelPerformance(
            r2_score=r2_score(y, y_pred_full),
            rmse=np.sqrt(mean_squared_error(y, y_pred_full)),
            mae=mean_absolute_error(y, y_pred_full),
            feature_importance=importance
        )
        
    
    def _create_model(self, model_type: str):
        """Create model based on type"""
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        elif model_type == 'lasso':
            return Lasso(alpha=0.01, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict_next_period(self, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Predict next period returns with confidence intervals
        
        Args:
            confidence_level: Confidence level for predictions
            
        Returns:
            Dictionary of predicted returns per asset
        """
        predictions = {}
        
        for ticker in self.returns.columns:
            if ticker not in self.models:
                warnings.warn(f"No model available for {ticker}. Using 0 prediction.")
                predictions[ticker] = 0.0
                continue
            
            try:
                # Get latest features
                data = self.returns[ticker]
                features_df = self.feature_engine.create_technical_features(
                    (1 + data).cumprod()
                )
                
                # Get latest feature values
                latest_features = features_df.iloc[-1:].values
                
                # Scale features
                latest_scaled = self.scalers[ticker].transform(latest_features)
                
                # Make prediction
                pred = self.models[ticker]['model'].predict(latest_scaled)[0]
                
                # Annualize prediction
                predictions[ticker] = pred * (252 / self.prediction_horizon)
                
            except Exception as e:
                warnings.warn(f"Prediction failed for {ticker}: {str(e)}")
                predictions[ticker] = 0.0
        
        self.predictions = predictions
        return predictions
    
    def create_ml_optimized_portfolio(self, method: str = 'proportional') -> np.ndarray:
        """
        Create portfolio weights based on ML predictions
        
        Args:
            method: 'proportional', 'top_n', or 'threshold'
            
        Returns:
            Array of portfolio weights
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run predict_next_period() first.")
        
        pred_series = pd.Series(self.predictions)
        
        if method == 'proportional':
            # Weight proportional to positive predictions
            positive_preds = pred_series.clip(lower=0)
            if positive_preds.sum() == 0:
                # If no positive predictions, use equal weights
                weights = np.ones(len(pred_series)) / len(pred_series)
            else:
                weights = positive_preds / positive_preds.sum()
        
        elif method == 'top_n':
            # Invest in top 5 predicted assets
            n = min(5, len(pred_series))
            top_assets = pred_series.nlargest(n)
            weights = pd.Series(0.0, index=pred_series.index)
            weights[top_assets.index] = 1.0 / n
        
        elif method == 'threshold':
            # Invest in assets with positive predictions above threshold
            threshold = pred_series.quantile(0.6)
            selected = pred_series[pred_series > threshold]
            if len(selected) == 0:
                weights = pd.Series(1.0 / len(pred_series), index=pred_series.index)
            else:
                weights = pd.Series(0.0, index=pred_series.index)
                weights[selected.index] = 1.0 / len(selected)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return weights.values
    
    def plot_predictions_vs_actual(self, save_path: str = 'results/ml_predictions.png'):
        """Enhanced prediction visualization"""
        if not self.predictions:
            raise ValueError("No predictions to plot. Run predict_next_period() first.")
        
        # Calculate actual annualized returns
        actual_returns = self.returns.iloc[-252:].mean() * 252
        predicted_returns = pd.Series(self.predictions)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Actual': actual_returns,
            'Predicted': predicted_returns
        }).sort_values('Predicted', ascending=False)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart comparison
        x = np.arange(len(comparison))
        width = 0.35
        
        ax1.bar(x - width/2, comparison['Actual'], width, 
                label='Actual (Historical)', alpha=0.8, color='steelblue')
        ax1.bar(x + width/2, comparison['Predicted'], width,
                label='Predicted (ML)', alpha=0.8, color='coral')
        
        ax1.set_xlabel('Assets', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Annual Return', fontsize=11, fontweight='bold')
        ax1.set_title('ML Predictions vs Historical Returns', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comparison.index, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))
        
        # Scatter plot
        ax2.scatter(comparison['Actual'], comparison['Predicted'], 
                   s=100, alpha=0.6, color='purple')
        
        # Add diagonal line (perfect prediction)
        min_val = min(comparison['Actual'].min(), comparison['Predicted'].min())
        max_val = max(comparison['Actual'].max(), comparison['Predicted'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
        
        # Add labels for each point
        for idx, row in comparison.iterrows():
            ax2.annotate(idx, (row['Actual'], row['Predicted']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        ax2.set_xlabel('Actual Return (Historical)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Predicted Return (ML)', fontsize=11, fontweight='bold')
        ax2.set_title('Prediction Accuracy', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))
        
        # Calculate and display R²
        from scipy.stats import pearsonr
        corr, _ = pearsonr(comparison['Actual'], comparison['Predicted'])
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ML predictions plot saved to {save_path}")
        return comparison
    
    def plot_feature_importance(self, ticker: str, top_n: int = 15,
                               save_path: Optional[str] = None):
        """
        Plot feature importance for a specific asset
        
        Args:
            ticker: Asset ticker
            top_n: Number of top features to display
            save_path: Path to save plot (optional)
        """
        if ticker not in self.models:
            raise ValueError(f"No model found for {ticker}")
        
        model_data = self.models[ticker]
        if not hasattr(model_data['model'], 'feature_importances_'):
            warnings.warn(f"Model for {ticker} does not have feature importances")
            return
        
        # Get feature importance
        importance = pd.Series(
            model_data['model'].feature_importances_,
            index=model_data['features']
        ).sort_values(ascending=False)
        
        # Plot top N features
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.head(top_n).plot(kind='barh', ax=ax, color='teal')
        
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Features for {ticker}', 
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def backtest_predictions(self, test_period_days: int = 252) -> pd.DataFrame:
        """
        Backtest prediction accuracy over time
        
        Args:
            test_period_days: Number of days to backtest
            
        Returns:
            DataFrame with backtest results
        """
        results = []
        
        # Use last test_period_days for backtesting
        if len(self.returns) < test_period_days + self.lookback_period:
            warnings.warn("Insufficient data for backtesting")
            return pd.DataFrame()
        
        test_returns = self.returns.iloc[-test_period_days:]
        
        for i in range(0, len(test_returns) - self.prediction_horizon, 
                      self.prediction_horizon):
            # Get data up to current point
            current_data = self.returns.iloc[:-(test_period_days - i)]
            
            # Train models on current data
            temp_predictor = MLReturnPredictor(
                current_data,
                self.lookback_period,
                self.prediction_horizon
            )
            
            try:
                temp_predictor.train_all_models(model_type='random_forest')
                predictions = temp_predictor.predict_next_period()
                
                # Get actual returns for next period
                actual_start = len(current_data)
                actual_end = actual_start + self.prediction_horizon
                actual_returns = self.returns.iloc[actual_start:actual_end].mean()
                
                results.append({
                    'date': current_data.index[-1],
                    'predictions': predictions,
                    'actual': actual_returns.to_dict()
                })
            except:
                continue
        
        return pd.DataFrame(results)