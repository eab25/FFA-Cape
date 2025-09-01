"""
Stochastic Model for FFA Price Dynamics

Implements mean-reverting models with seasonality for freight rate modeling.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple, Optional
import warnings


class StochasticModel:
    """
    Calibrates and implements stochastic models for FFA price dynamics.
    Uses mean-reverting processes with seasonality components.
    """
    
    def __init__(self):
        self.calibrated_params = {}
        self.routes = None
        self.seasonal_params = {}
        
    def calibrate_mean_reversion(self, data: pd.DataFrame, route: str) -> Dict:
        """
        Calibrate mean-reverting parameters for a specific route.
        Uses Ornstein-Uhlenbeck process: dX = κ(θ - X)dt + σdW
        
        Args:
            data: DataFrame with Date and price columns
            route: Route name to calibrate
            
        Returns:
            Dictionary with calibrated parameters {kappa, theta, sigma}
        """
        if route not in data.columns:
            raise ValueError(f"Route {route} not found in data")
        
        # Get price series and calculate log prices
        prices = data[route].dropna()
        log_prices = np.log(prices)
        
        # Calculate returns
        returns = log_prices.diff().dropna()
        lagged_prices = log_prices.shift(1).dropna()
        
        # Align the series
        min_length = min(len(returns), len(lagged_prices))
        returns = returns.iloc[-min_length:]
        lagged_prices = lagged_prices.iloc[-min_length:]
        
        # Estimate mean reversion parameters using OLS
        # Δlog(P) = α + β*log(P_{t-1}) + ε
        # Where β = -κΔt and α = κθΔt
        X = lagged_prices.values.reshape(-1, 1)
        y = returns.values
        
        # Fit linear regression
        reg = LinearRegression().fit(X, y)
        alpha = reg.intercept_
        beta = reg.coef_[0]
        
        # Convert to mean reversion parameters (assuming daily data, Δt = 1/252)
        dt = 1/252  # Daily time step in years
        
        if beta >= 0:
            # No mean reversion detected, use small positive kappa
            kappa = 0.1
            theta = np.mean(log_prices)
        else:
            kappa = -beta / dt
            theta = -alpha / beta
        
        # Calculate volatility (standard deviation of residuals)
        residuals = y - reg.predict(X)
        sigma = np.std(residuals) / np.sqrt(dt)
        
        params = {
            'kappa': kappa,
            'theta': theta, 
            'sigma': sigma,
            'mean_log_price': np.mean(log_prices),
            'std_log_price': np.std(log_prices)
        }
        
        return params
    
    def detect_seasonality(self, data: pd.DataFrame, route: str) -> Dict:
        """
        Detect seasonal patterns in freight rates.
        
        Args:
            data: DataFrame with Date and price columns
            route: Route name to analyze
            
        Returns:
            Dictionary with seasonal parameters
        """
        if route not in data.columns:
            raise ValueError(f"Route {route} not found in data")
        
        df = data[['Date', route]].dropna().copy()
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['log_price'] = np.log(df[route])
        
        # Calculate monthly averages
        monthly_avg = df.groupby('month')['log_price'].mean()
        overall_mean = df['log_price'].mean()
        
        # Seasonal deviations from mean
        seasonal_factors = monthly_avg - overall_mean
        
        # Calculate seasonal volatility
        seasonal_std = df.groupby('month')['log_price'].std().mean()
        
        seasonal_params = {
            'monthly_factors': seasonal_factors.to_dict(),
            'seasonal_volatility': seasonal_std,
            'overall_mean': overall_mean,
            'seasonal_strength': np.std(seasonal_factors)
        }
        
        return seasonal_params
    
    def calibrate_all_routes(self, data: pd.DataFrame) -> Dict:
        """
        Calibrate models for all routes in the dataset.
        
        Args:
            data: DataFrame with Date column and route columns
            
        Returns:
            Dictionary with calibrated parameters for all routes
        """
        routes = [col for col in data.columns if col != 'Date']
        self.routes = routes
        
        for route in routes:
            try:
                # Calibrate mean reversion
                mr_params = self.calibrate_mean_reversion(data, route)
                
                # Detect seasonality
                seasonal_params = self.detect_seasonality(data, route)
                
                # Combine parameters
                self.calibrated_params[route] = {
                    'mean_reversion': mr_params,
                    'seasonality': seasonal_params
                }
                
            except Exception as e:
                warnings.warn(f"Failed to calibrate route {route}: {str(e)}")
                continue
        
        return self.calibrated_params
    
    def get_correlation_structure(self, data: pd.DataFrame) -> np.ndarray:
        """
        Estimate correlation structure between routes.
        
        Args:
            data: DataFrame with route price data
            
        Returns:
            Correlation matrix as numpy array
        """
        routes = [col for col in data.columns if col != 'Date']
        
        # Calculate log returns
        log_returns = pd.DataFrame()
        for route in routes:
            prices = data[route].dropna()
            if len(prices) > 1:
                log_returns[route] = np.log(prices / prices.shift(1)).dropna()
        
        # Calculate correlation matrix
        correlation_matrix = log_returns.corr().fillna(0)
        
        return correlation_matrix.values
    
    def generate_scenarios(self, 
                          route: str, 
                          initial_price: float,
                          time_horizon: int,
                          n_scenarios: int = 1000,
                          dt: float = 1/252) -> np.ndarray:
        """
        Generate price scenarios using calibrated model.
        
        Args:
            route: Route name
            initial_price: Starting price
            time_horizon: Number of time steps
            n_scenarios: Number of scenarios to generate
            dt: Time step size (default: daily = 1/252 years)
            
        Returns:
            Array of shape (n_scenarios, time_horizon + 1) with price paths
        """
        if route not in self.calibrated_params:
            raise ValueError(f"Route {route} not calibrated. Call calibrate_all_routes() first.")
        
        params = self.calibrated_params[route]['mean_reversion']
        seasonal = self.calibrated_params[route]['seasonality']
        
        kappa = params['kappa']
        theta = params['theta']
        sigma = params['sigma']
        
        # Initialize price paths
        scenarios = np.zeros((n_scenarios, time_horizon + 1))
        scenarios[:, 0] = initial_price
        
        # Generate random shocks
        np.random.seed(42)  # For reproducibility
        dW = np.random.normal(0, np.sqrt(dt), (n_scenarios, time_horizon))
        
        # Simulate paths
        for t in range(time_horizon):
            current_log_price = np.log(scenarios[:, t])
            
            # Add seasonal effect (simplified)
            current_month = ((t // 22) % 12) + 1  # Approximate month
            seasonal_factor = seasonal['monthly_factors'].get(current_month, 0)
            
            # Mean reversion with seasonality
            drift = kappa * (theta + seasonal_factor - current_log_price) * dt
            diffusion = sigma * dW[:, t]
            
            # Update log price
            new_log_price = current_log_price + drift + diffusion
            scenarios[:, t + 1] = np.exp(new_log_price)
        
        return scenarios