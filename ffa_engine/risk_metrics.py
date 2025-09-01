"""
Risk Metrics Calculation Module

Calculates Value at Risk (VaR), Conditional Value at Risk (CVaR),
and other risk metrics for FFA portfolios and individual contracts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import warnings


class RiskMetrics:
    """
    Calculates comprehensive risk metrics for FFA contracts and portfolios.
    """
    
    def __init__(self, simulation_results: Optional[Dict[str, np.ndarray]] = None):
        self.simulation_results = simulation_results
        self.risk_metrics = {}
        
    def calculate_var(self, 
                      returns_or_payoffs: np.ndarray,
                      confidence_level: float = 0.95,
                      method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns_or_payoffs: Array of returns or payoffs
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical', 'parametric', or 'cornish_fisher'
            
        Returns:
            VaR value (positive number representing potential loss)
        """
        if len(returns_or_payoffs) == 0:
            return np.nan
        
        if method == 'historical':
            # Historical simulation method
            percentile = (1 - confidence_level) * 100
            var = -np.percentile(returns_or_payoffs, percentile)
            
        elif method == 'parametric':
            # Parametric method (assumes normal distribution)
            mean = np.mean(returns_or_payoffs)
            std = np.std(returns_or_payoffs)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std)
            
        elif method == 'cornish_fisher':
            # Cornish-Fisher expansion (accounts for skewness and kurtosis)
            mean = np.mean(returns_or_payoffs)
            std = np.std(returns_or_payoffs)
            skewness = stats.skew(returns_or_payoffs)
            kurtosis = stats.kurtosis(returns_or_payoffs)
            
            z = stats.norm.ppf(1 - confidence_level)
            
            # Cornish-Fisher adjustment
            z_cf = (z + 
                   (z**2 - 1) * skewness / 6 +
                   (z**3 - 3*z) * kurtosis / 24 -
                   (2*z**3 - 5*z) * skewness**2 / 36)
            
            var = -(mean + z_cf * std)
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return max(var, 0)  # VaR should be positive
    
    def calculate_cvar(self, 
                       returns_or_payoffs: np.ndarray,
                       confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns_or_payoffs: Array of returns or payoffs
            confidence_level: Confidence level
            
        Returns:
            CVaR value (average of losses beyond VaR)
        """
        if len(returns_or_payoffs) == 0:
            return np.nan
        
        # Find VaR first
        var = self.calculate_var(returns_or_payoffs, confidence_level, 'historical')
        
        # CVaR is the average of losses beyond VaR
        losses = -returns_or_payoffs  # Convert to losses
        tail_losses = losses[losses >= var]
        
        if len(tail_losses) == 0:
            return var  # If no losses beyond VaR, CVaR equals VaR
        
        cvar = np.mean(tail_losses)
        return cvar
    
    def calculate_portfolio_var(self,
                              portfolio_weights: Dict[str, float],
                              correlation_matrix: np.ndarray,
                              route_vars: Dict[str, float]) -> float:
        """
        Calculate portfolio VaR using correlation structure.
        
        Args:
            portfolio_weights: Dictionary with route weights
            correlation_matrix: Correlation matrix between routes
            route_vars: Individual VaR for each route
            
        Returns:
            Portfolio VaR
        """
        routes = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[route] for route in routes])
        vars_vector = np.array([route_vars[route] for route in routes])
        
        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(correlation_matrix, weights)) * np.outer(vars_vector, vars_vector)
        portfolio_var = np.sqrt(np.sum(portfolio_variance))
        
        return portfolio_var
    
    def calculate_risk_metrics_for_route(self,
                                       route: str,
                                       confidence_levels: List[float] = [0.90, 0.95, 0.99]) -> Dict:
        """
        Calculate comprehensive risk metrics for a specific route.
        
        Args:
            route: Route name
            confidence_levels: List of confidence levels for VaR/CVaR
            
        Returns:
            Dictionary with risk metrics
        """
        if self.simulation_results is None or route not in self.simulation_results:
            raise ValueError(f"No simulation results available for route {route}")
        
        scenarios = self.simulation_results[route]
        
        # Calculate returns (log returns from initial to final prices)
        initial_prices = scenarios[:, 0]
        final_prices = scenarios[:, -1]
        returns = np.log(final_prices / initial_prices)
        
        # Basic statistics
        metrics = {
            'route': route,
            'mean_return': np.mean(returns),
            'volatility': np.std(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'scenarios_count': len(returns)
        }
        
        # VaR and CVaR for different confidence levels
        for conf_level in confidence_levels:
            conf_pct = int(conf_level * 100)
            
            metrics[f'var_{conf_pct}'] = self.calculate_var(returns, conf_level, 'historical')
            metrics[f'cvar_{conf_pct}'] = self.calculate_cvar(returns, conf_level)
            metrics[f'var_{conf_pct}_parametric'] = self.calculate_var(returns, conf_level, 'parametric')
            
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            metrics[f'percentile_{p}'] = np.percentile(returns, p)
        
        return metrics
    
    def calculate_payoff_distribution_metrics(self, payoffs: np.ndarray) -> Dict:
        """
        Calculate metrics for payoff distributions.
        
        Args:
            payoffs: Array of contract payoffs
            
        Returns:
            Dictionary with distribution metrics
        """
        metrics = {
            'mean_payoff': np.mean(payoffs),
            'std_payoff': np.std(payoffs),
            'skewness': stats.skew(payoffs),
            'kurtosis': stats.kurtosis(payoffs),
            'min_payoff': np.min(payoffs),
            'max_payoff': np.max(payoffs),
            'positive_payoff_probability': np.mean(payoffs > 0),
            'negative_payoff_probability': np.mean(payoffs < 0),
            'zero_payoff_probability': np.mean(payoffs == 0)
        }
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            metrics[f'payoff_percentile_{p}'] = np.percentile(payoffs, p)
        
        # Profit/Loss metrics
        profits = payoffs[payoffs > 0]
        losses = payoffs[payoffs < 0]
        
        if len(profits) > 0:
            metrics['mean_profit'] = np.mean(profits)
            metrics['max_profit'] = np.max(profits)
        else:
            metrics['mean_profit'] = 0
            metrics['max_profit'] = 0
        
        if len(losses) > 0:
            metrics['mean_loss'] = np.mean(losses)
            metrics['max_loss'] = np.min(losses)  # Most negative value
        else:
            metrics['mean_loss'] = 0
            metrics['max_loss'] = 0
        
        return metrics
    
    def calculate_downside_risk_metrics(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> Dict:
        """
        Calculate downside risk metrics.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate for excess returns
            
        Returns:
            Dictionary with downside risk metrics
        """
        excess_returns = returns - risk_free_rate
        
        # Downside deviation (semi-standard deviation)
        negative_returns = returns[returns < risk_free_rate]
        if len(negative_returns) > 0:
            downside_deviation = np.sqrt(np.mean((negative_returns - risk_free_rate)**2))
        else:
            downside_deviation = 0.0
        
        # Sortino ratio
        mean_excess_return = np.mean(excess_returns)
        sortino_ratio = mean_excess_return / downside_deviation if downside_deviation > 0 else np.inf
        
        # Maximum drawdown (simplified for return series)
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        metrics = {
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': abs(max_drawdown),
            'negative_periods': len(negative_returns),
            'negative_period_ratio': len(negative_returns) / len(returns)
        }
        
        return metrics
    
    def generate_risk_report(self, route: str) -> pd.DataFrame:
        """
        Generate a comprehensive risk report for a route.
        
        Args:
            route: Route name
            
        Returns:
            DataFrame with risk report
        """
        risk_metrics = self.calculate_risk_metrics_for_route(route)
        
        # Convert to DataFrame for better presentation
        report_data = []
        for metric, value in risk_metrics.items():
            if isinstance(value, (int, float)):
                report_data.append({
                    'Metric': metric,
                    'Value': value,
                    'Description': self._get_metric_description(metric)
                })
        
        return pd.DataFrame(report_data)
    
    def _get_metric_description(self, metric: str) -> str:
        """
        Get description for risk metrics.
        """
        descriptions = {
            'mean_return': 'Average log return',
            'volatility': 'Standard deviation of returns',
            'skewness': 'Third moment - asymmetry of distribution',
            'kurtosis': 'Fourth moment - tail thickness',
            'var_95': '95% Value at Risk - potential loss exceeded 5% of time',
            'cvar_95': '95% Conditional VaR - average loss beyond VaR',
            'var_99': '99% Value at Risk - potential loss exceeded 1% of time',
            'cvar_99': '99% Conditional VaR - average loss beyond VaR',
            'scenarios_count': 'Number of Monte Carlo scenarios'
        }
        
        return descriptions.get(metric, 'Risk metric')