"""
Monte Carlo Simulation Engine for FFA Markets

Orchestrates the simulation of thousands of price scenarios 
for multiple routes and time horizons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .stochastic_model import StochasticModel
import warnings


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for FFA price scenarios.
    """
    
    def __init__(self, stochastic_model: StochasticModel):
        self.model = stochastic_model
        self.simulation_results = {}
        
    def run_simulation(self,
                      initial_prices: Dict[str, float],
                      time_horizon_days: int = 252,  # 1 year
                      n_scenarios: int = 10000,
                      correlation_matrix: Optional[np.ndarray] = None,
                      random_seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulation for all routes.
        
        Args:
            initial_prices: Dictionary with route names and starting prices
            time_horizon_days: Number of days to simulate
            n_scenarios: Number of scenarios to generate
            correlation_matrix: Optional correlation matrix for correlated scenarios
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation results for each route
        """
        np.random.seed(random_seed)
        
        routes = list(initial_prices.keys())
        n_routes = len(routes)
        
        # Initialize results storage
        simulation_results = {}
        
        if correlation_matrix is not None and correlation_matrix.shape == (n_routes, n_routes):
            # Generate correlated scenarios
            simulation_results = self._generate_correlated_scenarios(
                initial_prices, time_horizon_days, n_scenarios, correlation_matrix
            )
        else:
            # Generate independent scenarios for each route
            for route in routes:
                if route in self.model.calibrated_params:
                    scenarios = self.model.generate_scenarios(
                        route=route,
                        initial_price=initial_prices[route],
                        time_horizon=time_horizon_days,
                        n_scenarios=n_scenarios
                    )
                    simulation_results[route] = scenarios
                else:
                    warnings.warn(f"Route {route} not calibrated, skipping simulation")
        
        self.simulation_results = simulation_results
        return simulation_results
    
    def _generate_correlated_scenarios(self,
                                     initial_prices: Dict[str, float],
                                     time_horizon_days: int,
                                     n_scenarios: int,
                                     correlation_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate correlated price scenarios using Cholesky decomposition.
        
        Args:
            initial_prices: Starting prices for each route
            time_horizon_days: Simulation horizon
            n_scenarios: Number of scenarios
            correlation_matrix: Correlation matrix between routes
            
        Returns:
            Dictionary with correlated scenarios
        """
        routes = list(initial_prices.keys())
        n_routes = len(routes)
        
        # Check if correlation matrix is positive definite
        try:
            cholesky = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            warnings.warn("Correlation matrix is not positive definite, using independent scenarios")
            return self._generate_independent_scenarios(initial_prices, time_horizon_days, n_scenarios)
        
        # Initialize scenario storage
        scenarios = {}
        for route in routes:
            scenarios[route] = np.zeros((n_scenarios, time_horizon_days + 1))
            scenarios[route][:, 0] = initial_prices[route]
        
        # Generate correlated random shocks
        dt = 1/252  # Daily time step
        
        for t in range(time_horizon_days):
            # Generate independent random variables
            independent_shocks = np.random.normal(0, 1, (n_scenarios, n_routes))
            
            # Apply correlation structure
            correlated_shocks = independent_shocks @ cholesky.T
            
            # Apply shocks to each route
            for i, route in enumerate(routes):
                if route in self.model.calibrated_params:
                    params = self.model.calibrated_params[route]['mean_reversion']
                    seasonal = self.model.calibrated_params[route]['seasonality']
                    
                    kappa = params['kappa']
                    theta = params['theta']
                    sigma = params['sigma']
                    
                    current_log_price = np.log(scenarios[route][:, t])
                    
                    # Add seasonal effect
                    current_month = ((t // 22) % 12) + 1
                    seasonal_factor = seasonal['monthly_factors'].get(current_month, 0)
                    
                    # Mean reversion with seasonality
                    drift = kappa * (theta + seasonal_factor - current_log_price) * dt
                    diffusion = sigma * np.sqrt(dt) * correlated_shocks[:, i]
                    
                    # Update price
                    new_log_price = current_log_price + drift + diffusion
                    scenarios[route][:, t + 1] = np.exp(new_log_price)
        
        return scenarios
    
    def _generate_independent_scenarios(self,
                                      initial_prices: Dict[str, float],
                                      time_horizon_days: int,
                                      n_scenarios: int) -> Dict[str, np.ndarray]:
        """
        Generate independent scenarios for each route.
        """
        scenarios = {}
        for route, initial_price in initial_prices.items():
            if route in self.model.calibrated_params:
                scenarios[route] = self.model.generate_scenarios(
                    route=route,
                    initial_price=initial_price,
                    time_horizon=time_horizon_days,
                    n_scenarios=n_scenarios
                )
        return scenarios
    
    def get_scenario_statistics(self, route: str, percentiles: List[float] = [5, 25, 50, 75, 95]) -> Dict:
        """
        Calculate statistics for simulated scenarios.
        
        Args:
            route: Route name
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary with scenario statistics
        """
        if route not in self.simulation_results:
            raise ValueError(f"No simulation results for route {route}")
        
        scenarios = self.simulation_results[route]
        final_prices = scenarios[:, -1]  # Final prices across all scenarios
        
        stats = {
            'mean_final_price': np.mean(final_prices),
            'std_final_price': np.std(final_prices),
            'min_final_price': np.min(final_prices),
            'max_final_price': np.max(final_prices),
            'percentiles': {}
        }
        
        for p in percentiles:
            stats['percentiles'][f'p{p}'] = np.percentile(final_prices, p)
        
        return stats
    
    def get_path_statistics(self, route: str) -> pd.DataFrame:
        """
        Get statistics for the entire price path evolution.
        
        Args:
            route: Route name
            
        Returns:
            DataFrame with path statistics over time
        """
        if route not in self.simulation_results:
            raise ValueError(f"No simulation results for route {route}")
        
        scenarios = self.simulation_results[route]
        
        path_stats = pd.DataFrame({
            'day': range(scenarios.shape[1]),
            'mean': np.mean(scenarios, axis=0),
            'std': np.std(scenarios, axis=0),
            'p5': np.percentile(scenarios, 5, axis=0),
            'p25': np.percentile(scenarios, 25, axis=0),
            'p50': np.percentile(scenarios, 50, axis=0),
            'p75': np.percentile(scenarios, 75, axis=0),
            'p95': np.percentile(scenarios, 95, axis=0),
            'min': np.min(scenarios, axis=0),
            'max': np.max(scenarios, axis=0)
        })
        
        return path_stats
    
    def export_scenarios(self, route: str, n_export: int = 100) -> pd.DataFrame:
        """
        Export a subset of scenarios for analysis.
        
        Args:
            route: Route name
            n_export: Number of scenarios to export
            
        Returns:
            DataFrame with selected scenarios
        """
        if route not in self.simulation_results:
            raise ValueError(f"No simulation results for route {route}")
        
        scenarios = self.simulation_results[route]
        n_scenarios = min(n_export, scenarios.shape[0])
        
        # Select random scenarios
        selected_indices = np.random.choice(scenarios.shape[0], n_scenarios, replace=False)
        selected_scenarios = scenarios[selected_indices]
        
        # Create DataFrame
        columns = [f'day_{i}' for i in range(selected_scenarios.shape[1])]
        df = pd.DataFrame(selected_scenarios, columns=columns)
        df['scenario_id'] = selected_indices
        
        return df