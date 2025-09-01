"""
FFA Contract Pricing Module

Implements pricing for Forward Freight Agreement contracts
based on Monte Carlo simulation results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings


class FFAPricer:
    """
    Prices FFA contracts using Monte Carlo simulation results.
    """
    
    def __init__(self, simulation_results: Dict[str, np.ndarray]):
        self.simulation_results = simulation_results
        self.contract_prices = {}
        
    def price_monthly_average_contract(self,
                                     route: str,
                                     contract_month: int,
                                     strike_price: Optional[float] = None,
                                     contract_type: str = 'forward') -> Dict:
        """
        Price a monthly average FFA contract.
        
        Args:
            route: Route name (e.g., 'C5TC')
            contract_month: Month index (1-12) to calculate average for
            strike_price: Strike price for option contracts
            contract_type: 'forward', 'call', or 'put'
            
        Returns:
            Dictionary with contract pricing results
        """
        if route not in self.simulation_results:
            raise ValueError(f"No simulation results for route {route}")
        
        scenarios = self.simulation_results[route]
        n_scenarios, time_horizon = scenarios.shape
        
        # Calculate days for the contract month (approximate)
        days_per_month = time_horizon // 12
        start_day = (contract_month - 1) * days_per_month
        end_day = min(contract_month * days_per_month, time_horizon)
        
        if start_day >= time_horizon:
            raise ValueError(f"Contract month {contract_month} exceeds simulation horizon")
        
        # Calculate monthly averages for each scenario
        monthly_averages = np.mean(scenarios[:, start_day:end_day], axis=1)
        
        # Price different contract types
        if contract_type == 'forward':
            # Forward price is the expected value
            contract_price = np.mean(monthly_averages)
            payoffs = monthly_averages - contract_price
            
        elif contract_type == 'call' and strike_price is not None:
            # Call option payoff: max(S - K, 0)
            payoffs = np.maximum(monthly_averages - strike_price, 0)
            contract_price = np.mean(payoffs)
            
        elif contract_type == 'put' and strike_price is not None:
            # Put option payoff: max(K - S, 0)
            payoffs = np.maximum(strike_price - monthly_averages, 0)
            contract_price = np.mean(payoffs)
            
        else:
            raise ValueError(f"Invalid contract type: {contract_type}")
        
        # Calculate additional metrics
        results = {
            'contract_price': contract_price,
            'route': route,
            'contract_month': contract_month,
            'contract_type': contract_type,
            'strike_price': strike_price,
            'monthly_averages': monthly_averages,
            'payoffs': payoffs,
            'expected_payoff': np.mean(payoffs),
            'payoff_std': np.std(payoffs),
            'scenarios_count': n_scenarios,
            'contract_period_days': end_day - start_day
        }
        
        return results
    
    def price_calendar_spread(self,
                            route: str,
                            month1: int,
                            month2: int,
                            spread_direction: str = 'long_near') -> Dict:
        """
        Price a calendar spread between two contract months.
        
        Args:
            route: Route name
            month1: First contract month
            month2: Second contract month  
            spread_direction: 'long_near' (long month1, short month2) or 'long_far'
            
        Returns:
            Dictionary with spread pricing results
        """
        # Price individual contracts
        contract1 = self.price_monthly_average_contract(route, month1, contract_type='forward')
        contract2 = self.price_monthly_average_contract(route, month2, contract_type='forward')
        
        avg1 = contract1['monthly_averages']
        avg2 = contract2['monthly_averages']
        
        if spread_direction == 'long_near':
            # Long near month, short far month
            spread_payoffs = avg1 - avg2
        else:
            # Long far month, short near month
            spread_payoffs = avg2 - avg1
        
        spread_price = np.mean(spread_payoffs)
        
        results = {
            'spread_price': spread_price,
            'route': route,
            'month1': month1,
            'month2': month2,
            'spread_direction': spread_direction,
            'spread_payoffs': spread_payoffs,
            'spread_std': np.std(spread_payoffs),
            'scenarios_count': len(spread_payoffs),
            'contract1_price': contract1['contract_price'],
            'contract2_price': contract2['contract_price']
        }
        
        return results
    
    def price_strip_contract(self,
                           route: str,
                           months: List[int],
                           weights: Optional[List[float]] = None) -> Dict:
        """
        Price a strip contract (average of multiple months).
        
        Args:
            route: Route name
            months: List of contract months to include
            weights: Optional weights for each month (default: equal weights)
            
        Returns:
            Dictionary with strip pricing results
        """
        if weights is None:
            weights = [1.0 / len(months)] * len(months)
        
        if len(weights) != len(months):
            raise ValueError("Weights must have same length as months")
        
        # Price individual monthly contracts
        monthly_contracts = []
        for month in months:
            contract = self.price_monthly_average_contract(route, month, contract_type='forward')
            monthly_contracts.append(contract)
        
        # Calculate weighted average of monthly averages
        weighted_averages = np.zeros(len(monthly_contracts[0]['monthly_averages']))
        
        for i, (contract, weight) in enumerate(zip(monthly_contracts, weights)):
            weighted_averages += weight * contract['monthly_averages']
        
        strip_price = np.mean(weighted_averages)
        strip_payoffs = weighted_averages - strip_price
        
        results = {
            'strip_price': strip_price,
            'route': route,
            'months': months,
            'weights': weights,
            'weighted_averages': weighted_averages,
            'strip_payoffs': strip_payoffs,
            'strip_std': np.std(weighted_averages),
            'scenarios_count': len(weighted_averages),
            'monthly_prices': [contract['contract_price'] for contract in monthly_contracts]
        }
        
        return results
    
    def calculate_contract_greeks(self,
                                route: str,
                                contract_month: int,
                                strike_price: float,
                                contract_type: str,
                                bump_size: float = 0.01) -> Dict:
        """
        Calculate option Greeks using finite differences.
        
        Args:
            route: Route name
            contract_month: Contract month
            strike_price: Strike price
            contract_type: 'call' or 'put'
            bump_size: Size of price bump for finite differences
            
        Returns:
            Dictionary with Greeks
        """
        if contract_type not in ['call', 'put']:
            raise ValueError("Greeks only available for option contracts")
        
        # Base price
        base_contract = self.price_monthly_average_contract(route, contract_month, strike_price, contract_type)
        base_price = base_contract['contract_price']
        
        # Delta: sensitivity to underlying price
        # Approximate by bumping all scenario prices
        bumped_scenarios = self.simulation_results[route].copy()
        bumped_scenarios *= (1 + bump_size)
        
        # Temporarily replace simulation results
        original_results = self.simulation_results[route]
        self.simulation_results[route] = bumped_scenarios
        
        bumped_contract = self.price_monthly_average_contract(route, contract_month, strike_price, contract_type)
        bumped_price = bumped_contract['contract_price']
        
        # Restore original results
        self.simulation_results[route] = original_results
        
        # Calculate delta
        avg_underlying_price = np.mean(original_results[:, -1])
        delta = (bumped_price - base_price) / (avg_underlying_price * bump_size)
        
        # Gamma: second derivative (simplified approximation)
        # Would need more sophisticated bumping for accurate gamma
        gamma = 0.0  # Placeholder
        
        # Theta: time decay (simplified - would need time-bumped scenarios)
        theta = 0.0  # Placeholder
        
        # Vega: volatility sensitivity (simplified)
        vega = 0.0  # Placeholder
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'base_price': base_price,
            'underlying_price': avg_underlying_price
        }
        
        return greeks
    
    def get_pricing_summary(self, route: str) -> pd.DataFrame:
        """
        Generate a summary of contract prices for different months.
        
        Args:
            route: Route name
            
        Returns:
            DataFrame with pricing summary
        """
        months = list(range(1, 13))  # 12 months
        
        summary_data = []
        for month in months:
            try:
                forward_contract = self.price_monthly_average_contract(route, month, contract_type='forward')
                
                summary_data.append({
                    'month': month,
                    'forward_price': forward_contract['contract_price'],
                    'payoff_std': forward_contract['payoff_std'],
                    'period_days': forward_contract['contract_period_days']
                })
            except Exception as e:
                warnings.warn(f"Failed to price month {month}: {str(e)}")
                continue
        
        return pd.DataFrame(summary_data)