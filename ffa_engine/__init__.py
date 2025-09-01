"""
FFA (Forward Freight Agreement) Monte Carlo Simulation Engine

This package provides tools for:
- Historical data processing and forward curve construction
- Stochastic model calibration (mean-reversion, seasonality, correlations)
- Monte Carlo simulation of freight rate paths
- FFA contract pricing and risk metrics calculation
"""

__version__ = "1.0.0"
__author__ = "FFA Cape Team"

from .data_processor import DataProcessor
from .stochastic_model import StochasticModel
from .monte_carlo import MonteCarloEngine
from .ffa_pricer import FFAPricer
from .risk_metrics import RiskMetrics

__all__ = [
    'DataProcessor',
    'StochasticModel', 
    'MonteCarloEngine',
    'FFAPricer',
    'RiskMetrics'
]