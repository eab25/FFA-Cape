"""
Data Processing Module for FFA Engine

Handles historical data ingestion, forward curve construction, 
and data validation for the FFA Monte Carlo simulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings


class DataProcessor:
    """
    Processes historical FFA data and constructs forward curves.
    """
    
    def __init__(self):
        self.data = None
        self.routes = None
        self.forward_curves = None
        
    def load_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Load and validate historical FFA data.
        
        Args:
            data: DataFrame with Date column and route columns
            
        Returns:
            Cleaned and validated DataFrame
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        # Convert date column to datetime
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%Y', errors='coerce')
        
        # Remove rows with invalid dates
        data = data.dropna(subset=['Date'])
        
        # Sort by date
        data = data.sort_values('Date').reset_index(drop=True)
        
        # Get route columns (all columns except Date)
        self.routes = [col for col in data.columns if col != 'Date']
        
        # Validate numeric data
        for route in self.routes:
            data[route] = pd.to_numeric(data[route], errors='coerce')
        
        # Remove rows with all NaN route values
        data = data.dropna(subset=self.routes, how='all')
        
        self.data = data
        return data
    
    def get_returns(self) -> pd.DataFrame:
        """
        Calculate log returns for all routes.
        
        Returns:
            DataFrame with log returns
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        returns_data = self.data.copy()
        
        for route in self.routes:
            # Calculate log returns
            returns_data[f'{route}_return'] = np.log(returns_data[route] / returns_data[route].shift(1))
        
        return returns_data.dropna()
    
    def construct_forward_curves(self, tenors: List[int] = [1, 2, 3, 6, 12]) -> Dict[str, pd.DataFrame]:
        """
        Construct forward curves for different tenors (months ahead).
        
        Args:
            tenors: List of tenor months [1, 2, 3, 6, 12]
            
        Returns:
            Dictionary of forward curves by route
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        forward_curves = {}
        
        for route in self.routes:
            route_curves = self.data[['Date', route]].copy()
            
            # Create forward curve columns for each tenor
            for tenor in tenors:
                # Simple forward curve approximation: shift data by tenor months
                route_curves[f'F{tenor}M'] = route_curves[route].shift(-tenor * 22)  # Approx 22 trading days per month
            
            # Remove rows with NaN forward values
            route_curves = route_curves.dropna()
            forward_curves[route] = route_curves
        
        self.forward_curves = forward_curves
        return forward_curves
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between routes.
        
        Returns:
            Correlation matrix DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Calculate correlations on price levels
        price_data = self.data[self.routes]
        correlation_matrix = price_data.corr()
        
        return correlation_matrix
    
    def get_statistics(self) -> Dict:
        """
        Get basic statistics for the dataset.
        
        Returns:
            Dictionary with data statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        stats = {
            'data_range': {
                'start_date': self.data['Date'].min(),
                'end_date': self.data['Date'].max(),
                'total_days': len(self.data)
            },
            'routes': self.routes,
            'price_statistics': self.data[self.routes].describe(),
            'missing_data': self.data[self.routes].isnull().sum()
        }
        
        return stats