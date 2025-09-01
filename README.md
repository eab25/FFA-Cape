# ⚓ FFA Monte Carlo Simulation Engine

**Forward Freight Agreement (FFA) Market Analysis & Risk Management Platform**

A comprehensive Monte Carlo simulation engine for analyzing and pricing Forward Freight Agreement (FFA) contracts with advanced risk metrics and interactive visualizations.

![FFA Engine Overview](https://github.com/user-attachments/assets/6a76204a-2e02-453e-9ac9-747492a9cafe)

## 🚀 Features

### Core Engine Components
- **📊 Data Processing**: Historical FFA data ingestion and forward curve construction
- **🔧 Stochastic Modeling**: Mean-reverting models with seasonality and correlation structure
- **🎲 Monte Carlo Simulation**: Generate thousands of correlated price scenarios
- **💰 FFA Contract Pricing**: Price monthly average, calendar spread, and strip contracts
- **⚠️ Risk Metrics**: VaR, CVaR, and comprehensive risk analytics

### Interactive Streamlit Interface
- **Multi-page Navigation**: Data overview, model calibration, simulation, pricing, and risk analysis
- **Real-time Visualizations**: Price evolution charts, correlation heatmaps, and distribution plots
- **Configurable Parameters**: Adjust scenarios, time horizons, correlations, and model settings
- **Export Capabilities**: Download results as CSV and generate detailed reports

![Monte Carlo Simulation](https://github.com/user-attachments/assets/e7772a47-db0f-4c32-a09d-df12bfa0350d)

## 📈 Supported Analysis

### Historical Data Analysis
- Baltic TC index data (C5TC, HS7TC, P4TC, S10TC routes)
- Correlation analysis across routes and time periods
- Statistical summaries and data quality validation

### Stochastic Model Calibration
- **Mean Reversion**: Ornstein-Uhlenbeck process parameters (κ, θ, σ)
- **Seasonality Detection**: Monthly seasonal factors and patterns
- **Correlation Structure**: Cross-route correlation modeling

### Monte Carlo Simulation
- Up to 10,000 scenarios with configurable time horizons
- Correlated price paths using Cholesky decomposition
- Multiple simulation modes (independent vs. correlated)

### FFA Contract Pricing
- **Forward Contracts**: Monthly average settlement pricing
- **Options**: Call and put options with strike price sensitivity
- **Spreads**: Calendar spreads between contract months
- **Strips**: Weighted averages across multiple months

### Risk Analytics
- **Value at Risk (VaR)**: 90%, 95%, 99% confidence levels
- **Conditional VaR (CVaR)**: Expected shortfall calculations
- **Distribution Analysis**: Skewness, kurtosis, and tail risk
- **Scenario Statistics**: Percentile analysis and stress testing

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- `streamlit` - Interactive web application framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scipy` - Scientific computing and optimization
- `scikit-learn` - Machine learning algorithms
- `seaborn` - Statistical data visualization
- `matplotlib` - Plotting and visualization

## 🚀 Quick Start

### 1. Launch the Application
```bash
streamlit run streamlit_app.py
```

### 2. Navigate the Interface
1. **Data Overview**: Review historical data and correlations
2. **Model Calibration**: Calibrate stochastic models for all routes
3. **Monte Carlo Simulation**: Configure and run price simulations
4. **FFA Contract Pricing**: Price specific contracts and analyze payoffs
5. **Risk Analysis**: Calculate comprehensive risk metrics

### 3. Example Workflow
```python
from ffa_engine import DataProcessor, StochasticModel, MonteCarloEngine, FFAPricer, RiskMetrics
import pandas as pd

# Load data
data = pd.read_csv('Baltic TC index all vsl')
processor = DataProcessor()
processed_data = processor.load_data(data)

# Calibrate model
model = StochasticModel()
model.calibrate_all_routes(processed_data)

# Run simulation
initial_prices = {route: processed_data[route].iloc[-1] for route in model.routes}
mc_engine = MonteCarloEngine(model)
results = mc_engine.run_simulation(initial_prices, time_horizon_days=252, n_scenarios=5000)

# Price contracts
pricer = FFAPricer(results)
contract = pricer.price_monthly_average_contract('C5TC', contract_month=6)

# Calculate risk metrics
risk_calc = RiskMetrics(results)
risk_metrics = risk_calc.calculate_risk_metrics_for_route('C5TC')
```

## 📊 Data Format

The engine expects CSV data with the following structure:
```csv
Date,C5TC,HS7TC,P4TC,S10TC
02-Jan-2020,11976,8376,7695,7539
03-Jan-2020,10825,8139,7201,7277
...
```

- **Date**: DD-MMM-YYYY format
- **Route Columns**: Numeric price data for each route
- **Missing Data**: Handled automatically with interpolation

## 🔧 Architecture

### Module Structure
```
ffa_engine/
├── __init__.py              # Package initialization
├── data_processor.py        # Data ingestion and preprocessing
├── stochastic_model.py      # Model calibration and dynamics
├── monte_carlo.py           # Simulation engine
├── ffa_pricer.py           # Contract pricing logic
└── risk_metrics.py         # Risk calculations
```

### Key Classes
- **`DataProcessor`**: Historical data management and forward curve construction
- **`StochasticModel`**: Mean-reverting model with seasonality calibration
- **`MonteCarloEngine`**: Scenario generation with correlation handling
- **`FFAPricer`**: Contract pricing for forwards, options, and spreads
- **`RiskMetrics`**: VaR, CVaR, and comprehensive risk analytics

## 📈 Model Features

### Mean-Reverting Dynamics
```
dX = κ(θ + S(t) - X)dt + σdW
```
- **κ**: Mean reversion speed
- **θ**: Long-term mean level
- **S(t)**: Seasonal adjustment
- **σ**: Volatility parameter

### Correlation Modeling
- Cross-route correlation estimation
- Cholesky decomposition for scenario generation
- Dynamic correlation adjustment capabilities

### Contract Types
- **Monthly Average Forwards**: E[Average(S_t) over month]
- **Call Options**: E[max(Average(S_t) - K, 0)]
- **Put Options**: E[max(K - Average(S_t), 0)]
- **Calendar Spreads**: Long/short positions across months
- **Strip Contracts**: Weighted averages across periods

## 🎯 Use Cases

### Trading & Risk Management
- Portfolio risk assessment and VaR calculation
- Contract pricing and mark-to-market valuation
- Scenario analysis and stress testing
- Correlation trading strategy development

### Research & Analysis
- Market structure analysis and seasonal pattern detection
- Model validation and backtesting frameworks
- Academic research on freight market dynamics
- Regulatory capital calculation support

### Business Intelligence
- Forward curve construction and market outlook
- Client reporting and presentation materials
- Investment decision support analytics
- Market making and liquidity provision tools

## 📚 Technical Documentation

### Performance Optimization
- Vectorized numpy operations for speed
- Efficient memory management for large simulations
- Parallel processing capabilities for multiple routes
- Caching mechanisms for repeated calculations

### Validation & Testing
- Statistical model validation against historical data
- Monte Carlo convergence testing
- Cross-validation for out-of-sample performance
- Unit tests for all core components

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-analysis`
3. **Commit changes**: `git commit -am 'Add new risk metric'`
4. **Push to branch**: `git push origin feature/new-analysis`
5. **Submit pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Baltic Exchange for providing freight rate benchmarks
- Streamlit community for the excellent web framework
- Scientific Python ecosystem (NumPy, SciPy, Pandas)
- Financial modeling research community

---

**Built with ❤️ for the freight derivatives community**
