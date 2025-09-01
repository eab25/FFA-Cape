import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from ffa_engine import DataProcessor, StochasticModel, MonteCarloEngine, FFAPricer, RiskMetrics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="FFA Monte Carlo Simulation Engine",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_csv(file) -> pd.DataFrame:
    """Load CSV file with error handling."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

def load_default_data() -> pd.DataFrame:
    """Load the default Baltic TC data."""
    try:
        return pd.read_csv('Baltic TC index all vsl')
    except Exception as e:
        st.error(f"Error loading default data: {e}")
        return pd.DataFrame()

def show_data_overview(df: pd.DataFrame, processor: DataProcessor):
    """Display data overview and statistics."""
    st.subheader("📊 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Summary:**")
        stats = processor.get_statistics()
        st.write(f"- Date range: {stats['data_range']['start_date'].strftime('%Y-%m-%d')} to {stats['data_range']['end_date'].strftime('%Y-%m-%d')}")
        st.write(f"- Total observations: {stats['data_range']['total_days']}")
        st.write(f"- Routes: {', '.join(stats['routes'])}")
    
    with col2:
        st.write("**Correlation Matrix:**")
        corr_matrix = processor.get_correlation_matrix()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax)
        ax.set_title("Route Correlations")
        st.pyplot(fig)
    
    # Price evolution chart
    st.write("**Historical Price Evolution:**")
    fig, ax = plt.subplots(figsize=(12, 6))
    for route in processor.routes:
        ax.plot(df['Date'], df[route], label=route, linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("FFA Route Prices Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def show_model_calibration(df: pd.DataFrame, model: StochasticModel):
    """Display model calibration results."""
    st.subheader("🔧 Model Calibration")
    
    # Calibrate the model
    with st.spinner("Calibrating stochastic models..."):
        calibrated_params = model.calibrate_all_routes(df)
    
    st.success("Model calibration completed!")
    
    # Display calibration results
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Mean Reversion Parameters:**")
        mr_data = []
        for route, params in calibrated_params.items():
            mr_params = params['mean_reversion']
            mr_data.append({
                'Route': route,
                'Kappa (Mean Reversion Speed)': f"{mr_params['kappa']:.4f}",
                'Theta (Long-term Mean)': f"{mr_params['theta']:.4f}",
                'Sigma (Volatility)': f"{mr_params['sigma']:.4f}"
            })
        st.dataframe(pd.DataFrame(mr_data))
    
    with col2:
        st.write("**Seasonality Analysis:**")
        seasonal_data = []
        for route, params in calibrated_params.items():
            seasonal_params = params['seasonality']
            seasonal_data.append({
                'Route': route,
                'Seasonal Strength': f"{seasonal_params['seasonal_strength']:.4f}",
                'Overall Mean (Log)': f"{seasonal_params['overall_mean']:.4f}",
                'Seasonal Volatility': f"{seasonal_params['seasonal_volatility']:.4f}"
            })
        st.dataframe(pd.DataFrame(seasonal_data))
    
    return calibrated_params

def run_monte_carlo_simulation(df: pd.DataFrame, model: StochasticModel):
    """Run Monte Carlo simulation with user parameters."""
    st.subheader("🎲 Monte Carlo Simulation")
    
    # Simulation parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_scenarios = st.selectbox("Number of Scenarios", [1000, 5000, 10000], index=1)
        time_horizon = st.selectbox("Time Horizon (Days)", [63, 126, 252], index=2)  # ~3M, 6M, 1Y
    
    with col2:
        use_correlation = st.checkbox("Use Route Correlations", value=True)
        random_seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)
    
    with col3:
        selected_routes = st.multiselect("Routes to Simulate", 
                                       options=model.routes, 
                                       default=model.routes)
    
    if st.button("Run Simulation", type="primary"):
        if not selected_routes:
            st.error("Please select at least one route to simulate.")
            return None
        
        # Get initial prices (latest available)
        initial_prices = {}
        for route in selected_routes:
            initial_prices[route] = df[route].iloc[-1]
        
        # Prepare correlation matrix if needed
        correlation_matrix = None
        if use_correlation:
            full_corr = model.get_correlation_structure(df)
            # Filter correlation matrix for selected routes
            route_indices = [model.routes.index(route) for route in selected_routes]
            correlation_matrix = full_corr[np.ix_(route_indices, route_indices)]
        
        # Run simulation
        mc_engine = MonteCarloEngine(model)
        
        with st.spinner(f"Running {n_scenarios:,} Monte Carlo scenarios..."):
            simulation_results = mc_engine.run_simulation(
                initial_prices=initial_prices,
                time_horizon_days=time_horizon,
                n_scenarios=n_scenarios,
                correlation_matrix=correlation_matrix,
                random_seed=random_seed
            )
        
        st.success(f"Simulation completed! Generated {n_scenarios:,} scenarios over {time_horizon} days.")
        
        return mc_engine, simulation_results
    
    return None, None

def show_simulation_results(mc_engine: MonteCarloEngine, simulation_results: dict):
    """Display simulation results and analysis."""
    st.subheader("📈 Simulation Results")
    
    # Route selection for detailed analysis
    selected_route = st.selectbox("Select Route for Detailed Analysis", 
                                options=list(simulation_results.keys()))
    
    if selected_route:
        col1, col2 = st.columns(2)
        
        with col1:
            # Scenario statistics
            st.write("**Scenario Statistics:**")
            stats = mc_engine.get_scenario_statistics(selected_route)
            stats_df = pd.DataFrame([
                {"Metric": "Mean Final Price", "Value": f"${stats['mean_final_price']:.2f}"},
                {"Metric": "Std Final Price", "Value": f"${stats['std_final_price']:.2f}"},
                {"Metric": "Min Final Price", "Value": f"${stats['min_final_price']:.2f}"},
                {"Metric": "Max Final Price", "Value": f"${stats['max_final_price']:.2f}"},
                {"Metric": "95th Percentile", "Value": f"${stats['percentiles']['p95']:.2f}"},
                {"Metric": "5th Percentile", "Value": f"${stats['percentiles']['p5']:.2f}"}
            ])
            st.dataframe(stats_df)
        
        with col2:
            # Price distribution
            st.write("**Final Price Distribution:**")
            final_prices = simulation_results[selected_route][:, -1]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(final_prices, bins=50, alpha=0.7, density=True, color='skyblue')
            ax.axvline(np.mean(final_prices), color='red', linestyle='--', label='Mean')
            ax.axvline(np.percentile(final_prices, 5), color='orange', linestyle='--', label='5th Percentile')
            ax.axvline(np.percentile(final_prices, 95), color='orange', linestyle='--', label='95th Percentile')
            ax.set_xlabel("Final Price")
            ax.set_ylabel("Density")
            ax.set_title(f"{selected_route} Final Price Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Path evolution
        st.write("**Price Path Evolution:**")
        path_stats = mc_engine.get_path_statistics(selected_route)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(path_stats['day'], path_stats['p5'], path_stats['p95'], 
                       alpha=0.2, color='lightblue', label='90% Confidence Band')
        ax.fill_between(path_stats['day'], path_stats['p25'], path_stats['p75'], 
                       alpha=0.3, color='lightblue', label='50% Confidence Band')
        ax.plot(path_stats['day'], path_stats['mean'], color='red', linewidth=2, label='Mean Path')
        ax.plot(path_stats['day'], path_stats['p50'], color='blue', linewidth=2, label='Median Path')
        
        # Plot a few individual scenarios
        scenarios = simulation_results[selected_route]
        for i in range(min(5, scenarios.shape[0])):
            ax.plot(path_stats['day'], scenarios[i], alpha=0.3, color='gray', linewidth=0.5)
        
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.set_title(f"{selected_route} Simulated Price Paths")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def show_ffa_pricing(mc_engine: MonteCarloEngine, simulation_results: dict):
    """Display FFA contract pricing interface."""
    st.subheader("💰 FFA Contract Pricing")
    
    pricer = FFAPricer(simulation_results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Contract Specifications:**")
        pricing_route = st.selectbox("Route for Pricing", options=list(simulation_results.keys()))
        contract_type = st.selectbox("Contract Type", ["forward", "call", "put"])
        contract_month = st.slider("Contract Month", min_value=1, max_value=12, value=6)
        
        if contract_type in ["call", "put"]:
            # Get current price for strike reference
            current_price = simulation_results[pricing_route][0, 0]  # Initial price
            strike_price = st.number_input("Strike Price", 
                                         min_value=current_price * 0.5, 
                                         max_value=current_price * 1.5, 
                                         value=current_price)
        else:
            strike_price = None
    
    with col2:
        if st.button("Price Contract", type="primary"):
            try:
                contract_result = pricer.price_monthly_average_contract(
                    route=pricing_route,
                    contract_month=contract_month,
                    strike_price=strike_price,
                    contract_type=contract_type
                )
                
                st.success("Contract priced successfully!")
                
                # Display pricing results
                st.write("**Pricing Results:**")
                pricing_df = pd.DataFrame([
                    {"Metric": "Contract Price", "Value": f"${contract_result['contract_price']:.2f}"},
                    {"Metric": "Expected Payoff", "Value": f"${contract_result['expected_payoff']:.2f}"},
                    {"Metric": "Payoff Std Dev", "Value": f"${contract_result['payoff_std']:.2f}"},
                    {"Metric": "Contract Period", "Value": f"{contract_result['contract_period_days']} days"},
                    {"Metric": "Scenarios Used", "Value": f"{contract_result['scenarios_count']:,}"}
                ])
                st.dataframe(pricing_df)
                
                # Payoff distribution
                st.write("**Payoff Distribution:**")
                fig, ax = plt.subplots(figsize=(10, 5))
                payoffs = contract_result['payoffs']
                ax.hist(payoffs, bins=50, alpha=0.7, density=True, color='lightgreen')
                ax.axvline(np.mean(payoffs), color='red', linestyle='--', label='Mean Payoff')
                ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='Break-even')
                ax.set_xlabel("Payoff")
                ax.set_ylabel("Density")
                ax.set_title(f"{contract_type.title()} Contract Payoff Distribution")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error pricing contract: {e}")
    
    # Forward curve summary
    st.write("**Forward Curve Summary:**")
    if st.button("Generate Forward Curve"):
        try:
            summary = pricer.get_pricing_summary(pricing_route)
            st.dataframe(summary)
            
            # Plot forward curve
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(summary['month'], summary['forward_price'], marker='o', linewidth=2)
            ax.set_xlabel("Contract Month")
            ax.set_ylabel("Forward Price")
            ax.set_title(f"{pricing_route} Forward Curve")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error generating forward curve: {e}")

def show_risk_metrics(mc_engine: MonteCarloEngine, simulation_results: dict):
    """Display comprehensive risk metrics."""
    st.subheader("⚠️ Risk Metrics")
    
    risk_calculator = RiskMetrics(simulation_results)
    
    # Route selection
    risk_route = st.selectbox("Route for Risk Analysis", options=list(simulation_results.keys()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Calculate Risk Metrics", type="primary"):
            try:
                risk_metrics = risk_calculator.calculate_risk_metrics_for_route(risk_route)
                
                st.success("Risk metrics calculated!")
                
                # Basic risk metrics
                st.write("**Value at Risk (VaR) Metrics:**")
                var_df = pd.DataFrame([
                    {"Confidence Level": "95%", "VaR": f"{risk_metrics['var_95']:.4f}", "CVaR": f"{risk_metrics['cvar_95']:.4f}"},
                    {"Confidence Level": "99%", "VaR": f"{risk_metrics['var_99']:.4f}", "CVaR": f"{risk_metrics['cvar_99']:.4f}"}
                ])
                st.dataframe(var_df)
                
                # Distribution metrics
                st.write("**Distribution Metrics:**")
                dist_df = pd.DataFrame([
                    {"Metric": "Mean Return", "Value": f"{risk_metrics['mean_return']:.4f}"},
                    {"Metric": "Volatility", "Value": f"{risk_metrics['volatility']:.4f}"},
                    {"Metric": "Skewness", "Value": f"{risk_metrics['skewness']:.4f}"},
                    {"Metric": "Kurtosis", "Value": f"{risk_metrics['kurtosis']:.4f}"}
                ])
                st.dataframe(dist_df)
                
            except Exception as e:
                st.error(f"Error calculating risk metrics: {e}")
    
    with col2:
        # Risk visualization would go here
        st.write("**Risk Visualization:**")
        st.info("Risk charts will be displayed after calculation.")

def main():
    """Main Streamlit application."""
    st.title("⚓ FFA Monte Carlo Simulation Engine")
    st.markdown("""
    **Forward Freight Agreement (FFA) Market Analysis & Risk Management Platform**
    
    This application provides comprehensive tools for analyzing freight markets through:
    - Historical data analysis and model calibration
    - Monte Carlo simulation of price scenarios  
    - FFA contract pricing and risk metrics
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose Analysis", [
        "Data Overview",
        "Model Calibration", 
        "Monte Carlo Simulation",
        "FFA Contract Pricing",
        "Risk Analysis"
    ])
    
    # Data loading section
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio("Select Data Source", ["Default Baltic TC Data", "Upload CSV"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            df = load_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file to proceed.")
            return
    else:
        df = load_default_data()
    
    if df.empty:
        st.error("No data available. Please check your data source.")
        return
    
    # Initialize data processor
    processor = DataProcessor()
    processed_df = processor.load_data(df)
    
    # Store objects in session state for persistence
    if 'stochastic_model' not in st.session_state:
        st.session_state.stochastic_model = StochasticModel()
    
    if 'mc_engine' not in st.session_state:
        st.session_state.mc_engine = None
    
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    
    # Page routing
    if page == "Data Overview":
        show_data_overview(processed_df, processor)
    
    elif page == "Model Calibration":
        calibrated_params = show_model_calibration(processed_df, st.session_state.stochastic_model)
    
    elif page == "Monte Carlo Simulation":
        mc_engine, simulation_results = run_monte_carlo_simulation(processed_df, st.session_state.stochastic_model)
        if mc_engine and simulation_results:
            st.session_state.mc_engine = mc_engine
            st.session_state.simulation_results = simulation_results
            show_simulation_results(mc_engine, simulation_results)
    
    elif page == "FFA Contract Pricing":
        if st.session_state.simulation_results:
            show_ffa_pricing(st.session_state.mc_engine, st.session_state.simulation_results)
        else:
            st.warning("Please run Monte Carlo simulation first.")
    
    elif page == "Risk Analysis":
        if st.session_state.simulation_results:
            show_risk_metrics(st.session_state.mc_engine, st.session_state.simulation_results)
        else:
            st.warning("Please run Monte Carlo simulation first.")


if __name__ == "__main__":
    main()


