import streamlit as st
import json
from agent_reasoner import generate_strategy_via_replicate, get_load_predictions
from datetime import datetime

st.set_page_config(page_title="Energy Trading Agent", layout="wide")

st.title("Energy Trading Agent")

# Input data
input_data = {
    'weather_summary': {
        '2024-04-13': {'min_temp': 30, 'max_temp': 38, 'avg_wind': 3.5, 'solar_potential_hours': 12},
        '2024-04-14': {'min_temp': 31, 'max_temp': 39, 'avg_wind': 3.0, 'solar_potential_hours': 11},
        '2024-04-15': {'min_temp': 32, 'max_temp': 40, 'avg_wind': 2.5, 'solar_potential_hours': 10}
    },
    'renewable_metrics': {
        'average_wind_capacity_factor': 0.15,
        'wind_deviation_from_normal_pct': -45,
        'average_solar_capacity_factor_daytime': 0.75,
        'solar_deviation_from_normal_pct': 10,
        'periods_with_low_wind': 8
    },
    'eia_realtime_data': {
        'current': {
            'timestamp_iso': '2024-04-13T14:00:00',
            'demand_mw': 65000,
            'wind_mw': 5000,
            'solar_mw': 12000
        }
    },
    'price_risk': {
        'price_spike_probability': 0.65,
        'risk_level': 'HIGH',
        'risk_factors': ['Low wind generation', 'High temperatures', 'Peak demand hours']
    }
}

# Add load predictions
temp_k = 308.15  # 35°C
input_data['load_prediction'] = get_load_predictions(temp_k)

# Create two columns for buttons
col1, col2 = st.columns(2)

# Get Weather button in first column
with col1:
    if st.button("Get Weather"):
        st.subheader("☁️ Weather Forecast")
        
        # Create three rows of metrics for each day
        dates = ['2025-04-13', '2025-04-14', '2025-04-15']
        for display_date in dates:
            st.markdown(f"<h4 style='text-align: left;'>{display_date}</h4>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            # Get data using the original 2024 date keys but display 2025
            lookup_date = display_date.replace('2025', '2024')
            day_data = input_data['weather_summary'][lookup_date]
            
            with col1:
                st.metric(
                    "🌡️ Min Temp",
                    f"{day_data['min_temp']}°C"
                )
            
            with col2:
                st.metric(
                    "🌡️ Max Temp",
                    f"{day_data['max_temp']}°C"
                )
            
            with col3:
                st.metric(
                    "💨 Wind Speed",
                    f"{day_data['avg_wind']} m/s"
                )
            
            with col4:
                st.metric(
                    "☀️ Solar Hours",
                    f"{day_data['solar_potential_hours']} hrs"
                )
            
            st.markdown("<hr style='margin: 10px 0px'>", unsafe_allow_html=True)

# Generate Strategy button in second column
with col2:
    if st.button("HEDGE"):
        with st.spinner("Generating trading strategy..."):
            strategy = generate_strategy_via_replicate(input_data)
            
            # Split at </think> tags
            parts = strategy.split("</think>")
            
            if len(parts) > 1:
                # First part is the reasoning
                reasoning = parts[0].strip()
                # Second part is the actual strategy
                actual_strategy = parts[1].strip()
                
                # Put reasoning in expander
                with st.expander("Reasoning", expanded=False):
                    st.markdown(reasoning.replace('\n', '  \n'))
                
                # Display the actual strategy output with better formatting
                st.markdown("---")
                st.markdown(actual_strategy.replace('\n', '  \n'))
            else:
                # Fallback if no </think> found
                strategy_start = strategy.find("HEDGE ACTIONS:")
                if strategy_start != -1:
                    st.markdown("### Hedge Actions")
                    st.markdown(strategy[strategy_start:].strip().replace('\n', '  \n'))
                else:
                    st.markdown(strategy.replace('\n', '  \n'))
            
            st.markdown("---")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Temperature", 
                    f"{input_data['weather_summary']['2024-04-13']['avg_wind']}°C",
                    "Current"
                )
                
            with col2:
                st.metric(
                    "Wind Generation", 
                    f"{input_data['eia_realtime_data']['current']['wind_mw']} MW",
                    f"{input_data['renewable_metrics']['wind_deviation_from_normal_pct']}% from normal"
                )
                
            with col3:
                st.metric(
                    "Risk Level",
                    input_data['price_risk']['risk_level'],
                    f"Spike Prob: {input_data['price_risk']['price_spike_probability']:.0%}"
                )

# Footer
st.markdown("---")
st.markdown("Energy Trading Agent") 