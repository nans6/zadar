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

# Create two columns for buttons with adjusted widths
col1, col2 = st.columns([1, 1.2])  # Make the second column slightly wider

# Get Weather button in first column
with col1:
    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] > div:first-child button {
            background-color: #f0f8ff;
            border: 1px solid #3498db;
            color: #2c3e50;
        }
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
            background-color: #3498db;
            border: none;
            color: white;
            font-size: 1.2em;
            padding: 0.5em 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if st.button("Get Weather", use_container_width=True):
        st.markdown("""
        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px;'>
            <h3 style='color: #2c3e50; margin-bottom: 20px;'>☁️ Weather Forecast</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create three rows of metrics for each day
        dates = ['2025-04-13', '2025-04-14', '2025-04-15']
        for display_date in dates:
            # Calculate if it's a hot day
            lookup_date = display_date.replace('2025', '2024')
            day_data = input_data['weather_summary'][lookup_date]
            is_hot = day_data['max_temp'] > 35
            is_windy = day_data['avg_wind'] > 3.0
            good_solar = day_data['solar_potential_hours'] > 10
            
            # Date header with weather summary icons
            weather_icons = []
            if is_hot:
                weather_icons.append("🌡️")
            if is_windy:
                weather_icons.append("💨")
            if good_solar:
                weather_icons.append("☀️")
            
            st.markdown(f"""
            <div style='background-color: #ffffff; padding: 10px; border-radius: 5px; margin: 10px 0px; border-left: 4px solid #3498db;'>
                <h4 style='color: #2c3e50; margin: 0;'>{display_date} {' '.join(weather_icons)}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🌡️ Min",
                    f"{day_data['min_temp']}°C",
                    delta=None,
                    help="Minimum temperature for the day"
                )
            
            with col2:
                delta_color = "inverse" if is_hot else "normal"
                st.metric(
                    "🌡️ Max",
                    f"{day_data['max_temp']}°C",
                    delta="Hot" if is_hot else None,
                    delta_color=delta_color,
                    help="Maximum temperature for the day"
                )
            
            with col3:
                delta_color = "normal" if is_windy else "inverse"
                st.metric(
                    "💨 Wind",
                    f"{day_data['avg_wind']} m/s",
                    delta="Strong" if is_windy else "Light",
                    delta_color=delta_color,
                    help="Average wind speed"
                )
            
            with col4:
                delta_color = "normal" if good_solar else "inverse"
                st.metric(
                    "☀️ Solar",
                    f"{day_data['solar_potential_hours']} hrs",
                    delta="Optimal" if good_solar else "Limited",
                    delta_color=delta_color,
                    help="Hours of good solar potential"
                )
            
            # Add a more subtle divider
            if display_date != dates[-1]:  # Don't add after last date
                st.markdown("<hr style='margin: 10px 0px; opacity: 0.2;'>", unsafe_allow_html=True)
        
        # Add a summary box at the bottom
        total_solar = sum(input_data['weather_summary'][d.replace('2025', '2024')]['solar_potential_hours'] for d in dates)
        avg_wind = sum(input_data['weather_summary'][d.replace('2025', '2024')]['avg_wind'] for d in dates) / len(dates)
        
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; border: 1px solid #dee2e6;'>
            <p style='color: #2c3e50; margin: 0;'>
                <strong>3-Day Summary:</strong> Total Solar Hours: {total_solar}hrs | Average Wind: {avg_wind:.1f}m/s
            </p>
        </div>
        """, unsafe_allow_html=True)

# HEDGE button in second column
with col2:
    if st.button("HEDGE", use_container_width=True):
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