# main_agent.py
import data_fetcher
import models.renewables_model as renewables_model
import models.price_risk_model as risk_model  
import agent_reasoner  
import pandas as pd


from pprint import pprint
from dotenv import load_dotenv

def run_agent_cycle():
    print("--- Running Agent Cycle ---")
    load_dotenv() 

    # Perception
    print("Fetching weather data...")
    weather_data = data_fetcher.get_weather_forecast()
    if "error" in weather_data:
        print(f"ERROR fetching weather: {weather_data['error']}")
        return # Stop cycle if essential data is missing
    
    print("\nWeather Summary:")
    if 'daily_summaries' in weather_data:
        for date, summary in weather_data['daily_summaries'].items():
            print(f"\n{date}:")
            print(f"  Temperature: {summary['min_temp']}°C to {summary['max_temp']}°C (avg: {summary['avg_temp']}°C)")
            print(f"  Wind: {summary['avg_wind']} m/s")
            print(f"  Solar Potential: {summary['solar_potential_hours']} hours")
    else:
        print("No weather summary available.")

    print("\nFetching real-time EIA grid data...")
    eia_realtime_data = data_fetcher.get_eia_realtime_grid_data(region_id="TEX")
    if eia_realtime_data:
        print("\nCurrent Grid Status:")
        current = eia_realtime_data['current']
        print(f"Timestamp: {current['timestamp_iso']}")
        print(f"Demand: {current['demand_mw']} MW")
        print(f"Wind: {current['wind_mw']} MW")
        print(f"Solar: {current['solar_mw']} MW")
    else:
        print("Warning: Could not retrieve EIA data. Proceeding without it.")

    # Reasoning - Quantitative Models
    print("\nCalculating renewable energy forecast metrics...")
    # Use detailed_forecast instead of forecasts
    renewable_metrics = {}
    if 'detailed_forecast' in weather_data:
         renewable_metrics = renewables_model.calculate_renewable_metrics(weather_data['detailed_forecast'])
         # Add latest actuals to metrics if available
         if eia_realtime_data:
              renewable_metrics['latest_actual_wind_mw'] = eia_realtime_data['current']['wind_mw']
              renewable_metrics['latest_actual_solar_mw'] = eia_realtime_data['current']['solar_mw']
              renewable_metrics['latest_actual_demand_mw'] = eia_realtime_data['current']['demand_mw']
              renewable_metrics['latest_data_timestamp'] = eia_realtime_data['current']['timestamp_iso']
         print("\nRenewable Generation Metrics:")
         pprint(renewable_metrics)
    else:
         print("Warning: No detailed forecast found in weather data to calculate renewable metrics.")

    # Get current temperature from weather data
    print("\nPredicting load...")
    current_temp_k = None
    if 'detailed_forecast' in weather_data and weather_data['detailed_forecast']:
        # Get the first forecast point's temperature
        first_forecast = weather_data['detailed_forecast'][0]
        if 'temperature' in first_forecast:
            current_temp_k = first_forecast['temperature'] + 273.15  # Convert Celsius to Kelvin
    
    if current_temp_k is None:
        print("Warning: Could not get temperature from weather data, using default")
        current_temp_k = 298.15  # 25°C in Kelvin

    load_prediction = agent_reasoner.get_load_predictions(current_temp_k)
    if load_prediction:
        print("\nLoad Prediction Results:")
        pprint(load_prediction)
    else:
        print("Warning: Load prediction failed")
        load_prediction = {"predicted_load_gw": 0, "load_vs_normal_pct": 0}

    print("\nLoading market data from CSV...")
    try:
        market_data = pd.read_csv('data/synthetic_market_data.csv').to_dict(orient='records')[0]  # Get first scenario by default
    except Exception as e:
        print(f"Warning: Could not load market data from CSV: {e}")
        market_data = {}

    print("\nAssessing price risk...")
    price_risk = risk_model.assess_price_risk(
        load_prediction=load_prediction,
        renewable_metrics=renewable_metrics,
        market_data=market_data  
    )
    print("\nPrice Risk Assessment:")
    pprint(price_risk)

    # If we have risk metrics, try to find a more appropriate market scenario
    if price_risk and 'price_spike_probability' in price_risk:
        try:
            risk_level = price_risk['price_spike_probability']
            df = pd.read_csv('data/synthetic_market_data.csv')
            # Find the scenario with the closest risk level
            closest_scenario = df.iloc[(df['risk_level'] - risk_level).abs().argsort()[:1]]
            market_data = closest_scenario.to_dict(orient='records')[0]
            print("\nUpdated Market Data (matched to risk level):")
            pprint(market_data)
        except Exception as e:
            print(f"Warning: Could not update market data based on risk level: {e}")

    llm_input_data = {
        "weather_summary": weather_data.get('daily_summaries', {}),
        "renewable_metrics": renewable_metrics,
        "eia_realtime_data": eia_realtime_data if eia_realtime_data else {},
        "load_prediction": load_prediction,
        "price_risk": price_risk,
        "market_prices": market_data
    }
    print("\n--- Data Sent to LLM ---")
    pprint(llm_input_data)

    print("\n--- Generating Strategy ---")
    strategy = agent_reasoner.generate_strategy_via_replicate(llm_input_data)
    print("\nStrategy:")
    print(strategy)

if __name__ == '__main__':
    run_agent_cycle()
