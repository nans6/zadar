# main_agent.py
import data_fetcher
import models.renewables_model as renewables_model
# Import other future modules placeholder:
# import models.load_forecast_model as load_model
import models.price_risk_model as risk_model  # Add price risk model import
import agent_reasoner  # Add the new import
# import strategy_engine

from pprint import pprint
from dotenv import load_dotenv

def run_agent_cycle():
    print("--- Running Agent Cycle ---")
    load_dotenv() # Ensure keys are loaded for this run

    # 1. Perception
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

    # 2. Reasoning - Quantitative Models
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


    # --- Placeholders for other models ---
    print("\nPredicting load (Placeholder)...")
    # load_prediction = load_model.predict_load(weather_data)
    load_prediction = {"predicted_peak_load_gw": 72.5, "load_vs_normal_pct": 5.0} # Dummy data
    print(load_prediction)

    print("\nAssessing price risk...")
    # Replace placeholder with actual price risk assessment
    price_risk = risk_model.assess_price_risk(
        load_prediction=load_prediction,
        renewable_metrics=renewable_metrics,
        market_data=None  # Optional: Add market data when available
    )
    print("\nPrice Risk Assessment:")
    pprint(price_risk)
    # --- End Placeholders ---

    # --- Create Input Dictionary for LLM ---
    llm_input_data = {
        "weather_summary": weather_data.get('daily_summaries', {}),
        "renewable_metrics": renewable_metrics,
        "eia_realtime_data": eia_realtime_data if eia_realtime_data else {},
        "load_prediction": load_prediction,
        "price_risk": price_risk
    }
    print("\n--- Data Sent to LLM ---")
    pprint(llm_input_data)
    print("------------------------")

    # --- 3. Reasoning - Call LLM via Replicate ---
    print("\nGenerating strategy with Replicate LLM...")
    strategy_recommendation_text = agent_reasoner.generate_strategy_via_replicate(llm_input_data)

    print(f"\n--- Raw LLM Output ---")
    print(strategy_recommendation_text)
    print("----------------------")

    # --- 4. Action - Format/Output Strategy (Placeholder) ---
    print("\nFormatting final strategy (Placeholder)...")
    print("\n--- Agent Recommendation ---")
    print(strategy_recommendation_text)
    print("---------------------------\n")

if __name__ == "__main__":
    run_agent_cycle()
