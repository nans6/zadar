# agent_reasoner.py
import replicate
import os
from dotenv import load_dotenv
from typing import Dict, Any
from pprint import pprint # For testing output
import pandas as pd

# Load environment variables
load_dotenv()

# Configure Replicate client
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    print("CRITICAL WARNING: REPLICATE_API_TOKEN not found in environment variables. LLM calls will fail.")

def generate_strategy_via_replicate(llm_input_data: Dict[str, Any]) -> str:
    """
    Generates hedging strategy recommendations using an LLM on Replicate.

    Args:
        llm_input_data: Dictionary containing structured inputs:
                        'weather_summary', 'renewable_metrics',
                        'eia_realtime_data', 'load_prediction', 'price_risk'.

    Returns:
        String with LLM's strategy recommendation or an error message.
    """
    if not REPLICATE_API_TOKEN:
         return "Error: Replicate API Token not configured."

    print("Formatting prompt for Replicate LLM...")

    # --- 1. Prepare Data Snippets for Prompt ---
    weather_summaries = llm_input_data.get('weather_summary', {})
    weather_str = ""
    if isinstance(weather_summaries, dict):
         days = list(weather_summaries.keys())[:3]
         for day in days:
             summary = weather_summaries.get(day, {})
             weather_str += (f"\n  - {day}: Temp {summary.get('min_temp','N/A')} to {summary.get('max_temp','N/A')} C; "
                             f"Avg Wind {summary.get('avg_wind','N/A')} m/s; Solar Hrs {summary.get('solar_potential_hours','N/A')}")
    else: weather_str = "Not Available"

    renew_metrics = llm_input_data.get('renewable_metrics', {})
    wind_dev = renew_metrics.get('wind_deviation_from_normal_pct','N/A')
    risk_prob_raw = llm_input_data.get('price_risk', {}).get('price_spike_probability', 'N/A')
    risk_prob_str = f"{risk_prob_raw*100:.0f}%" if isinstance(risk_prob_raw, (int, float)) else "N/A"

    renew_str = (f"Avg Wind CF {renew_metrics.get('average_wind_capacity_factor','N/A')} (Dev: {wind_dev}%), "
                 f"Avg Solar CF {renew_metrics.get('average_solar_capacity_factor_daytime','N/A')} (Dev: {renew_metrics.get('solar_deviation_from_normal_pct','N/A')}%), "
                 f"Low Wind Periods: {renew_metrics.get('periods_with_low_wind','N/A')}")

    eia_data = llm_input_data.get('eia_realtime_data', {}).get('current', {})
    eia_str = (f"Timestamp: {eia_data.get('timestamp_iso', 'N/A')}, "
               f"Demand: {eia_data.get('demand_mw', 'N/A')} MW, "
               f"Wind Gen: {eia_data.get('wind_mw', 'N/A')} MW, "
               f"Solar Gen: {eia_data.get('solar_mw', 'N/A')} MW")

    load_pred = llm_input_data.get('load_prediction', {})
    load_str = (f"Peak: {load_pred.get('predicted_peak_load_gw','N/A')} GW "
                f"({load_pred.get('load_vs_normal_pct','N/A')}% vs Normal)")

    price_risk_pred = llm_input_data.get('price_risk', {})
    risk_str = (f"Prob. Spike > $500/MWh: {risk_prob_str}, "
                f"Expected Max: ${price_risk_pred.get('expected_max_price','N/A')}/MWh")

    # Format market data
    market_data = llm_input_data.get('market_prices', {})
    market_str = ""
    if market_data:
        # Calculate reference price (North Hub Peak) for option strikes
        ref_price = market_data.get('north_peak', 0)
        atm_strike = ref_price
        otm_strike = ref_price + 50 if ref_price > 0 else 'N/A'
        
        market_str = f"""
*   **Current Market Prices and Products:**
    NATURAL GAS:
    - NG.FUT | Price: ${market_data.get('nat_gas_price', 'N/A')}/MMBtu | Benchmark Henry Hub Natural Gas

    POWER FUTURES ($/MWh):
    - NTH.PK | North Hub Peak: ${market_data.get('north_peak', 'N/A')} | Reference: ERCOT North
    - NTH.OP | North Hub Off-Peak: ${market_data.get('north_offpeak', 'N/A')}
    - HOU.PK | Houston Hub Peak: ${market_data.get('houston_peak', 'N/A')} | Reference: ERCOT Houston
    - HOU.OP | Houston Hub Off-Peak: ${market_data.get('houston_offpeak', 'N/A')}
    - STH.PK | South Hub Peak: ${market_data.get('south_peak', 'N/A')} | Reference: ERCOT South
    - STH.OP | South Hub Off-Peak: ${market_data.get('south_offpeak', 'N/A')}
    - WST.PK | West Hub Peak: ${market_data.get('west_peak', 'N/A')} | Reference: ERCOT West
    - WST.OP | West Hub Off-Peak: ${market_data.get('west_offpeak', 'N/A')}

    WEATHER DERIVATIVES:
    - DAL.CDD | Dallas Cooling Days: ${market_data.get('dallas_cdd', 'N/A')} | Temperature > 65°F
    - DAL.HDD | Dallas Heating Days: ${market_data.get('dallas_hdd', 'N/A')} | Temperature < 65°F
    - HOU.CDD | Houston Cooling Days: ${market_data.get('houston_cdd', 'N/A')} | Temperature > 65°F
    - HOU.HDD | Houston Heating Days: ${market_data.get('houston_hdd', 'N/A')} | Temperature < 65°F

    POWER OPTIONS ($/MWh):
    - OPT.ATM.C | Premium: ${market_data.get('atm_call_premium', 'N/A')} | Strike: ${atm_strike} | North Hub Peak ATM Call
    - OPT.ATM.P | Premium: ${market_data.get('atm_put_premium', 'N/A')} | Strike: ${atm_strike} | North Hub Peak ATM Put
    - OPT.OTM.C | Premium: ${market_data.get('otm_call_premium', 'N/A')} | Strike: ${otm_strike} | North Hub Peak OTM Call"""
    else:
        market_str = "Not Available"

    # --- 2. Construct the Prompt with Specific Renewable Switch Instruction ---
    prompt_text = f"""
System: You are an expert ERCOT energy risk hedging assistant. Analyze the data and provide a CONCISE recommendation. Skip detailed explanations and focus on actionable items.

**ERCOT Situation Analysis:**
*   **Weather Forecast Summary (Next ~3 Days):** {weather_str}
*   **Renewable Generation Forecast Metrics:** {renew_str}
*   **Current Grid State (from EIA):** {eia_str}
*   **Load Forecast:** {load_str}
*   **Price Risk Assessment:** {risk_str}
{market_str}

**Market Data Usage Guide:**
1. Natural Gas (NG.FUT):
   - Base fuel for power generation
   - Rule of thumb: Power Price ≈ (Gas Price × 7) + Risk Premium
   - Higher gas prices typically mean higher power prices

2. Power Hub Spreads:
   - North vs Houston: Congestion between regions
   - West vs Other Hubs: Wind generation impact
   - Peak vs Off-Peak: Load/Solar impact
   - Trading Opportunity: Buy low hub, sell high hub

3. Weather Derivatives:
   - CDD (Cooling): Higher value = More AC demand = Higher prices
   - HDD (Heating): Higher value = More heating = Higher gas demand
   - Use to hedge temperature-driven demand spikes
   - Dallas vs Houston spread indicates regional weather patterns

4. Options Strategy (All based on North Hub Peak):
   - ATM Calls/Puts: Strike price = Current North Hub Peak price
   - OTM Calls: Strike price = Current North Hub Peak price + $50
   - Premium vs Strike spread indicates market's view of risk
   - High Put/Call ratio suggests market expects downside

**Required Output Format:**
1. HEDGE ACTIONS: [List 1-2 specific actions using exact ticker symbols, quantities, and both premium & strike prices for options]
2. RENEWABLE SWITCH: [YES/NO] - Required if wind deviation < -25% AND price risk > 60%
   Current values: Wind Dev = {wind_dev}%, Price Risk = {risk_prob_str}
3. JUSTIFICATION: Reference specific tickers, spreads, and strikes in your rationale.
   Explain product selection based on market conditions.

Keep the entire response under 200 words. Be direct and specific.
"""

    # --- 3. Choose Model and Define Input ---
    model_identifier = "deepseek-ai/deepseek-r1"

    input_data = {
        "prompt": prompt_text,
        "max_tokens": 2048,  # Increased for safety
        "temperature": 0.3,  # Lower temp for more focused responses
        "top_p": 0.9,
        "presence_penalty": 0.1,  # Slight penalty to avoid repetition
        "frequency_penalty": 0.1
    }

    # --- 4. Call Replicate API ---
    try:
        print(f"Calling Replicate API for model: {model_identifier}...")
        output_iterator = replicate.stream(model_identifier, input=input_data)
        llm_output = "".join([str(event) for event in output_iterator])
        print("Replicate Response Received.")
        return llm_output.strip()
    except Exception as e:
        print(f"ERROR calling Replicate API: {type(e).__name__} - {e}")
        # Add hints based on common errors
        if REPLICATE_API_TOKEN and ("authenticate" in str(e).lower() or "token" in str(e).lower()):
             print("Hint: Double-check your REPLICATE_API_TOKEN value in the .env file and on Replicate.")
        elif not REPLICATE_API_TOKEN:
             print("Hint: REPLICATE_API_TOKEN environment variable is not set.")
        elif "Model" in str(e) and "not found" in str(e):
             print(f"Hint: Check if model identifier '{model_identifier}' is correct on Replicate.")
        return f"Error generating strategy via Replicate: {type(e).__name__} - {e}"

# --- Test Harness ---
if __name__ == '__main__':
    print("--- Running agent_reasoner.py Standalone Test ---")

    # Load market data scenarios
    try:
        market_scenarios = pd.read_csv('data/synthetic_market_data.csv').to_dict(orient='records')
        print("\nLoaded market scenarios from CSV successfully.")
    except Exception as e:
        print(f"\nError loading market scenarios: {e}")
        market_scenarios = []

    # Test Case 1: Low Risk, High Wind (Should NOT recommend switch)
    print("\n=== Test Case 1: Low Risk, Normal/High Wind ===")
    test_input_1 = {
        'weather_summary': {'2024-07-01': {'min_temp': 25, 'max_temp': 35, 'avg_wind': 8, 'solar_potential_hours': 8}},
        'renewable_metrics': {'wind_deviation_from_normal_pct': 10.0, 'periods_with_low_wind': 1, 
                            'solar_deviation_from_normal_pct': 5.0, 'average_wind_capacity_factor': 0.4, 
                            'average_solar_capacity_factor_daytime': 0.28},
        'eia_realtime_data': {'current': {'timestamp_iso': '2024-07-01T14:00:00Z', 
                                         'demand_mw': 60000, 'wind_mw': 18000, 'solar_mw': 5000}},
        'load_prediction': {'predicted_peak_load_gw': 70, 'load_vs_normal_pct': 2.0},
        'price_risk': {'price_spike_probability': 0.15, 'expected_max_price': 100},
        'market_prices': next((scenario for scenario in market_scenarios if scenario['scenario'] == 'normal_conditions'), {})
    }
    print("\nInput Data:")
    pprint(test_input_1)
    strategy1 = generate_strategy_via_replicate(test_input_1)
    print(f"\nGenerated Strategy 1:\n{strategy1}")
    print("-" * 80)

    # Test Case 2: High Risk, Low Wind (Should recommend switch)
    print("\n=== Test Case 2: High Risk, Low Wind ===")
    test_input_2 = {
        'weather_summary': {'2024-08-15': {'min_temp': 30, 'max_temp': 40, 'avg_wind': 3, 'solar_potential_hours': 9}},
        'renewable_metrics': {'wind_deviation_from_normal_pct': -55.0, 'periods_with_low_wind': 20, 
                            'solar_deviation_from_normal_pct': -10.0, 'average_wind_capacity_factor': 0.15, 
                            'average_solar_capacity_factor_daytime': 0.22},
        'eia_realtime_data': {'current': {'timestamp_iso': '2024-08-15T14:00:00Z', 
                                         'demand_mw': 78000, 'wind_mw': 4000, 'solar_mw': 6000}},
        'load_prediction': {'predicted_peak_load_gw': 82, 'load_vs_normal_pct': 18.0},
        'price_risk': {'price_spike_probability': 0.80, 'expected_max_price': 1500},
        'market_prices': next((scenario for scenario in market_scenarios if scenario['scenario'] == 'extreme_heat_high_risk'), {})
    }
    print("\nInput Data:")
    pprint(test_input_2)
    strategy2 = generate_strategy_via_replicate(test_input_2)
    print(f"\nGenerated Strategy 2:\n{strategy2}")
    print("-" * 80)

    # Test Case 3: Extreme Cold (Test weather derivatives)
    print("\n=== Test Case 3: Extreme Cold ===")
    test_input_3 = {
        'weather_summary': {'2024-01-15': {'min_temp': -5, 'max_temp': 5, 'avg_wind': 6, 'solar_potential_hours': 4}},
        'renewable_metrics': {'wind_deviation_from_normal_pct': -20.0, 'periods_with_low_wind': 8, 
                            'solar_deviation_from_normal_pct': -30.0, 'average_wind_capacity_factor': 0.25, 
                            'average_solar_capacity_factor_daytime': 0.15},
        'eia_realtime_data': {'current': {'timestamp_iso': '2024-01-15T14:00:00Z', 
                                         'demand_mw': 65000, 'wind_mw': 8000, 'solar_mw': 2000}},
        'load_prediction': {'predicted_peak_load_gw': 75, 'load_vs_normal_pct': 12.0},
        'price_risk': {'price_spike_probability': 0.65, 'expected_max_price': 800},
        'market_prices': next((scenario for scenario in market_scenarios if scenario['scenario'] == 'extreme_cold_high_risk'), {})
    }
    print("\nInput Data:")
    pprint(test_input_3)
    strategy3 = generate_strategy_via_replicate(test_input_3)
    print(f"\nGenerated Strategy 3:\n{strategy3}")
    print("-" * 80)
