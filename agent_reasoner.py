# agent_reasoner.py
import replicate
import os
from dotenv import load_dotenv
from typing import Dict, Any
from pprint import pprint # For testing output

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

    # --- 2. Construct the Prompt with Specific Renewable Switch Instruction ---
    prompt_text = f"""
System: You are an expert ERCOT energy risk hedging assistant. Analyze the data and provide a CONCISE recommendation. Skip detailed explanations and focus on actionable items.

**ERCOT Situation Analysis:**
*   **Weather Forecast Summary (Next ~3 Days):** {weather_str}
*   **Renewable Generation Forecast Metrics:** {renew_str}
*   **Current Grid State (from EIA):** {eia_str}
*   **Load Forecast:** {load_str}
*   **Price Risk Assessment:** {risk_str}

**Required Output Format:**
1. HEDGE ACTIONS: [List 1-2 specific actions with quantities]
2. RENEWABLE SWITCH: [YES/NO] - Required if wind deviation < -25% AND price risk > 60%
   Current values: Wind Dev = {wind_dev}%, Price Risk = {risk_prob_str}
3. JUSTIFICATION: Explain the rationale for the hedge actions and the decision to switch 
   to renewables. Include a detailed explanation of the risk and reward of each action, and why we chose specific quantities and option positions (strike price, exp date, in the money/out of the money)
4. Also how are u coming to this conclusion ie prices etc am I giving you that energy derivaite market data or what?

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

# --- Test Harness (inspired by test_replicate.py) ---
if __name__ == '__main__':
    print("--- Running agent_reasoner.py Standalone Test ---")

    # Test Case 1: Low Risk, High Wind (Should NOT recommend switch)
    print("\n--- Test Case 1: Low Risk, Normal/High Wind ---")
    test_input_1 = {
        'weather_summary': {'2024-07-01': {'min_temp': 25, 'max_temp': 35, 'avg_wind': 8, 'solar_potential_hours': 8}},
        'renewable_metrics': {'wind_deviation_from_normal_pct': 10.0, 'periods_with_low_wind': 1, 'solar_deviation_from_normal_pct': 5.0, 'average_wind_capacity_factor': 0.4, 'average_solar_capacity_factor_daytime': 0.28},
        'eia_realtime_data': {'current': {'timestamp_iso': '...', 'demand_mw': 60000, 'wind_mw': 18000, 'solar_mw': 5000}},
        'load_prediction': {'predicted_peak_load_gw': 70, 'load_vs_normal_pct': 2.0},
        'price_risk': {'price_spike_probability': 0.15, 'expected_max_price': 100}
    }
    pprint(test_input_1)
    strategy1 = generate_strategy_via_replicate(test_input_1)
    print(f"\nGenerated Strategy 1:\n{strategy1}")
    print("-" * 30)

    # Test Case 2: High Risk, Low Wind (Should recommend switch)
    print("\n--- Test Case 2: High Risk, Low Wind ---")
    test_input_2 = {
        'weather_summary': {'2024-08-15': {'min_temp': 30, 'max_temp': 40, 'avg_wind': 3, 'solar_potential_hours': 9}},
        'renewable_metrics': {'wind_deviation_from_normal_pct': -55.0, 'periods_with_low_wind': 20, 'solar_deviation_from_normal_pct': -10.0, 'average_wind_capacity_factor': 0.15, 'average_solar_capacity_factor_daytime': 0.22},
        'eia_realtime_data': {'current': {'timestamp_iso': '...', 'demand_mw': 78000, 'wind_mw': 4000, 'solar_mw': 6000}},
        'load_prediction': {'predicted_peak_load_gw': 82, 'load_vs_normal_pct': 18.0},
        'price_risk': {'price_spike_probability': 0.80, 'expected_max_price': 1500}
    }
    pprint(test_input_2)
    strategy2 = generate_strategy_via_replicate(test_input_2)
    print(f"\nGenerated Strategy 2:\n{strategy2}")
    print("-" * 30)
