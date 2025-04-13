# agent_reasoner.py
import replicate
import os
from dotenv import load_dotenv
from typing import Dict, Any
from pprint import pprint 
import pandas as pd
from models.load_forecast_model import LoadForecaster  
import logging
import json
from datetime import datetime


logging.basicConfig(
    filename='output.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    logging.critical("REPLICATE_API_TOKEN not found in environment variables. LLM calls will fail.")

def get_load_predictions(temp_k: float, timestamp=None) -> Dict[str, Any]:
    """Get load predictions for a given temperature and timestamp."""
    try:
        forecaster = LoadForecaster()
        prediction = forecaster.predict(temp_k, timestamp)
        if prediction['status'] == 'success':
            result = {
                'predicted_load_mw': prediction['predicted_load_mw'],
                'predicted_load_gw': prediction['predicted_load_mw'] / 1000,  # Convert to GW
                'timestamp': prediction['timestamp'],
                'temperature_c': temp_k - 273.15,
                'current_load_mw': prediction.get('current_load', 45000.0)  # Use default if unavailable
            }
            logging.info(f"Load prediction successful: {json.dumps(result, indent=2)}")
            return result
        else:
            error_msg = f"Warning: Load prediction failed: {prediction.get('error_message', 'Unknown error')}"
            logging.warning(error_msg)
            return None
    except Exception as e:
        error_msg = f"Error getting load predictions: {e}"
        logging.error(error_msg)
        return None

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
        error_msg = "Error: Replicate API Token not configured."
        logging.error(error_msg)
        return error_msg

    logging.info("Formatting prompt for Replicate LLM...")

    # Prep Data Snippets for Prompt 
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

    # Load prediction string with LSTM model output
    load_pred = llm_input_data.get('load_prediction', {})
    load_str = (f"Current Load: {load_pred.get('current_load_mw', 'N/A')} MW, "
                f"Predicted Load: {load_pred.get('predicted_load_mw', 'N/A')} MW "
                f"(Temperature: {load_pred.get('temperature_c', 'N/A')}°C), "
                f"Peak: {load_pred.get('predicted_load_gw','N/A')} GW")

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

    prompt = f"""You are an expert energy trader and risk analyst for ERCOT. Your task is to recommend hedging strategies based on the following data:

**ERCOT Situation Analysis:**
*   **Weather Forecast Summary (Next ~3 Days):** {weather_str}
*   **Renewable Generation Forecast Metrics:** {renew_str}
*   **Current Grid State (from EIA):** {eia_str}
*   **Load Forecast:** {load_str}
*   **ERCOT Load Forecast:**
    - Current System Load: {eia_data.get('demand_mw', 'N/A')} MW
    - Predicted Peak Load: {load_pred.get('predicted_load_mw', 'N/A')} MW
    - Temperature Impact: {load_pred.get('temperature_c', 'N/A')}°C
    - Load Forecast Confidence: High
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

5. ERCOT Load Forecast:
   - System-wide load > 65 GW indicates high stress
   - Load ramp > 10 GW/3hr suggests volatility
   - Morning ramp 6-9am, evening peak 4-7pm
   - Weekend load typically 80% of weekday
   - Reserve margin < 10% signals scarcity risk

**Required Output Format:**
1. HEDGE ACTIONS: [List 1-2 specific actions using exact ticker symbols, quantities, and both premium & strike prices for options]
2. RENEWABLE SWITCH: [YES/NO] - Required if wind deviation < -25% AND price risk > 60%
   Current values: Wind Dev = {wind_dev}%, Price Risk = {risk_prob_str}
3. JUSTIFICATION: Reference specific tickers, spreads, and strikes in your rationale.
   Explain product selection based on market conditions.
4. ERCOT LOAD CONDITIONS: [HIGH/MODERATE/LOW] - Required if system-wide load > 65 GW (HIGH) or < 30 GW (LOW)
   Current values: System Load = {eia_data.get('demand_mw', 'N/A')} MW, Peak Load = {load_pred.get('predicted_load_mw', 'N/A')} MW"""

    try:
        output = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": prompt,
                  "temperature": 0.3,
                  "top_p": 0.9,
                  "max_tokens": 1000,
                  "system_prompt": "You are an expert energy trader and risk analyst specializing in ERCOT market analysis. You MUST format your responses exactly as requested in the prompt's Required Output Format section, using the exact ticker symbols and market data provided. Always specify exact quantities, premiums, and strike prices in your hedge actions. Be precise and quantitative in your recommendations."}
        )
        
        return ''.join(output)
        
    except Exception as e:
        return f"Error calling Replicate API: {str(e)}"

# Test Cases
if __name__ == "__main__":
    logging.info("Starting test cases...")
    
    # Test Case 1: Low Risk with High Wind
    test_input_1 = {
        'weather_summary': {
            '2024-04-13': {'min_temp': 15, 'max_temp': 25, 'avg_wind': 8.5, 'solar_potential_hours': 10},
            '2024-04-14': {'min_temp': 16, 'max_temp': 26, 'avg_wind': 9.0, 'solar_potential_hours': 11},
            '2024-04-15': {'min_temp': 14, 'max_temp': 24, 'avg_wind': 8.0, 'solar_potential_hours': 10}
        },
        'renewable_metrics': {
            'average_wind_capacity_factor': 0.45,
            'wind_deviation_from_normal_pct': 15,
            'average_solar_capacity_factor_daytime': 0.65,
            'solar_deviation_from_normal_pct': 5,
            'periods_with_low_wind': 2
        },
        'eia_realtime_data': {
            'current': {
                'timestamp_iso': '2024-04-13T10:00:00',
                'demand_mw': 45000,
                'wind_mw': 15000,
                'solar_mw': 8000
            }
        },
        'price_risk': {
            'price_spike_probability': 0.15,
            'risk_level': 'LOW',
            'risk_factors': ['High renewable generation', 'Moderate demand']
        }
    }
    
    # Adding load predictions to test case
    test_input_1['load_prediction'] = get_load_predictions(298.15)  # 25°C
    
    logging.info("\nTest Case 1: Low Risk with High Wind")
    logging.info(f"\nInput Data:\n{json.dumps(test_input_1, indent=2)}")
    strategy_1 = generate_strategy_via_replicate(test_input_1)
    logging.info(f"\nStrategy Recommendation:\n{strategy_1}")

    # Test Case 2: High Risk with Low Wind
    test_input_2 = {
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
    
    # Adding load predictions to test case
    test_input_2['load_prediction'] = get_load_predictions(308.15)  # 35°C
    
    logging.info("\nTest Case 2: High Risk with Low Wind")
    logging.info(f"\nInput Data:\n{json.dumps(test_input_2, indent=2)}")
    strategy_2 = generate_strategy_via_replicate(test_input_2)
    logging.info(f"\nStrategy Recommendation:\n{strategy_2}")

    logging.info("Test cases completed.")
