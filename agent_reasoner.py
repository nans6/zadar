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
from datetime import datetime, timedelta
from api_manager import APIManager, APIError
import time

logging.basicConfig(
    filename='output.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    logging.critical("REPLICATE_API_TOKEN not found in environment variables. LLM calls will fail.")

logger = logging.getLogger(__name__)

class AgentReasoner:
    def __init__(self, api_manager=None):
        self.api_manager = api_manager or APIManager()
        self.data_needs = set()
        self.cached_data = {}
        self.last_strategy_time = None
        self.error_counts = {}
        
    def assess_data_needs(self):
        """Agent determines what data it needs based on current state"""
        self.data_needs.clear()
        current_time = datetime.now()
        
        # Always need basic weather data for strategy
        self.data_needs.add(('weather', 'temperature'))
        self.data_needs.add(('weather', 'wind'))
        
        # Check if we need solar potential (only during daylight hours)
        hour = current_time.hour
        if 6 <= hour <= 18:  # Daylight hours
            self.data_needs.add(('weather', 'solar_potential'))
            
        # Always need current grid demand
        self.data_needs.add(('grid', 'demand'))
        
        # Need ERCOT data for detailed analysis
        if self._needs_detailed_forecast():
            self.data_needs.add(('ercot', 'forecasts'))
            self.data_needs.add(('ercot', 'prices'))
            
        logger.info(f"Assessed data needs: {self.data_needs}")
        
    def _needs_detailed_forecast(self):
        """Determine if we need detailed ERCOT forecast data"""
        # Get detailed forecast if:
        # 1. We haven't gotten it recently
        # 2. Current conditions suggest high volatility
        if not self.last_strategy_time or \
           datetime.now() - self.last_strategy_time > timedelta(hours=1):
            return True
            
        try:
            current_temp = self.cached_data.get(('weather', 'temperature'))
            if current_temp and current_temp > 35:  # High temperature
                return True
        except Exception:
            pass
            
        return False
        
    def fetch_required_data(self):
        """Agent autonomously fetches data it determines it needs"""
        for api, data_type in self.data_needs:
            if self.api_manager.should_refresh(api):
                try:
                    data = self.api_manager.fetch_data(api, data_type)
                    self.cached_data[(api, data_type)] = data
                    self.error_counts[(api, data_type)] = 0  # Reset error count
                    logger.info(f"Successfully fetched {data_type} from {api}")
                except APIError as e:
                    self._handle_api_failure(api, data_type)
                    
    def _handle_api_failure(self, api, data_type):
        """Handle API failures with increasing backoff and alternatives"""
        key = (api, data_type)
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        logger.warning(f"API failure for {api} {data_type}. Attempt {self.error_counts[key]}")
        
        if self.error_counts[key] <= 3:
            # Use cached data if available
            if key in self.cached_data:
                logger.info(f"Using cached data for {api} {data_type}")
                return
                
        # Try alternative data source
        alt_data = self._get_alternative_data(api, data_type)
        if alt_data is not None:
            self.cached_data[key] = alt_data
            logger.info(f"Using alternative data for {api} {data_type}")
            
    def _get_alternative_data(self, api, data_type):
        """Get alternative data when primary source fails"""
        if api == 'weather':
            # Use backup weather API or historical averages
            return self._get_backup_weather_data(data_type)
        elif api == 'grid':
            # Use ERCOT data as backup for grid data
            try:
                return self.api_manager.fetch_data('ercot', 'load')
            except APIError:
                return None
        return None
        
    def _get_backup_weather_data(self, data_type):
        """Get backup weather data from alternative source"""
        # Implement backup weather data source here
        # For now, return None to indicate no backup available
        return None
        
    def generate_strategy(self):
        """Generate trading strategy based on available data"""
        self.assess_data_needs()
        self.fetch_required_data()
        
        try:
            # Get weather data from the forecast
            weather_data = self.cached_data.get(('weather', 'forecast'))
            if not weather_data or 'daily_summaries' not in weather_data:
                logger.error("Missing or invalid weather forecast data")
                return None
            
            # Get current date key and daily summary
            current_date = datetime.now().strftime('%Y-%m-%d')
            daily_summary = weather_data['daily_summaries'].get(current_date, {})
            
            if not daily_summary:
                logger.error(f"No weather summary found for {current_date}")
                return None
            
            # Extract weather metrics
            current_temp_c = daily_summary.get('avg_temp')
            current_temp_k = current_temp_c + 273.15 if current_temp_c is not None else 298.15  # Default to 25°C
            solar_potential_hours = daily_summary.get('solar_potential_hours', 0)
            wind_speed = daily_summary.get('avg_wind', 0)
            
            # Get load predictions
            load_prediction = get_load_predictions(current_temp_k)
            if not load_prediction:
                logger.error("Failed to get load predictions")
                return None
            
            # Calculate renewable metrics from detailed forecast
            renewable_metrics = calculate_renewable_metrics(weather_data.get('detailed_forecast', []))
            
            # Assess price risk based on weather and load
            risk_factors = {
                'temperature_risk': abs(current_temp_c - 25) / 15,  # Normalized deviation from 25°C
                'wind_risk': 1.0 - (wind_speed / 10.0) if wind_speed < 10 else 0,  # Risk increases with low wind
                'solar_risk': 1.0 - (solar_potential_hours / 12.0),  # Risk increases with low solar potential
                'load_risk': (load_prediction['predicted_load_mw'] / 65000) if load_prediction['predicted_load_mw'] > 65000 else 0
            }
            
            overall_risk = sum(risk_factors.values()) / len(risk_factors)
            
            # Prepare input data for LLM
            llm_input_data = {
                'weather_summary': {
                    'temperature_c': current_temp_c,
                    'wind_speed': wind_speed,
                    'solar_potential_hours': solar_potential_hours,
                    'storm_hours': daily_summary.get('storm_hours', 0),
                    'extreme_heat_hours': daily_summary.get('extreme_heat_hours', 0)
                },
                'load_prediction': {
                    'current_load_mw': load_prediction['current_load_mw'],
                    'predicted_load_mw': load_prediction['predicted_load_mw'],
                    'predicted_load_gw': load_prediction['predicted_load_gw']
                },
                'renewable_metrics': renewable_metrics,
                'risk_assessment': {
                    'risk_factors': risk_factors,
                    'overall_risk': overall_risk
                }
            }
            
            # Generate strategy using LLM
            strategy = self.generate_strategy_via_replicate(llm_input_data)
            if strategy:
                self.last_strategy_time = datetime.now()
                logger.info(f"Successfully generated strategy with risk level {overall_risk:.2f}")
                return strategy
            else:
                logger.error("Failed to generate strategy via LLM")
                return None
            
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return None
            
    def calculate_next_cycle_time(self):
        """Calculate when to run next cycle based on data needs"""
        next_refresh = float('inf')
        for api, _ in self.data_needs:
            if api in self.api_manager.last_fetch:
                refresh_rate = self.api_manager.available_apis[api]['refresh_rate']
                time_since_last = time.time() - self.api_manager.last_fetch[api]
                next_refresh = min(next_refresh, refresh_rate - time_since_last)
                
        return max(next_refresh, 60)  # At least 60 seconds between cycles

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
    Generates hedging strategy recommendations using DeepSeek on Replicate.

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

    logging.info("Formatting prompt for DeepSeek LLM...")

    # Format input data for prompt
    weather_str = format_weather_summary(llm_input_data.get('weather_summary', {}))
    renew_str = format_renewable_metrics(llm_input_data.get('renewable_metrics', {}))
    eia_str = format_eia_data(llm_input_data.get('eia_realtime_data', {}).get('current', {}))
    load_str = format_load_prediction(llm_input_data.get('load_prediction', {}))
    risk_str = format_price_risk(llm_input_data.get('price_risk', {}))
    market_str = format_market_data(llm_input_data.get('market_prices', {}))

    prompt = f"""You are an expert energy trader and risk analyst for ERCOT. Based on the following data, provide a concise trading strategy:

Weather: {weather_str}
Renewables: {renew_str}
Grid State: {eia_str}
Load: {load_str}
Risk: {risk_str}
{market_str}

Required Format:
1. HEDGE ACTIONS: List 2-3 specific trades with exact quantities and prices
2. RENEWABLE SWITCH: Yes/No (required if wind deviation < -25% AND price risk > 60%)
3. JUSTIFICATION: Brief explanation referencing specific market conditions
4. LOAD CONDITIONS: HIGH/MODERATE/LOW based on system load"""

    try:
        output = replicate.run(
            "deepseek-ai/deepseek-r1",
            input={
                "prompt": prompt,
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 4000,
                "system_prompt": "You are an expert energy trader. Keep responses concise and focused on specific trading actions."
            }
        )
        
        # Handle both streaming and non-streaming responses
        if hasattr(output, '__iter__'):
            response = "".join(str(chunk) for chunk in output)
        else:
            response = str(output)
            
        logging.info(f"Strategy generated successfully: {response[:100]}...")
        return response
        
    except Exception as e:
        error_msg = f"Error generating strategy: {str(e)}"
        logging.error(error_msg)
        return error_msg

def format_weather_summary(weather_data: Dict) -> str:
    if not weather_data:
        return "Not available"
    days = list(weather_data.keys())[:3]
    summaries = []
    for day in days:
        summary = weather_data.get(day, {})
        summaries.append(
            f"{day}: {summary.get('min_temp','N/A')}-{summary.get('max_temp','N/A')}°C, "
            f"Wind {summary.get('avg_wind','N/A')}m/s"
        )
    return "; ".join(summaries)

def format_renewable_metrics(metrics: Dict) -> str:
    if not metrics:
        return "Not available"
    return (f"Wind CF {metrics.get('average_wind_capacity_factor','N/A')}, "
            f"Dev {metrics.get('wind_deviation_from_normal_pct','N/A')}%, "
            f"Solar CF {metrics.get('average_solar_capacity_factor_daytime','N/A')}")

def format_eia_data(data: Dict) -> str:
    if not data:
        return "Not available"
    return (f"Demand {data.get('demand_mw','N/A')}MW, "
            f"Wind {data.get('wind_mw','N/A')}MW, "
            f"Solar {data.get('solar_mw','N/A')}MW")

def format_load_prediction(pred: Dict) -> str:
    if not pred:
        return "Not available"
    return (f"Current {pred.get('current_load_mw','N/A')}MW, "
            f"Predicted {pred.get('predicted_load_mw','N/A')}MW")

def format_price_risk(risk: Dict) -> str:
    if not risk:
        return "Not available"
    return (f"Spike prob {risk.get('price_spike_probability','N/A')}, "
            f"Level {risk.get('risk_level','N/A')}")

def format_market_data(data: Dict) -> str:
    if not data:
        return ""
    return f"\nMarket Prices: North Peak ${data.get('north_peak','N/A')}, West Peak ${data.get('west_peak','N/A')}"

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
