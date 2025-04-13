import os
import requests
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
from statistics import mean
from datetime import datetime, timedelta, timezone
import pandas as pd

# Load environment variables
load_dotenv()

# Weather condition constants
STORM_RISK_CONDITIONS = ["Thunderstorm", "Rain", "Drizzle", "Snow"]
HIGH_HUMIDITY_THRESHOLD = 85
EXTREME_HEAT_THRESHOLD = 38.0  # Celsius (~100°F)
SOLAR_GENERATION_HOURS = range(6, 20)  # 6 AM to 8 PM

def get_eia_realtime_grid_data(region_id: str = "TEX", hours_back: int = 24) -> Optional[Dict[str, Any]]:
    """
    Fetches hourly grid data (Demand, Wind Gen, Solar Gen) for specified hours.
    
    Args:
        region_id: The Balancing Authority ID (e.g., 'TEX' for ERCOT).
        hours_back: Number of hours of historical data to fetch.

    Returns:
        Dictionary containing current and historical grid data with timestamps.
    """
    API_KEY = os.getenv("EIA_API_KEY")
    if not API_KEY:
        print("Error: EIA_API_KEY not found in environment variables.")
        return None

    BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data"
    
    # Use current time and get historical data
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=hours_back)

    params = {
        "api_key": API_KEY,
        "frequency": "hourly",
        "data[]": ["value"],
        "facets[respondent][]": [region_id],
        "facets[type][]": ["D", "NG.WND", "NG.SUN"],
        "start": start_dt.strftime("%Y-%m-%dT%H"),
        "end": end_dt.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": hours_back * 3,  # Account for 3 metrics per hour
        "offset": 0
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("response", {}).get("total", 0) == 0:
            print("Warning: EIA API returned no data for the requested period/series.")
            return None

        df = pd.DataFrame(data['response']['data'])
        
        # Convert value column to numeric before pivoting
        df['value'] = pd.to_numeric(df['value'])
        
        # Create a mapping for the types to more readable names
        type_mapping = {
            "D": "demand_mw",
            "NG.WND": "wind_mw",
            "NG.SUN": "solar_mw"
        }
        
        # Pivot the data and rename columns
        pivot_df = df.pivot(index='period', columns='type', values='value')
        pivot_df = pivot_df.rename(columns=type_mapping)
        
        # Sort by most recent first
        pivot_df = pivot_df.sort_index(ascending=False)
        
        # Get latest complete data point
        latest_complete_data = pivot_df.iloc[0]
        latest_timestamp_str = latest_complete_data.name

        # Calculate statistics from historical data
        stats = {
            "max_demand": pivot_df['demand_mw'].max(),
            "min_demand": pivot_df['demand_mw'].min(),
            "avg_demand": round(pivot_df['demand_mw'].mean(), 2),
            "max_wind": pivot_df['wind_mw'].max() if 'wind_mw' in pivot_df else None,
            "max_solar": pivot_df['solar_mw'].max() if 'solar_mw' in pivot_df else None,
        }

        # Prepare the return dictionary with both current and historical data
        result = {
            "current": {
                "timestamp_iso": latest_timestamp_str,
                "demand_mw": float(latest_complete_data.get("demand_mw", 0)),
                "wind_mw": float(latest_complete_data.get("wind_mw", 0)) if not pd.isna(latest_complete_data.get("wind_mw")) else 0,
                "solar_mw": float(latest_complete_data.get("solar_mw", 0)) if not pd.isna(latest_complete_data.get("solar_mw")) else 0,
                "region": region_id,
            },
            "statistics": stats,
            "historical_data": pivot_df.to_dict(orient='index'),
            "data_source": "EIA Hourly Grid Monitor API v2"
        }

        return result

    except requests.exceptions.RequestException as e:
        print(f"Error fetching EIA data: {str(e)}")
        if 'response' in locals() and response.content:
            print(f"EIA API Response Content: {response.content}")
        return None
    except (KeyError, IndexError, ValueError, TypeError) as e:
        print(f"Error processing EIA response data: {str(e)}")
        if 'data' in locals():
            print(f"Raw EIA Data causing error: {data}")
        return None

def get_weather_forecast() -> Dict[str, Any]:
    """
    Gets detailed 5-day weather forecast for Austin with enhanced analysis.
    Returns: Dictionary containing:
        - Detailed forecast data in 3-hour intervals
        - Enhanced statistics and risk metrics
        - Solar generation potential indicators
        - Hourly weather patterns
    """
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"
    
    params = {
        "lat": 30.2672,
        "lon": -97.7431,
        "units": "metric",
        "appid": API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        forecast_data = response.json()
        
        # Enhanced forecast analysis
        simplified_forecast = []
        daily_stats = {}
        current_date = None
        
        for item in forecast_data["list"]:
            # Parse timestamp and extract hour
            timestamp = datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S")
            date_key = timestamp.strftime("%Y-%m-%d")
            hour = timestamp.hour
            
            # Extract weather metrics
            temp = item["main"]["temp"]
            wind_speed = item["wind"]["speed"]
            humidity = item["main"]["humidity"]
            weather_main = item["weather"][0]["main"]
            clouds = item["clouds"]["all"]
            
            # Calculate weather risks
            storm_risk = weather_main in STORM_RISK_CONDITIONS
            high_humidity = humidity >= HIGH_HUMIDITY_THRESHOLD
            extreme_heat = temp > EXTREME_HEAT_THRESHOLD
            
            # Calculate solar generation potential
            solar_potential = "High" if (
                hour in SOLAR_GENERATION_HOURS and 
                clouds < 50 and 
                not storm_risk
            ) else "Low"
            
            # Initialize daily stats if new date
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    "temperatures": [],
                    "wind_speeds": [],
                    "humidity_values": [],
                    "storm_hours": 0,
                    "high_humidity_hours": 0,
                    "extreme_heat_hours": 0,
                    "solar_potential_hours": 0
                }
            
            # Update daily statistics
            stats = daily_stats[date_key]
            stats["temperatures"].append(temp)
            stats["wind_speeds"].append(wind_speed)
            stats["humidity_values"].append(humidity)
            if storm_risk: stats["storm_hours"] += 1
            if high_humidity: stats["high_humidity_hours"] += 1
            if extreme_heat: stats["extreme_heat_hours"] += 1
            if solar_potential == "High": stats["solar_potential_hours"] += 1
            
            # Add to detailed forecast
            simplified_forecast.append({
                "timestamp": item["dt_txt"],
                "temperature": temp,
                "feels_like": item["main"]["feels_like"],
                "humidity": humidity,
                "wind_speed": wind_speed,
                "description": item["weather"][0]["description"],
                "weather_main": weather_main,
                "clouds": clouds,
                "storm_risk": storm_risk,
                "high_humidity": high_humidity,
                "extreme_heat": extreme_heat,
                "solar_potential": solar_potential
            })
        
        # Calculate overall statistics and daily summaries
        daily_summaries = {}
        for date, stats in daily_stats.items():
            daily_summaries[date] = {
                "max_temp": max(stats["temperatures"]),
                "min_temp": min(stats["temperatures"]),
                "avg_temp": round(mean(stats["temperatures"]), 2),
                "avg_wind": round(mean(stats["wind_speeds"]), 2),
                "avg_humidity": round(mean(stats["humidity_values"]), 2),
                "storm_hours": stats["storm_hours"],
                "high_humidity_hours": stats["high_humidity_hours"],
                "extreme_heat_hours": stats["extreme_heat_hours"],
                "solar_potential_hours": stats["solar_potential_hours"]
            }
        
        return {
            "city": forecast_data["city"]["name"],
            "country": forecast_data["city"]["country"],
            "timezone": forecast_data["city"]["timezone"],
            "detailed_forecast": simplified_forecast,
            "daily_summaries": daily_summaries,
            "data_source": "OpenWeather API 5-day Forecast"
        }
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch weather data: {str(e)}"}

if __name__ == "__main__":
    # Test weather forecast
    print("\nTesting weather forecast...")
    weather_data = get_weather_forecast()
    if "error" not in weather_data:
        print(f"\nForecast for {weather_data['city']}, {weather_data['country']}")
        for date, summary in weather_data["daily_summaries"].items():
            print(f"\n{date}:")
            print(f"  Temperature: {summary['min_temp']}°C to {summary['max_temp']}°C")
            print(f"  Wind: {summary['avg_wind']} m/s")
            print(f"  Storm Hours: {summary['storm_hours']}")
            print(f"  Solar Potential Hours: {summary['solar_potential_hours']}")
    else:
        print(f"Error: {weather_data['error']}")

    # Test EIA data fetch
    print("\nTesting EIA data fetch...")
    eia_data = get_eia_realtime_grid_data("TEX")
    if eia_data:
        print("\nCurrent Grid Status:")
        current = eia_data["current"]
        print(f"Timestamp: {current['timestamp_iso']}")
        print(f"Demand: {current['demand_mw']} MW")
        print(f"Wind: {current['wind_mw']} MW")
        print(f"Solar: {current['solar_mw']} MW")
    else:
        print("Error: Could not retrieve EIA data")
