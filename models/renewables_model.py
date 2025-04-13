import statistics
from datetime import datetime

# --- Configuration / Heuristics ---
NORMAL_WIND_CF = 0.35  # Typical average capacity factor for ERCOT wind
NORMAL_SOLAR_CF = 0.25 # Typical average capacity factor for ERCOT solar (daytime average)

WIND_CUT_IN_SPEED = 3.0   # m/s
WIND_RATED_SPEED = 12.0  # m/s
WIND_CUT_OUT_SPEED = 25.0 # m/s
MAX_WIND_CF = 0.9

SOLAR_PEAK_CF = 0.85
SOLAR_CLOUDY_CF = 0.4
SOLAR_STORM_CF = 0.1
DAY_START_HOUR = 6 # Simplified UTC hour check
DAY_END_HOUR = 20  # Simplified UTC hour check
# --- End Configuration ---

def estimate_wind_capacity_factor(wind_speed: float) -> float:
    """Estimates wind turbine capacity factor based on wind speed."""
    if wind_speed < WIND_CUT_IN_SPEED or wind_speed >= WIND_CUT_OUT_SPEED:
        return 0.0
    elif wind_speed < WIND_RATED_SPEED:
        return MAX_WIND_CF * (wind_speed - WIND_CUT_IN_SPEED) / (WIND_RATED_SPEED - WIND_CUT_IN_SPEED)
    else:
        return MAX_WIND_CF

def estimate_solar_capacity_factor(weather_main: str, timestamp_str: str) -> float:
    """Estimates solar panel capacity factor based on weather and time (UTC hour)."""
    try:
        dt_obj = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        hour = dt_obj.hour # Using UTC hour
        
        if not (DAY_START_HOUR <= hour < DAY_END_HOUR):
            return 0.0 # Night time

        if weather_main == "Clear":
            return SOLAR_PEAK_CF
        elif weather_main == "Clouds":
            return SOLAR_CLOUDY_CF
        elif weather_main in ["Rain", "Drizzle", "Snow", "Thunderstorm", "Fog", "Mist"]:
             return SOLAR_STORM_CF
        else: 
             return SOLAR_CLOUDY_CF * 0.75 
    except ValueError:
        print(f"Warning: Could not parse timestamp {timestamp_str} for solar estimation.")
        return 0.0

def calculate_renewable_metrics(forecasts: list) -> dict:
    """
    Calculates renewable energy generation metrics from a list of forecast data points.
    """
    wind_cfs = []
    solar_cfs = []
    detailed_estimates = []
    low_wind_periods = 0
    low_solar_periods_daytime = 0

    for entry in forecasts:
        wind_speed = entry.get('wind_speed', 0.0)
        weather_main = entry.get('weather_main', 'Unknown')
        timestamp = entry.get('timestamp', None)

        wind_cf = estimate_wind_capacity_factor(wind_speed)
        solar_cf = 0.0
        is_daytime = False
        if timestamp:
             solar_cf = estimate_solar_capacity_factor(weather_main, timestamp)
             dt_obj = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
             hour = dt_obj.hour
             is_daytime = DAY_START_HOUR <= hour < DAY_END_HOUR

        wind_cfs.append(wind_cf)
        solar_cfs.append(solar_cf)

        detailed_estimates.append({
             "timestamp": timestamp,
             "wind_cf_estimated": round(wind_cf, 3),
             "solar_cf_estimated": round(solar_cf, 3)
        })

        if wind_cf < (NORMAL_WIND_CF * 0.3): 
            low_wind_periods += 1
        if is_daytime and solar_cf < (NORMAL_SOLAR_CF * 0.3): 
             low_solar_periods_daytime += 1

    avg_wind_cf = statistics.mean(wind_cfs) if wind_cfs else 0.0
    daytime_solar_cfs = [cf for cf, entry in zip(solar_cfs, forecasts) 
                         if entry.get('timestamp') and DAY_START_HOUR <= datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S').hour < DAY_END_HOUR]
    avg_solar_cf_daytime = statistics.mean(daytime_solar_cfs) if daytime_solar_cfs else 0.0
    
    wind_deviation_pct = ((avg_wind_cf - NORMAL_WIND_CF) / NORMAL_WIND_CF * 100) if NORMAL_WIND_CF else 0
    solar_deviation_pct = ((avg_solar_cf_daytime - NORMAL_SOLAR_CF) / NORMAL_SOLAR_CF * 100) if NORMAL_SOLAR_CF else 0

    return {
        "average_wind_capacity_factor": round(avg_wind_cf, 3),
        "average_solar_capacity_factor_daytime": round(avg_solar_cf_daytime, 3),
        "wind_deviation_from_normal_pct": round(wind_deviation_pct, 1),
        "solar_deviation_from_normal_pct": round(solar_deviation_pct, 1),
        "periods_with_low_wind": low_wind_periods,
        "periods_with_low_solar_daytime": low_solar_periods_daytime,
        "detailed_estimates": detailed_estimates 
    }

# Optional: Add the __main__ block for testing within this file
if __name__ == '__main__':
    dummy_forecasts = [
        {'timestamp': '2024-01-01 12:00:00', 'wind_speed': 10.0, 'weather_main': 'Clear'},
        {'timestamp': '2024-01-01 15:00:00', 'wind_speed': 15.0, 'weather_main': 'Clouds'},
        {'timestamp': '2024-01-01 18:00:00', 'wind_speed': 5.0, 'weather_main': 'Rain'},
        {'timestamp': '2024-01-01 21:00:00', 'wind_speed': 2.0, 'weather_main': 'Clear'}, # Night
    ]
    print("Testing renewable metrics calculation:")
    metrics = calculate_renewable_metrics(dummy_forecasts)
    from pprint import pprint
    pprint(metrics)
