import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

def get_ercot_token():
    load_dotenv()
    
    username = os.getenv('ERCOT_USERNAME')
    password = os.getenv('ERCOT_PASSWORD')
    subscription_key = os.getenv('ERCOT_API_KEY')
    
    if not all([username, password, subscription_key]):
        print("\nMissing required credentials in .env file")
        print("Please ensure ERCOT_USERNAME, ERCOT_PASSWORD, and ERCOT_API_KEY are set")
        return None, None
    
    auth_url = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    auth_url += f"?username={username}"
    auth_url += f"&password={password}"
    auth_url += "&grant_type=password"
    auth_url += "&scope=openid+fec253ea-0d06-4272-a5e6-b478baeecd70+offline_access"
    auth_url += "&client_id=fec253ea-0d06-4272-a5e6-b478baeecd70"
    auth_url += "&response_type=id_token"
    
    # Headers
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    try:
        print("\nRequesting authentication token...")
        response = requests.post(auth_url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            return None, None
            
        token_data = response.json()
        access_token = token_data.get('access_token')
        id_token = token_data.get('id_token')
        
        if not access_token or not id_token:
            print("Failed to obtain tokens from response")
            return None, None
            
        return access_token, id_token
        
    except requests.exceptions.RequestException as e:
        print(f"\nError obtaining token: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response text: {e.response.text}")
        return None, None

def get_weather_forecast(target_date):
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("OpenWeather API Key not found")
        return None
        
    # Austin, TX coordinates
    lat = 30.2672
    lon = -97.7431
    
    # OpenWeather API endpoint
    url = f"https://api.openweathermap.org/data/2.5/forecast"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'  # Get temperature in Celsius
    }
    
    try:
        print("\nFetching weather forecast...")
        response = requests.get(url, params=params)
        print(f"Weather API Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error getting weather forecast: {response.text}")
            return None
            
        weather_data = response.json()
        print(f"Retrieved {len(weather_data.get('list', []))} weather records")
        return weather_data
        
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return None

def get_load_and_weather_data():
    # Get authentication token
    access_token, id_token = get_ercot_token()
    if not access_token or not id_token:
        return None
    
    # Get API key
    api_key = os.getenv('ERCOT_API_KEY')
    if not api_key:
        print("API Key not found")
        return None
    
    # Set up the request parameters for ERCOT data (2024)
    ercot_date = datetime(2024, 4, 13, 10, 0)  # 2024 date for ERCOT
    weather_date = datetime.now()  # Get current date for weather forecast
    
    to_time = ercot_date.strftime("%Y-%m-%dT%H:00:00")
    from_time = (ercot_date - timedelta(hours=24)).strftime("%Y-%m-%dT%H:00:00")
    
    print(f"\nRequesting ERCOT data for time range (2024):")
    print(f"From: {from_time}")
    print(f"To: {to_time}")
    
    # Get ERCOT load data
    url = "https://api.ercot.com/api/public-reports/np3-910-er/2d_agg_load_summary"
    params = {
        'SCEDTimestampFrom': from_time,
        'SCEDTimestampTo': to_time,
        'size': 100
    }
    
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    
    try:
        # Get ERCOT data
        print("\nFetching ERCOT data...")
        ercot_response = requests.get(url, headers=headers, params=params)
        print(f"ERCOT API Status Code: {ercot_response.status_code}")
        
        if ercot_response.status_code != 200:
            print(f"Error getting ERCOT data: {ercot_response.text}")
            return None
            
        ercot_data = ercot_response.json()
        print(f"Retrieved {len(ercot_data.get('data', []))} ERCOT records")
        
        # Get current weather forecast
        print(f"\nGetting current weather forecast to use as pattern for 2025...")
        weather_data = get_weather_forecast(weather_date)
        if not weather_data:
            return None
            
        # Process and combine the data
        combined_data = []
        
        if ercot_data.get('data'):
            records = ercot_data['data']
            print("\nProcessing and combining data...")
            
            # Create a mapping of hour -> temperature from current forecast
            hourly_temps = {}
            for forecast in weather_data['list']:
                forecast_dt = datetime.fromtimestamp(forecast['dt'])
                hourly_temps[forecast_dt.hour] = forecast['main']['temp']
            
            for record in records:
                # Parse ERCOT timestamp from 2024
                ercot_dt = datetime.strptime(record[0], "%Y-%m-%dT%H:%M:%S")
                # Get temperature for this hour from our mapping
                temp = hourly_temps.get(ercot_dt.hour)
                
                if temp is not None:
                    # Create 2025 version of the timestamp
                    dt_2025 = ercot_dt.replace(year=2025)
                    combined_data.append({
                        'datetime_2024': ercot_dt,
                        'datetime_2025': dt_2025,
                        'hour': ercot_dt.hour,
                        'month': ercot_dt.month,
                        'temperature': temp,
                        'load': float(record[4]),  # aggLoadSummary
                        'generation': float(record[2])  # sumTelemGenMW
                    })
        
        print(f"\nSuccessfully combined {len(combined_data)} records")
        
        # Print sample of combined data
        if combined_data:
            print("\nCombined Data Sample:")
            for i, data in enumerate(combined_data[:5]):
                print(f"\nRecord {i+1}:")
                print(f"ERCOT Datetime (2024): {data['datetime_2024']}")
                print(f"Weather Datetime (2025): {data['datetime_2025']}")
                print(f"Hour: {data['hour']}")
                print(f"Month: {data['month']}")
                print(f"Temperature: {data['temperature']}°C")
                print(f"Load (2024): {data['load']:.2f} MW")
                print(f"Generation (2024): {data['generation']:.2f} MW")
        
        return combined_data
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Fetching ERCOT Load and Weather Data")
    print("-" * 50)
    
    combined_data = get_load_and_weather_data()
    if combined_data:
        print(f"\nSuccessfully processed {len(combined_data)} records")
    else:
        print("\nFailed to get combined data") 