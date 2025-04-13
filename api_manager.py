import time
import os
import requests
from datetime import datetime, timedelta
import logging
from ratelimit import limits, sleep_and_retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIError(Exception):
    pass

class RateLimitError(APIError):
    pass

class APIManager:
    def __init__(self):
        self.available_apis = {
            'weather': {
                'endpoint': 'openweather',
                'data_types': ['temperature', 'wind', 'solar_potential'],
                'refresh_rate': 3600,  # seconds
                'rate_limit': 60  # calls per minute
            },
            'grid': {
                'endpoint': 'eia',
                'data_types': ['demand', 'generation', 'prices'],
                'refresh_rate': 300,
                'rate_limit': 100
            },
            'ercot': {
                'endpoint': 'ercot',
                'data_types': ['load', 'forecasts', 'prices'],
                'refresh_rate': 900,
                'rate_limit': 30
            }
        }
        self.last_fetch = {}
        self.cached_data = {}
        self.api_calls = {api: [] for api in self.available_apis}
        
    def get_available_data_types(self):
        """Let agent discover what data it can request"""
        return {api: info['data_types'] for api, info in self.available_apis.items()}
        
    def should_refresh(self, api_name):
        """Agent can check if data needs refreshing"""
        if api_name not in self.last_fetch:
            return True
        elapsed = time.time() - self.last_fetch[api_name]
        return elapsed > self.available_apis[api_name]['refresh_rate']
    
    def _check_rate_limit(self, api_name):
        """Check if we're within rate limits"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old calls
        self.api_calls[api_name] = [t for t in self.api_calls[api_name] if t > minute_ago]
        
        # Check if we're at limit
        if len(self.api_calls[api_name]) >= self.available_apis[api_name]['rate_limit']:
            raise RateLimitError(f"Rate limit exceeded for {api_name}")
            
        # Add new call
        self.api_calls[api_name].append(now)
        
    @sleep_and_retry
    @limits(calls=60, period=60)
    def _fetch_weather_data(self, data_type):
        """Fetch weather data with rate limiting"""
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            raise APIError("OpenWeather API key not found")
            
        self._check_rate_limit('weather')
        
        # Example coordinates for Texas
        lat, lon = 31.9686, -99.9018
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.last_fetch['weather'] = time.time()
            self.cached_data[('weather', data_type)] = data
            
            if data_type == 'temperature':
                return data['main']['temp'] - 273.15  # Convert to Celsius
            elif data_type == 'wind':
                return data['wind']['speed']
            elif data_type == 'solar_potential':
                return data['clouds']['all']  # Use cloud coverage as proxy
                
        except requests.exceptions.Timeout:
            logger.error("Weather API timeout")
            return self._get_cached_or_default('weather', data_type)
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API error: {str(e)}")
            return self._get_cached_or_default('weather', data_type)
            
    @sleep_and_retry
    @limits(calls=100, period=60)
    def _fetch_grid_data(self, data_type):
        """Fetch grid data with rate limiting"""
        api_key = os.getenv('EIA_API_KEY')
        if not api_key:
            raise APIError("EIA API key not found")
            
        self._check_rate_limit('grid')
        
        url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/?api_key={api_key}&frequency=hourly&data[0]=value&facets[respondent][]=ERCO"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.last_fetch['grid'] = time.time()
            self.cached_data[('grid', data_type)] = data
            
            if data_type == 'demand':
                return data['response']['data'][0]['value']
            # Add other data type processing as needed
                
        except requests.exceptions.Timeout:
            logger.error("Grid API timeout")
            return self._get_cached_or_default('grid', data_type)
        except requests.exceptions.RequestException as e:
            logger.error(f"Grid API error: {str(e)}")
            return self._get_cached_or_default('grid', data_type)
            
    @sleep_and_retry
    @limits(calls=30, period=60)
    def _fetch_ercot_data(self, data_type):
        """Fetch ERCOT data with rate limiting"""
        username = os.getenv('ERCOT_USERNAME')
        password = os.getenv('ERCOT_PASSWORD')
        if not username or not password:
            raise APIError("ERCOT credentials not found")
            
        self._check_rate_limit('ercot')
        
        # Implement actual ERCOT API call here
        try:
            # Placeholder for actual ERCOT API implementation
            self.last_fetch['ercot'] = time.time()
            return {"message": "ERCOT API call placeholder"}
        except Exception as e:
            logger.error(f"ERCOT API error: {str(e)}")
            return self._get_cached_or_default('ercot', data_type)
            
    def _get_cached_or_default(self, api_name, data_type):
        """Get cached data or return sensible default"""
        if (api_name, data_type) in self.cached_data:
            logger.info(f"Using cached data for {api_name} {data_type}")
            return self.cached_data[(api_name, data_type)]
            
        # Return sensible defaults
        defaults = {
            'weather': {
                'temperature': 25.0,
                'wind': 5.0,
                'solar_potential': 50
            },
            'grid': {
                'demand': 45000.0
            },
            'ercot': {
                'load': 45000.0,
                'prices': {'price': 50.0}
            }
        }
        
        return defaults.get(api_name, {}).get(data_type)
        
    def fetch_data(self, api_name, data_type):
        """Fetch data with error handling and rate limiting"""
        if api_name not in self.available_apis:
            raise APIError(f"Unknown API: {api_name}")
        if data_type not in self.available_apis[api_name]['data_types']:
            raise APIError(f"Unknown data type {data_type} for API {api_name}")
            
        try:
            if api_name == 'weather':
                return self._fetch_weather_data(data_type)
            elif api_name == 'grid':
                return self._fetch_grid_data(data_type)
            elif api_name == 'ercot':
                return self._fetch_ercot_data(data_type)
        except RateLimitError:
            logger.warning(f"Rate limit hit for {api_name}, using cached/default data")
            return self._get_cached_or_default(api_name, data_type)
        except Exception as e:
            logger.error(f"Error fetching {data_type} from {api_name}: {str(e)}")
            return self._get_cached_or_default(api_name, data_type) 