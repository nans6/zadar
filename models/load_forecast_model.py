import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Tuple, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

def inspect_model_state_dict(model_path: str):
    """Helper function to inspect the model's state dictionary"""
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print("\nModel State Dictionary Keys:")
        print("-" * 50)
        for key in state_dict.keys():
            print(f"Key: {key}, Shape: {state_dict[key].shape}")
        return state_dict
    except Exception as e:
        print(f"Error loading model state dict: {str(e)}")
        return None

class LoadForecastLSTM(nn.Module):
    def __init__(self, input_size=38, hidden_size=256, num_layers=1):
        super(LoadForecastLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer 
        self.lstm1 = nn.LSTM(
            input_size=input_size,      # 38 features
            hidden_size=hidden_size,     # 256 hidden units
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)  # 256 -> 128
        self.fc2 = nn.Linear(128, 4)            # 128 -> 4
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm1(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

class LoadForecaster:
    def __init__(self, model_path: str = 'lstm_4h_optimal_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        self.model = self._load_model(model_path)
        self.scaler = None
        
    def _load_model(self, model_path: str) -> LoadForecastLSTM:
        """Load the LSTM model from disk."""
        try:
            print(f"\nLoading model from: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            print("\nState dict loaded successfully")
            
            model = LoadForecastLSTM()
            print("\nModel instance created")
            
            model.load_state_dict(state_dict)
            print("\nState dict loaded into model")
            
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

    def _fetch_ercot_load(self) -> float:
        """Fetch current ERCOT load data."""
        api_key = os.getenv('ERCOT_API_KEY')
        username = os.getenv('ERCOT_USERNAME')
        password = os.getenv('ERCOT_PASSWORD')
        
        if not all([api_key, username, password]):
            print("\nCredentials check:")
            print(f"API Key present: {'Yes' if api_key else 'No'}")
            print(f"Username present: {'Yes' if username else 'No'}")
            print(f"Password present: {'Yes' if password else 'No'}")
            raise ValueError("Missing ERCOT credentials in environment variables")

        auth_url = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
        auth_url += f"?username={username}"
        auth_url += f"&password={password}"
        auth_url += "&grant_type=password"
        auth_url += "&scope=openid+fec253ea-0d06-4272-a5e6-b478baeecd70+offline_access"
        auth_url += "&client_id=fec253ea-0d06-4272-a5e6-b478baeecd70"
        auth_url += "&response_type=id_token"
        
        try:
            print("\nAttempting ERCOT authentication...")
            auth_response = requests.post(auth_url, headers={
                'Ocp-Apim-Subscription-Key': api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            })
            print(f"Auth response status: {auth_response.status_code}")
            if auth_response.status_code != 200:
                print(f"Auth error response: {auth_response.text}")
            auth_response.raise_for_status()
            token_data = auth_response.json()
            access_token = token_data.get('access_token')
            
            if not access_token:
                print(f"Token data received: {token_data.keys()}")
                raise ValueError("Failed to obtain access token")

            print("Successfully obtained access token")

            now = datetime.now()
            now_2024 = now.replace(year=2024)
            to_time = now_2024.strftime("%Y-%m-%dT%H:00:00")
            from_time = (now_2024 - timedelta(hours=24)).strftime("%Y-%m-%dT%H:00:00")

            print(f"\nFetching ERCOT data for time range (2024):")
            print(f"From: {from_time}")
            print(f"To: {to_time}")

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
            
            response = requests.get(url, headers=headers, params=params)
            print(f"ERCOT data response status: {response.status_code}")
            if response.status_code != 200:
                print(f"ERCOT data error response: {response.text}")
            response.raise_for_status()
            data = response.json()
            
            print("\nERCOT Response Structure:")
            print(f"Keys in response: {data.keys()}")
            print(f"Data field type: {type(data.get('data'))}")
            print(f"Number of records: {len(data.get('data', []))}")
            if data.get('data'):
                print(f"First record: {data['data'][0]}")
            
            if not isinstance(data.get('data'), list):
                raise ValueError(f"Expected 'data' to be a list, got {type(data.get('data'))}")
            if not data.get('data'):
                raise ValueError("Empty data array in ERCOT response")
                
            current_load = float(data['data'][0][4])  
            print(f"Successfully retrieved current load: {current_load} MW")
            return current_load
            
        except Exception as e:
            print(f"Warning: Failed to fetch ERCOT load: {e}")
            return 45000.0  # Using 45 GW as a reasonable default for ERCOT load

    def _preprocess_inputs(self, temp_k: float, timestamp: datetime) -> torch.Tensor:
        """Preprocess the input features."""
        hour = timestamp.hour
        month = timestamp.month
        
        try:
            current_load = self._fetch_ercot_load()
        except Exception as e:
            print(f"Warning: Failed to fetch ERCOT load: {e}")
            current_load = 0  # Use a default value or previous value
            
        # Create a zero tensor for all 38 features
        features = torch.zeros(1, 1, 38)  # Batch size 1, sequence length 1, 38 features
        
        
        features[0, 0, 0] = (temp_k - 273.15) / 50.0  # Temperature
        features[0, 0, 1] = hour / 23.0               # Hour
        features[0, 0, 2] = month / 12.0              # Month
        features[0, 0, 3] = current_load / 80000.0    # Load
        
        return features.to(self.device)

    def predict(self, temp_k: float, timestamp: datetime = None) -> Dict[str, Any]:
        """
        Make a load prediction using the current temperature and timestamp.
        
        Args:
            temp_k: Temperature in Kelvin
            timestamp: Timestamp for prediction (defaults to current time)
            
        Returns:
            Dictionary containing prediction and metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            current_time = datetime.now()
            
            inputs = self._preprocess_inputs(temp_k, current_time)
            
            with torch.no_grad():
                prediction = self.model(inputs)
            
            predicted_load = float(prediction[0, 0].item() * 80000.0)  # Denormalize
            
            return {
                "predicted_load_mw": predicted_load,
                "timestamp": timestamp.isoformat(),  # Use prediction timestamp
                "current_time": current_time.isoformat(),  # Add current time for reference
                "temperature_k": temp_k,
                "model_version": "lstm_4h_optimal",
                "status": "success",
                "all_outputs": prediction.tolist()  
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "timestamp": timestamp.isoformat() if timestamp else datetime.now().isoformat()
            }

if __name__ == "__main__":
    print("Testing Load Forecaster Model")
    print("-" * 50)
    
    model_path = 'lstm_4h_optimal_model.pt'
    print("\nInspecting model state dictionary...")
    state_dict = inspect_model_state_dict(model_path)
    
    if state_dict is not None:
        print("\nCreating LoadForecaster instance...")
        try:
            forecaster = LoadForecaster()
            
            print("\nTesting with dummy data...")
            current_temp_k = 298.15  # 25°C in Kelvin
            result = forecaster.predict(current_temp_k)
            
            print("\nPrediction Results:")
            print("-" * 50)
            for key, value in result.items():
                print(f"{key}: {value}")
                
        except Exception as e:
            print(f"\nError during testing: {str(e)}")
    else:
        print("\nFailed to load model state dictionary")
