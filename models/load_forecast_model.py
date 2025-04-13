import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Tuple, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LoadForecastLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, dropout=0.2):
        super(LoadForecastLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LoadForecaster:
    def __init__(self, model_path: str = 'lstm_4h_optimal_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = None  # Will be loaded from the model state dict
        
    def _load_model(self, model_path: str) -> LoadForecastLSTM:
        """Load the LSTM model from disk."""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model = LoadForecastLSTM()
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

    def _fetch_ercot_load(self) -> float:
        """Fetch current ERCOT load data."""
        api_key = os.getenv('ERCOT_API_KEY')
        if not api_key:
            raise ValueError("ERCOT API key not found in environment variables")

        url = "https://api.ercot.com/api/public-reports/np3-910-er/2d_agg_load_summary"
        headers = {
            "Ocp-Apim-Subscription-Key": api_key
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extract the most recent load value from the response
            # Note: You'll need to adjust this based on the actual response structure
            current_load = float(data['data'][0]['aggLoadSummary'])
            return current_load
        except Exception as e:
            raise RuntimeError(f"Failed to fetch ERCOT load data: {str(e)}")

    def _preprocess_inputs(self, temp_k: float, timestamp: datetime) -> torch.Tensor:
        """Preprocess the input features."""
        # Extract time features
        hour = timestamp.hour
        month = timestamp.month
        
        # Try to get current ERCOT load
        try:
            current_load = self._fetch_ercot_load()
        except Exception as e:
            print(f"Warning: Failed to fetch ERCOT load: {e}")
            current_load = 0  # Use a default value or previous value
            
        # Normalize features (you should use the same normalization as during training)
        # These values should match your training data normalization
        temp_norm = (temp_k - 273.15) / 50.0  # Normalize around typical temperature range
        hour_norm = hour / 23.0
        month_norm = month / 12.0
        load_norm = current_load / 80000.0  # Normalize around typical ERCOT load range
        
        # Create input tensor
        features = torch.tensor([[temp_norm, hour_norm, month_norm, load_norm]], dtype=torch.float32)
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
            # Preprocess inputs
            inputs = self._preprocess_inputs(temp_k, timestamp)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(inputs)
            
            # Denormalize prediction (adjust scale factor based on your training normalization)
            predicted_load = float(prediction.item() * 80000.0)  # Denormalize
            
            return {
                "predicted_load_mw": predicted_load,
                "timestamp": timestamp.isoformat(),
                "temperature_k": temp_k,
                "model_version": "lstm_4h_optimal",
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "timestamp": timestamp.isoformat() if timestamp else datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    # Test the model
    try:
        forecaster = LoadForecaster()
        
        # Test with current conditions
        current_temp_k = 298.15  # Example: 25°C in Kelvin
        result = forecaster.predict(current_temp_k)
        
        print("\nLoad Forecast Test Results:")
        print("-" * 50)
        for key, value in result.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error testing model: {str(e)}")
