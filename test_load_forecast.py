from models.load_forecast_model import LoadForecaster
from datetime import datetime
import os
from dotenv import load_dotenv

def test_load_forecast():
    # Load environment variables
    load_dotenv()
    
    # Verify ERCOT API key is set
    api_key = os.getenv('ERCOT_API_KEY')
    if not api_key:
        print("Warning: ERCOT_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize forecaster
        print("Initializing load forecaster...")
        forecaster = LoadForecaster()
        
        # Test cases with different temperatures and times
        test_cases = [
            (298.15, datetime.now()),  # Current time, 25°C
            (308.15, datetime.now()),  # Current time, 35°C
            (288.15, datetime.now()),  # Current time, 15°C
        ]
        
        # Run predictions
        print("\nRunning test predictions...")
        for temp_k, timestamp in test_cases:
            print(f"\nTest case: Temperature = {temp_k}K ({temp_k - 273.15:.1f}°C)")
            result = forecaster.predict(temp_k, timestamp)
            
            print("Results:")
            print("-" * 40)
            for key, value in result.items():
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"\nError during testing: {str(e)}")

if __name__ == "__main__":
    test_load_forecast() 