import torch
import sys

def test_load_model():
    try:
        # Try to load the model with CPU mapping
        print("Attempting to load model...")
        model = torch.load('lstm_4h_optimal_model.pt', map_location=torch.device('cpu'))
        print("\nModel loaded successfully!")
        print("\nModel information:")
        print(f"Type: {type(model)}")
        if hasattr(model, 'state_dict'):
            print("\nModel architecture:")
            for name, param in model.named_parameters():
                print(f"{name}: {param.size()}")
        return True
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}\n")
    
    success = test_load_model()
    sys.exit(0 if success else 1) 