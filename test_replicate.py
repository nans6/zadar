import os
from dotenv import load_dotenv
import replicate

def test_replicate_api():
    # Load environment variables
    load_dotenv()
    
    # Verify API token is set
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("Error: REPLICATE_API_TOKEN not found in environment variables")
        return
    
    try:
        print("\nGenerating Nicknames for Ananya:")
        # Run DeepSeek-R1 model with streaming
        for event in replicate.stream(
            "deepseek-ai/deepseek-r1",
            input={
                "prompt": "Generate 5 creative and fun nicknames for someone named Ananya. Make them playful and positive. Format the response as a simple numbered list.",
                "temperature": 0.7,
                "max_tokens": 20480,  # Increased to maximum supported size
                "top_p": 0.9,
                "presence_penalty": 0,
                "frequency_penalty": 0
            }
        ):
            print(str(event), end="")
        
    except Exception as e:
        print(f"\nError running model: {str(e)}")

if __name__ == "__main__":
    test_replicate_api() 