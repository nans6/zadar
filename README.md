# Energy Trading Agent Project

This project implements a modular energy trading agent that uses weather, market, and grid data to make intelligent hedging decisions using LLM-based reasoning.

## Project Structure

### Core Components

- `data_fetcher.py`: Perception layer that handles all external API interactions
  - Weather data fetching
  - Market data integration
  - Grid status monitoring
  
- `agent_reasoner.py`: LLM-based reasoning module
  - Formats and structures prompts
  - Handles LLM API calls
  - Processes raw LLM responses
  
- `strategy_engine.py`: Action generation module
  - Parses LLM reasoning output
  - Converts high-level decisions into concrete trading actions
  - Implements safety checks and validation
  
- `main_agent.py`: Main orchestrator
  - Implements the perceive → reason → act loop
  - Coordinates between all components
  - Handles main execution flow

### Models Package

- `models/load_forecast_model.py`: Predicts grid load based on weather forecasts
- `models/renewables_model.py`: Predicts renewable energy generation from weather data
- `models/price_risk_model.py`: Models and predicts price spike risks
- `models/third_party/`: Directory for external ML models or licensed code

### Configuration

- `.env`: Configuration file for storing:
  - API keys
  - Model parameters
  - Environment variables
  - Connection strings

## Development Guidelines

1. **Modularity**: Each component is designed to be independent and replaceable
2. **Parallel Development**: Structure allows multiple developers to work simultaneously
3. **Clean Architecture**: Clear separation between perception, reasoning, and action layers
4. **Extensibility**: Easy to add new models or swap existing components

## Getting Started

1. Copy `.env.example` to `.env` and fill in required credentials
2. Install dependencies (requirements.txt to be added)
3. Run `main_agent.py` to start the system

## License

[License information to be added] 