# main_agent.py
import logging
import time
import sys
import json
from agent_reasoner import AgentReasoner
from api_manager import APIManager
from pprint import pprint, pformat
from datetime import datetime

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_agent_cycle(max_cycles=2):
    """Run the main agent cycle with autonomous API decisions"""
    results = []
    
    # Initialize components
    api_manager = APIManager()
    agent = AgentReasoner(api_manager)
    
    try:
        cycle_count = 0
        while cycle_count < max_cycles:
            cycle_data = {
                'cycle_num': cycle_count + 1,
                'timestamp': datetime.now().isoformat(),
                'data_needs': None,
                'cached_data': None,
                'strategy': None,
                'next_cycle_time': None
            }
            
            # Agent autonomously decides what data it needs
            agent.assess_data_needs()
            cycle_data['data_needs'] = list(agent.data_needs)
            
            # Agent fetches required data
            agent.fetch_required_data()
            cycle_data['cached_data'] = agent.cached_data
            
            # Generate strategy based on available data
            strategy = agent.generate_strategy()
            cycle_data['strategy'] = strategy
            
            # Calculate next cycle time
            sleep_time = agent.calculate_next_cycle_time()
            cycle_data['next_cycle_time'] = sleep_time
            
            results.append(cycle_data)
            
            if cycle_count < max_cycles - 1:
                time.sleep(min(sleep_time, 5))
            
            cycle_count += 1
            
    except Exception as e:
        results.append({
            'error': str(e),
            'cycle_num': cycle_count + 1,
            'timestamp': datetime.now().isoformat()
        })
    
    # Write results to file
    with open('agent_results.txt', 'w') as f:
        f.write("=== Agent Execution Results ===\n\n")
        for cycle in results:
            f.write(f"Cycle {cycle['cycle_num']} at {cycle['timestamp']}\n")
            f.write("-" * 50 + "\n")
            
            if 'error' in cycle:
                f.write(f"ERROR: {cycle['error']}\n")
                continue
                
            f.write("Data Needs:\n")
            f.write(pformat(cycle['data_needs']) + "\n\n")
            
            f.write("Cached Data:\n")
            f.write(pformat(cycle['cached_data']) + "\n\n")
            
            f.write("Generated Strategy:\n")
            f.write(pformat(cycle['strategy']) + "\n\n")
            
            f.write(f"Next Cycle Time: {cycle['next_cycle_time']} seconds\n")
            f.write("=" * 50 + "\n\n")

if __name__ == "__main__":
    run_agent_cycle()
