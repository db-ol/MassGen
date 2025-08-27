#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Add massgen to path
sys.path.insert(0, str(Path(__file__).parent))

from massgen.cli import create_agents_from_config
import yaml

async def test_multi_agent():
    # Load config
    with open("massgen/configs/four_models_mmlu_pro.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create agents
    agents = create_agents_from_config(config)
    
    # Test question
    question = "What is 2+2?"
    print(f"Testing: {question}")
    print("=" * 50)
    
    # Get response from each agent
    for agent_id, agent in agents.items():
        print(f"\n🤖 {agent_id}:")
        try:
            response = await agent.get_response(question)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    asyncio.run(test_multi_agent())
