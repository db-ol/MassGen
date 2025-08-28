#!/usr/bin/env python3
"""
Test script for MassGen anonymous vs non-anonymous voting modes

This script demonstrates how to use both voting modes and compare their behavior.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from massgen.orchestrator import Orchestrator
from massgen.chat_agent import SingleAgent

from massgen.backend.gemini import GeminiBackend
from massgen.backend.response import ResponseBackend
from massgen.backend.claude import ClaudeBackend


async def test_voting_modes():
    """Test both anonymous and non-anonymous voting modes."""
    
    print("🧪 Testing MassGen Voting Modes")
    print("=" * 50)
    
    # Create test agents
    agents = {
        "gemini": SingleAgent(
            agent_id="gemini",
            backend=GeminiBackend(api_key="test_key")  # Will use env var
        ),
        "gpt4": SingleAgent(
            agent_id="gpt4", 
            backend=ResponseBackend(api_key="test_key")  # Will use env var
        ),
        "claude": SingleAgent(
            agent_id="claude",
            backend=ClaudeBackend(api_key="test_key")  # Will use env var
        )
    }
    
    test_question = "What are the three main principles of object-oriented programming?"
    
    # Test 1: Anonymous Voting (default)
    print("\n🔍 Test 1: Anonymous Voting Mode")
    print("-" * 30)
    
    try:
        orchestrator_anon = Orchestrator(
            agents=agents,
            anonymous_voting=True
        )
        
        print(f"✅ Anonymous voting orchestrator created successfully")
        print(f"   Configuration: anonymous_voting = {orchestrator_anon.anonymous_voting}")
        
        # Check workflow tools
        vote_tool = None
        for tool in orchestrator_anon.workflow_tools:
            if tool.get("function", {}).get("name") == "vote":
                vote_tool = tool
                break
        
        if vote_tool:
            agent_id_param = vote_tool["function"]["parameters"]["properties"]["agent_id"]
            print(f"   Vote tool agent_id enum: {agent_id_param.get('enum', 'Not set')}")
            print(f"   Vote tool description: {agent_id_param.get('description', 'Not set')}")
        
    except Exception as e:
        print(f"❌ Error creating anonymous voting orchestrator: {e}")
    
    # Test 2: Non-Anonymous Voting
    print("\n🔍 Test 2: Non-Anonymous Voting Mode")
    print("-" * 30)
    
    try:
        orchestrator_nonanon = Orchestrator(
            agents=agents,
            anonymous_voting=False
        )
        
        print(f"✅ Non-anonymous voting orchestrator created successfully")
        print(f"   Configuration: anonymous_voting = {orchestrator_nonanon.anonymous_voting}")
        
        # Check workflow tools
        vote_tool = None
        for tool in orchestrator_nonanon.workflow_tools:
            if tool.get("function", {}).get("name") == "vote":
                vote_tool = tool
                break
        
        if vote_tool:
            agent_id_param = vote_tool["function"]["parameters"]["properties"]["agent_id"]
            print(f"   Vote tool agent_id enum: {agent_id_param.get('enum', 'Not set')}")
            print(f"   Vote tool description: {agent_id_param.get('description', 'Not set')}")
        
    except Exception as e:
        print(f"❌ Error creating non-anonymous voting orchestrator: {e}")
    
    # Test 3: Message Templates
    print("\n🔍 Test 3: Message Template Differences")
    print("-" * 30)
    
    try:
        from massgen.message_templates import MessageTemplates
        
        templates = MessageTemplates()
        
        # Test anonymous voting templates
        anon_tools = templates.get_standard_tools(list(agents.keys()), anonymous_voting=True)
        anon_vote_tool = next((t for t in anon_tools if t.get("function", {}).get("name") == "vote"), None)
        
        if anon_vote_tool:
            print(f"✅ Anonymous voting tools created")
            agent_id_param = anon_vote_tool["function"]["parameters"]["properties"]["agent_id"]
            print(f"   Agent ID enum: {agent_id_param.get('enum', 'Not set')}")
            print(f"   Description: {agent_id_param.get('description', 'Not set')}")
        
        # Test non-anonymous voting templates
        nonanon_tools = templates.get_standard_tools(list(agents.keys()), anonymous_voting=False)
        nonanon_vote_tool = next((t for t in nonanon_tools if t.get("function", {}).get("name") == "vote"), None)
        
        if nonanon_vote_tool:
            print(f"✅ Non-anonymous voting tools created")
            agent_id_param = nonanon_vote_tool["function"]["parameters"]["properties"]["agent_id"]
            print(f"   Agent ID enum: {agent_id_param.get('enum', 'Not set')}")
            print(f"   Description: {agent_id_param.get('description', 'Not set')}")
        
        # Test answer formatting
        test_answers = {
            "gemini": "Encapsulation, inheritance, and polymorphism",
            "gpt4": "Abstraction, encapsulation, and inheritance", 
            "claude": "Encapsulation, inheritance, and polymorphism"
        }
        
        anon_format = templates.format_current_answers_with_summaries(test_answers, anonymous_voting=True)
        nonanon_format = templates.format_current_answers_with_summaries(test_answers, anonymous_voting=False)
        
        print(f"\n📝 Answer Formatting Comparison:")
        print(f"   Anonymous format preview: {anon_format[:100]}...")
        print(f"   Non-anonymous format preview: {nonanon_format[:100]}...")
        
    except Exception as e:
        print(f"❌ Error testing message templates: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Testing Complete!")
    print("\nTo run actual experiments:")
    print("1. Set up your API keys in environment variables")
    print("2. Use the voting_comparison_example.yaml config")
    print("3. Run with --non-anonymous-voting flag to compare modes")
    print("\nExample commands:")
    print("  # Anonymous voting (default)")
    print("  uv run python -m massgen.cli --config massgen/configs/voting_comparison_example.yaml 'Your question'")
    print("\n  # Non-anonymous voting")
    print("  uv run python -m massgen.cli --config massgen/configs/voting_comparison_example.yaml --non-anonymous-voting 'Your question'")


if __name__ == "__main__":
    asyncio.run(test_voting_modes())
