#!/usr/bin/env python3
"""
Automated parallel runner for all single model experiments on MuSR dataset.
This script runs all available single models in parallel using the API keys from .env file.
Each model will process all three MuSR tasks: murder_mysteries, object_placements, team_allocation.
"""

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_api_key(key_name):
    """Check if API key is available and not empty"""
    key = os.getenv(key_name)
    return key is not None and key.strip() != "" and not key.startswith("your-")

def run_single_model_script(script_name, model_name):
    """Run a single model script and return results"""
    print(f"🚀 Starting {model_name} experiment...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["python", script_name],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout (increased for processing all 3 tasks)
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_name} completed successfully in {duration:.2f}s")
            return {
                "model": model_name,
                "script": script_name,
                "status": "success",
                "duration": duration,
                "output": result.stdout
            }
        else:
            print(f"❌ {model_name} failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return {
                "model": model_name,
                "script": script_name,
                "status": "failed",
                "duration": duration,
                "error": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} timed out after 2 hours")
        return {
            "model": model_name,
            "script": script_name,
            "status": "timeout",
            "duration": 7200
        }
    except Exception as e:
        print(f"💥 {model_name} crashed with exception: {e}")
        return {
            "model": model_name,
            "script": script_name,
            "status": "crashed",
            "error": str(e)
        }

def main():
    print("🔍 Checking available API keys...")
    
    # Define model configurations
    models = [
        {
            "name": "GPT-5",
            "script": "gpt5_musr_parallel.py",
            "api_key": "OPENAI_API_KEY"
        },
        {
            "name": "Gemini 2.5 Pro",
            "script": "gemini25pro_musr_parallel.py",
            "api_key": "GEMINI_API_KEY"
        },
        {
            "name": "Grok-4",
            "script": "grok4_musr_parallel.py",
            "api_key": "XAI_API_KEY"
        },
        {
            "name": "Claude Sonnet 4",
            "script": "claude_sonnet4_musr_parallel.py",
            "api_key": "ANTHROPIC_API_KEY"
        }
    ]
    
    # Filter models with available API keys
    available_models = []
    for model in models:
        if check_api_key(model["api_key"]):
            print(f"✅ {model['name']}: API key available")
            available_models.append(model)
        else:
            print(f"❌ {model['name']}: API key missing or invalid")
    
    if not available_models:
        print("\n❌ No valid API keys found. Please check your .env file.")
        return
    
    print(f"\n🚀 Starting parallel execution of {len(available_models)} models...")
    print("📋 Each model will process all 3 MuSR tasks: murder_mysteries, object_placements, team_allocation")
    print("=" * 60)
    
    # Run models in parallel
    results = []
    with ThreadPoolExecutor(max_workers=len(available_models)) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(run_single_model_script, model["script"], model["name"]): model
            for model in available_models
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            result = future.result()
            results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print("\n🎉 Successful experiments:")
        for result in successful:
            print(f"  - {result['model']}: {result['duration']:.2f}s")
    
    if failed:
        print("\n💥 Failed experiments:")
        for result in failed:
            print(f"  - {result['model']}: {result['status']}")
            if "error" in result:
                print(f"    Error: {result['error'][:100]}...")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"single_models_results_{timestamp}.txt"
    
    with open(results_file, "w") as f:
        f.write("MuSR Single Models Parallel Experiment Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Script: {result['script']}\n")
            f.write(f"Status: {result['status']}\n")
            if "duration" in result:
                f.write(f"Duration: {result['duration']:.2f}s\n")
            if "output" in result:
                f.write(f"Output: {result['output'][:500]}...\n")
            if "error" in result:
                f.write(f"Error: {result['error']}\n")
            f.write("-" * 40 + "\n\n")
    
    print(f"\n📝 Detailed results saved to: {results_file}")
    print("\n🏁 All experiments completed!")

if __name__ == "__main__":
    main()