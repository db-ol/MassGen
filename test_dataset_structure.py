#!/usr/bin/env python3
"""Test script to check MMLU-Pro dataset structure."""

import os
from datasets import load_dataset

def test_mmlu_pro_structure():
    """Test the structure of MMLU-Pro dataset."""
    token = os.getenv("HF_API_KEY")
    if not token:
        print("❌ HF_API_KEY not found")
        return
    
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", token=token)
        
        print("📊 MMLU-Pro Dataset Structure:")
        print(f"Keys: {list(dataset.keys())}")
        
        if 'test' in dataset:
            sample = dataset['test'][0]
            print(f"\n📝 Sample Question Structure:")
            print(f"Keys: {list(sample.keys())}")
            
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"{key}: {value[:100]}...")
                else:
                    print(f"{key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_mmlu_pro_structure()
