"""
HLE Dataset loading module - can be used as standalone script or imported module
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset


def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip("\"'")
                    os.environ[key] = value


# Load .env file at module import
load_env_file()


class HLEDatasetLoader:
    """Shared dataset loader for HLE Lite questions."""
    
    def __init__(self, token: str = None):
        """Initialize with optional token."""
        self.token = token or os.getenv("HF_API_KEY")
        if not self.token:
            raise ValueError("HF_API_KEY not found in environment variables")
    
    def load_dataset(self, question_types: List[str] = None) -> List[Dict]:
        """Load and preprocess HLE dataset for benchmarking.
        
        Args:
            question_types: List of question types to include. 
                          Options: ['multipleChoice', 'exactMatch']
                          If None, loads all types.
        """
        print("📚 Loading HLE dataset...")
        
        dataset = load_dataset("koiwave/hle-lite", token=self.token)
        
        # Filter questions by type
        questions = []
        for sample in dataset['test']:
            answer_type = sample.get('answer_type')
            
            # Skip if not in requested types
            if question_types and answer_type not in question_types:
                continue
                
            question_text = sample.get('question', '')
            
            if answer_type == 'multipleChoice':
                # Format multiple choice questions
                if '\n\nAnswer Choices:\n' in question_text:
                    parts = question_text.split('\n\nAnswer Choices:\n')
                    base_question = parts[0].strip()
                    options_text = parts[1].strip()
                    formatted_question = f"{base_question}\n\nPlease choose from the following options:\n{options_text}\n\nThe answer is:"
                else:
                    formatted_question = f"{question_text}\n\nThe answer is:"
                    
            elif answer_type == 'exactMatch':
                # Format exact match questions
                formatted_question = f"{question_text}\n\nPlease provide the exact answer:"
                
            else:
                # Unknown type, use as is
                formatted_question = f"{question_text}\n\nThe answer is:"
            
            # Create processed question
            processed_question = {
                'id': sample.get('id'),
                'original_question': sample.get('question'),
                'formatted_question': formatted_question,
                'answer': sample.get('answer'),
                'answer_type': answer_type,
                'subject': sample.get('raw_subject'),
                'category': sample.get('category')
            }
            
            questions.append(processed_question)
        
        print(f"📊 Loaded {len(questions)} questions")
        return questions
    
    def load_multiple_choice_only(self) -> List[Dict]:
        """Load only multiple choice questions (for backward compatibility)."""
        return self.load_dataset(['multipleChoice'])
    
    def load_exact_match_only(self) -> List[Dict]:
        """Load only exact match questions."""
        return self.load_dataset(['exactMatch'])
    
    def get_sample_questions(self, num_samples: int = 3) -> List[Dict]:
        """Get sample questions for exploration."""
        dataset = load_dataset("koiwave/hle-lite", token=self.token)
        
        # Get all question types
        questions = []
        for sample in dataset['test']:
            if isinstance(sample, dict):
                questions.append(sample)
        
        return questions[:num_samples]
    
    def print_dataset_info(self):
        """Print dataset information and sample questions."""
        dataset = load_dataset("koiwave/hle-lite", token=self.token)
        
        if 'test' in dataset:
            # Count ALL question types
            question_types = {}
            for sample in dataset['test']:
                answer_type = sample.get('answer_type', 'unknown')
                question_types[answer_type] = question_types.get(answer_type, 0) + 1
            
            # Print only the distribution
            for qtype, count in question_types.items():
                print(f"{qtype}: {count}")


def main():
    """Standalone function for testing dataset access."""
    try:
        loader = HLEDatasetLoader()
        loader.print_dataset_info()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
