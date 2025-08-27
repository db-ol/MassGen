"""
Dataset loading module - supports both HLE Lite and MMLU-Pro datasets
"""

import os
import random
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


class MMLUProDatasetLoader:
    """Dataset loader for MMLU-Pro questions."""
    
    def __init__(self, token: str = None):
        """Initialize with optional token."""
        self.token = token or os.getenv("HF_API_KEY")
        if not self.token:
            raise ValueError("HF_API_KEY not found in environment variables")
    
    def load_dataset(self, question_types: List[str] = None, max_samples: int = 100) -> List[Dict]:
        """Load and preprocess MMLU-Pro dataset for benchmarking.
        
        Args:
            question_types: List of question types to include. 
                          Options: ['mcq', 'exact_match']
                          If None, loads all types.
            max_samples: Maximum number of samples to load (default: 100)
        """
        print("📚 Loading MMLU-Pro dataset...")
        
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", token=self.token)
        
        # Filter questions by type
        questions = []
        for sample in dataset['test']:
            # Determine question type based on MMLU-Pro format
            question_text = sample.get('question', '')
            options = sample.get('options', [])
            
            # Classify as MCQ if options are provided, otherwise exact match
            if options and len(options) > 0:
                answer_type = 'mcq'
            else:
                answer_type = 'exact_match'
            
            # Skip if not in requested types
            if question_types and answer_type not in question_types:
                continue
            
            # Format question based on type
            if answer_type == 'mcq':
                # Format multiple choice questions with proper option letters
                options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                formatted_question = f"{question_text}\n\nPlease choose from the following options:\n{options_text}\n\nThe answer is:"
            else:
                # Format exact match questions
                formatted_question = f"{question_text}\n\nPlease provide the exact answer:"
            
            # Create processed question with proper ID handling
            question_id = sample.get('question_id', f"mmlu_pro_{len(questions)+1:04d}")
            
            processed_question = {
                'id': question_id,  # ✅ Now using the actual question_id
                'original_question': sample.get('question'),
                'formatted_question': formatted_question,
                'answer': sample.get('answer'),  # ✅ This will be "I" for the sample
                'answer_type': answer_type,
                'subject': sample.get('category'),
                'category': sample.get('src'),
                'options': sample.get('options', []),
                'answer_index': sample.get('answer_index')  # ✅ Add this for debugging
            }
            
            questions.append(processed_question)
        
        # Randomly sample if we have more than max_samples
        if len(questions) > max_samples:
            questions = random.sample(questions, max_samples)
            print(f" Randomly sampled {max_samples} questions from {len(questions)} available")
        else:
            print(f"📊 Loaded {len(questions)} questions")
        
        return questions
    
    def load_mcq_only(self, max_samples: int = 100) -> List[Dict]:
        """Load only multiple choice questions."""
        return self.load_dataset(['mcq'], max_samples)
    
    def load_exact_match_only(self, max_samples: int = 100) -> List[Dict]:
        """Load only exact match questions."""
        return self.load_dataset(['exact_match'], max_samples)
    
    def print_dataset_info(self):
        """Print dataset information and sample questions."""
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", token=self.token)
        
        if 'test' in dataset:
            # Count question types
            question_types = {'mcq': 0, 'exact_match': 0}
            for sample in dataset['test']:
                options = sample.get('options', [])
                if options and len(options) > 0:
                    question_types['mcq'] += 1
                else:
                    question_types['exact_match'] += 1
            
            print("MMLU-Pro Dataset Distribution:")
            for qtype, count in question_types.items():
                print(f"{qtype}: {count}")


def get_dataset_loader(dataset_name: str, token: str = None):
    """Factory function to get the appropriate dataset loader."""
    if dataset_name.lower() in ['hle', 'hle-lite', 'koiwave/hle-lite']:
        return HLEDatasetLoader(token)
    elif dataset_name.lower() in ['mmlu-pro', 'mmlu_pro', 'tiger-lab/mmlu-pro']:
        return MMLUProDatasetLoader(token)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'hle-lite', 'mmlu-pro'")


def main():
    """Standalone function for testing dataset access."""
    try:
        print("=== HLE Dataset Info ===")
        hle_loader = HLEDatasetLoader()
        hle_loader.print_dataset_info()
        
        print("\n=== MMLU-Pro Dataset Info ===")
        mmlu_loader = MMLUProDatasetLoader()
        mmlu_loader.print_dataset_info()
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
