"""
Dataset loading module - can be used as standalone script or imported module
Supports multiple datasets including HLE, BigBenchHard, MuSR, and Hendrycks Math
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


class DatasetLoader:
    """Generic dataset loader for various benchmark datasets."""
    
    def __init__(self, token: str = None, dataset_name: str = "cais/hle"):
        """Initialize with optional token and dataset name."""
        self.token = token or os.getenv("HF_API_KEY")
        if not self.token:
            raise ValueError("HF_API_KEY not found in environment variables")
        self.dataset_name = dataset_name
        
    def load_dataset_by_name(self, question_type: str = "multipleChoice", filter_params: Dict = None) -> List[Dict]:
        """Load dataset based on the dataset name provided during initialization."""
        if self.dataset_name == "cais/hle" or self.dataset_name == "koiwave/hle-lite":
            return self.load_hle_dataset([question_type])
        elif self.dataset_name == "maveriq/bigbenchhard":
            return self.load_bigbenchhard_dataset(question_type)
        elif self.dataset_name == "TAUR-Lab/MuSR":
            return self.load_musr_dataset(question_type)
        elif self.dataset_name == "EleutherAI/hendrycks_math":
            return self.load_hendrycks_math_dataset(question_type, filter_params)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
    def load_hle_dataset(self, question_types: List[str] = None) -> List[Dict]:
        """Load and preprocess HLE dataset for benchmarking."""
        print(f"📚 Loading HLE dataset from {self.dataset_name}...")
        
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
        
    def load_bigbenchhard_dataset(self, question_type: str = "multipleChoice") -> List[Dict]:
        """Load and preprocess BigBenchHard dataset for benchmarking."""
        print(f"📚 Loading BigBenchHard dataset from {self.dataset_name}...")
        
        # Get config_name from benchmark config if available
        config_name = None
        if hasattr(self, 'config') and self.config and 'benchmark' in self.config:
            config_name = self.config['benchmark'].get('config_name')
        
        # If config_name is not available, use 'boolean_expressions' as default
        if not config_name:
            config_name = 'boolean_expressions'
            print(f"⚠️ No config_name specified, using default: {config_name}")
        
        print(f"📋 Using BigBenchHard config: {config_name}")
        dataset = load_dataset("maveriq/bigbenchhard", config_name)
        
        questions = []
        for sample in dataset['train']:  # Using 'train' split for BigBenchHard
            question_text = sample.get('input', '')
            
            # For boolean_expressions, the choices are True and False
            if config_name == 'boolean_expressions':
                choices = ['True', 'False']
                target = sample.get('target', '')
                
                # Set correct answer based on target
                if target in choices:
                    correct_index = choices.index(target)
                    correct_answer = chr(65 + correct_index)  # Convert to A, B, C, etc.
                else:
                    correct_answer = "Unknown"
                    print(f"⚠️ Warning: Could not determine correct answer for question: {question_text[:50]}...")
            else:
                # For other configs, try to use target_scores if available
                choices = sample.get('choices', [])
                if 'target_scores' in sample:
                    answer_index = sample.get('target_scores', {})
                    
                    # Find the correct answer index
                    correct_index = -1
                    for i, score in answer_index.items():
                        if score == 1:
                            correct_index = int(i)
                            break
                            
                    if correct_index >= 0 and correct_index < len(choices):
                        correct_answer = chr(65 + correct_index)  # Convert to A, B, C, etc.
                    else:
                        correct_answer = "Unknown"
                        print(f"⚠️ Warning: Could not determine correct answer for question: {question_text[:50]}...")
                elif 'target' in sample:
                    # If target is available but not target_scores
                    target = sample.get('target', '')
                    if target in choices:
                        correct_index = choices.index(target)
                        correct_answer = chr(65 + correct_index)  # Convert to A, B, C, etc.
                    else:
                        correct_answer = "Unknown"
                        print(f"⚠️ Warning: Could not determine correct answer for question: {question_text[:50]}...")
                else:
                    correct_answer = "Unknown"
                    print(f"⚠️ Warning: No target or target_scores found for question: {question_text[:50]}...")
                
            # Format choices as A, B, C, etc.
            formatted_choices = ""
            for i, choice in enumerate(choices):
                formatted_choices += f"{chr(65 + i)}. {choice}\n"
                
            formatted_question = f"{question_text}\n\nPlease choose from the following options:\n{formatted_choices}\nThe answer is:"
            
            processed_question = {
                'id': sample.get('idx', ''),
                'original_question': question_text,
                'formatted_question': formatted_question,
                'answer': correct_answer,
                'answer_type': 'multipleChoice',
                'subject': 'boolean_logic',
                'category': 'reasoning'
            }
            
            questions.append(processed_question)
            
        print(f"📊 Loaded {len(questions)} questions")
        return questions
        
    def load_musr_dataset(self, question_type: str = "multipleChoice") -> List[Dict]:
        """Load and preprocess MuSR dataset for benchmarking."""
        print(f"📚 Loading MuSR dataset from {self.dataset_name}...")
        
        dataset = load_dataset(self.dataset_name, token=self.token)
        print(f"📊 Available splits: {list(dataset.keys())}")
        print(f"📊 Dataset structure: {dataset}")
        
        # Use the first available split if 'train' doesn't exist
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        print(f"📊 Using split: {split_name}")
        
        questions = []
        for sample in dataset[split_name]:  # Using available split for MuSR
            narrative = sample.get('narrative', '')
            question_text = sample.get('question', '')
            choices = sample.get('choices', [])
            answer_choice = sample.get('answer_choice', '')
            
            # Parse choices string if it's a string representation of a list
            if isinstance(choices, str):
                import ast
                try:
                    choices = ast.literal_eval(choices)
                except (ValueError, SyntaxError):
                    choices = []
            
            # Format choices as A, B, C, etc.
            formatted_choices = ""
            choice_to_letter = {}
            for i, choice in enumerate(choices):
                letter = chr(65 + i)
                formatted_choices += f"{letter}. {choice}\n"
                choice_to_letter[choice] = letter
                
            # Convert answer_choice (person name) to letter format (A, B, C, etc.)
            correct_answer = choice_to_letter.get(answer_choice, "Unknown") if answer_choice else "Unknown"
                
            formatted_question = f"Story: {narrative}\n\nQuestion: {question_text}\n\nPlease choose from the following options:\n{formatted_choices}\nThe answer is:"
            
            processed_question = {
                'id': str(len(questions)),
                'original_question': question_text,
                'formatted_question': formatted_question,
                'answer': correct_answer,
                'answer_type': 'multipleChoice',
                'subject': 'story_understanding',
                'category': 'reading_comprehension'
            }
            
            questions.append(processed_question)
            
        print(f"📊 Loaded {len(questions)} questions")
        return questions
        
    def load_hendrycks_math_dataset(self, question_type: str = "multipleChoice", filter_params: Dict = None) -> List[Dict]:
        """Load and preprocess Hendrycks Math dataset for benchmarking."""
        print(f"📚 Loading Hendrycks Math dataset from {self.dataset_name}...")
        
        dataset = load_dataset("EleutherAI/hendrycks_math", token=self.token)
        
        # Apply filters if provided
        level = filter_params.get('level', None) if filter_params else None
        
        questions = []
        for sample in dataset['test']:  # Using 'test' split for Hendrycks Math
            # Filter by level if specified
            if level and str(sample.get('level')) != str(level):
                continue
                
            problem = sample.get('problem', '')
            solution = sample.get('solution', '')
            
            # Extract the final answer from the solution
            import re
            answer_match = re.search(r'The answer is\s*(\w+)', solution)
            if answer_match:
                answer = answer_match.group(1)
            else:
                # Try to find the answer at the end of the solution
                lines = solution.strip().split('\n')
                answer = lines[-1].strip() if lines else ""
            
            formatted_question = f"{problem}\n\nPlease solve this math problem step by step and provide the final answer in the format 'The answer is: X'."
            
            processed_question = {
                'id': str(len(questions)),
                'original_question': problem,
                'formatted_question': formatted_question,
                'answer': answer,
                'answer_type': 'exactMatch',
                'subject': sample.get('type', 'math'),
                'category': f"level_{sample.get('level', 'unknown')}"
            }
            
            questions.append(processed_question)
            
        print(f"📊 Loaded {len(questions)} questions")
        return questions


class HLEDatasetLoader(DatasetLoader):
    """Legacy class for backward compatibility."""
    
    def __init__(self, token: str = None):
        """Initialize with optional token."""
        super().__init__(token, "koiwave/hle-lite")
    
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
