"""
Script to extract all exact match questions and answers from HLE dataset
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import load_dataset
sys.path.append(str(Path(__file__).parent))

from load_dataset import HLEDatasetLoader


def extract_exact_match_questions_to_file(output_file: str = "exact_match_questions.txt"):
    """
    Extract all exact match questions and answers to a text file.
    
    Args:
        output_file: Path to the output text file
    """
    try:
        # Initialize the dataset loader
        loader = HLEDatasetLoader()
        
        # Load all exact match questions
        print("📚 Loading exact match questions from HLE dataset...")
        questions = loader.load_exact_match_only()
        
        print(f"📊 Found {len(questions)} exact match questions")
        
        # Write questions to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("HLE LITE - EXACT MATCH QUESTIONS AND ANSWERS\n")
            f.write("=" * 60 + "\n\n")
            
            for i, question in enumerate(questions, 1):
                f.write(f"QUESTION {i}\n")
                f.write("-" * 40 + "\n")
                f.write(f"ID: {question['id']}\n")
                f.write(f"Subject: {question.get('subject', 'Unknown')}\n")
                f.write(f"Category: {question.get('category', 'Unknown')}\n")
                f.write(f"Answer Type: {question['answer_type']}\n\n")
                
                f.write("QUESTION:\n")
                f.write(question['original_question'])
                f.write("\n\n")
                
                f.write("CORRECT ANSWER:\n")
                f.write(question['answer'])
                f.write("\n\n")
                
                f.write("=" * 60 + "\n\n")
        
        print(f"✅ Successfully extracted {len(questions)} questions to {output_file}")
        print(f"�� File location: {os.path.abspath(output_file)}")
        
        # Also create a summary file
        summary_file = "exact_match_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("HLE LITE - EXACT MATCH QUESTIONS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Questions: {len(questions)}\n\n")
            
            # Group by subject
            subjects = {}
            for q in questions:
                subject = q.get('subject', 'Unknown')
                if subject not in subjects:
                    subjects[subject] = []
                subjects[subject].append(q)
            
            f.write("QUESTIONS BY SUBJECT:\n")
            f.write("-" * 30 + "\n")
            for subject, qs in subjects.items():
                f.write(f"{subject}: {len(qs)} questions\n")
            
            f.write("\n\nSAMPLE QUESTIONS (first 5):\n")
            f.write("-" * 30 + "\n")
            for i, q in enumerate(questions[:5], 1):
                f.write(f"{i}. {q['original_question'][:100]}...\n")
                f.write(f"   Answer: {q['answer']}\n\n")
        
        print(f"�� Summary created: {summary_file}")
        
        return questions
        
    except Exception as e:
        print(f"❌ Error extracting questions: {e}")
        return None


def main():
    """Main function to run the extraction."""
    print("🚀 HLE Exact Match Questions Extractor")
    print("=" * 50)
    
    # Extract questions
    questions = extract_exact_match_questions_to_file()
    
    if questions:
        print(f"\n�� Extraction Summary:")
        print(f"   - Total questions: {len(questions)}")
        print(f"   - Output file: exact_match_questions.txt")
        print(f"   - Summary file: exact_match_summary.txt")
        
        # Show first few questions as preview
        print(f"\n📝 Preview (first 3 questions):")
        for i, q in enumerate(questions[:3], 1):
            print(f"\nQuestion {i}:")
            print(f"  Subject: {q.get('subject', 'Unknown')}")
            print(f"  Question: {q['original_question'][:100]}...")
            print(f"  Answer: {q['answer']}")
    else:
        print("❌ Failed to extract questions")


if __name__ == "__main__":
    main()
