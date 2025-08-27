import json
import glob
import re
from pathlib import Path
from typing import Dict, List, Any

def extract_results_from_coordination_logs():
    """Extract benchmark results from coordination log files."""
    
    # Find all coordination log files
    log_files = glob.glob("mass_coordination_20250826_*.json")
    log_files.sort()  # Sort by timestamp
    
    print(f"Found {len(log_files)} coordination log files")
    
    results = {
        "total_questions": len(log_files),
        "questions": [],
        "summary": {
            "correct_answers": 0,
            "incorrect_answers": 0,
            "accuracy": 0.0
        }
    }
    
    for i, log_file in enumerate(log_files, 1):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract question and answer information
            question_data = {
                "question_number": i,
                "log_file": log_file,
                "question": data.get("question", "Unknown"),
                "model_answer": data.get("response", "No response"),
                "selected_agent": data.get("selected_agent", "Unknown"),
                "vote_results": data.get("vote_results", {}),
                "success": data.get("success", False)
            }
            
            # Extract the answer from the model response
            model_answer = data.get("response", "")
            extracted_answer = extract_answer_from_response(model_answer)
            question_data["extracted_answer"] = extracted_answer
            
            # Try to find correct answer from the question text
            correct_answer = extract_correct_answer_from_question(data.get("question", ""))
            question_data["correct_answer"] = correct_answer
            
            # Determine if answer is correct
            is_correct = (extracted_answer and correct_answer and 
                         extracted_answer.strip().upper() == correct_answer.strip().upper())
            question_data["is_correct"] = is_correct
            
            if is_correct:
                results["summary"]["correct_answers"] += 1
            else:
                results["summary"]["incorrect_answers"] += 1
            
            results["questions"].append(question_data)
            
            print(f"Question {i}: Extracted={extracted_answer}, Correct={correct_answer}, {'✅' if is_correct else '❌'}")
            
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            results["questions"].append({
                "question_number": i,
                "log_file": log_file,
                "error": str(e)
            })
    
    # Calculate accuracy
    if results["summary"]["correct_answers"] + results["summary"]["incorrect_answers"] > 0:
        results["summary"]["accuracy"] = (
            results["summary"]["correct_answers"] / 
            (results["summary"]["correct_answers"] + results["summary"]["incorrect_answers"])
        )
    
    return results

def extract_answer_from_response(response: str) -> str:
    """Extract answer from model response."""
    if not response:
        return "No answer found"
    
    # Look for "The answer is: X" pattern
    pattern = r"The answer is:\s*([A-Z])"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Look for LaTeX boxed format: $\boxed{X}$
    pattern = r"\\boxed\{([A-Z])\}"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Look for single letter at the end
    lines = response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if len(line) == 1 and line.isalpha():
            return line.upper()
        # Check for patterns like "Answer: A" or "A)" or "A."
        match = re.search(r'[A-Z]\)?\.?$', line)
        if match:
            return match.group(0)[0].upper()
    
    return "No answer found"

def extract_correct_answer_from_question(question: str) -> str:
    """Extract correct answer from question text."""
    # Look for "Correct Answer: X" pattern
    pattern = r"Correct Answer:\s*([A-Z])"
    match = re.search(pattern, question, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Look for "The answer is:" at the end of question
    pattern = r"The answer is:\s*([A-Z])"
    match = re.search(pattern, question, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return "Unknown"

def save_results_to_files(results: Dict[str, Any]):
    """Save results to JSON and text files."""
    
    # Save to JSON
    with open("extracted_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save to text report
    with open("extracted_benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EXTRACTED BENCHMARK RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Questions: {results['total_questions']}\n")
        f.write(f"Correct Answers: {results['summary']['correct_answers']}\n")
        f.write(f"Incorrect Answers: {results['summary']['incorrect_answers']}\n")
        f.write(f"Accuracy: {results['summary']['accuracy']:.3f} ({results['summary']['accuracy']*100:.1f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("QUESTION-BY-QUESTION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for q in results["questions"]:
            f.write(f"Question {q['question_number']}:\n")
            f.write(f"  Log File: {q['log_file']}\n")
            f.write(f"  Question: {q['question'][:100]}...\n")
            f.write(f"  Extracted Answer: {q.get('extracted_answer', 'N/A')}\n")
            f.write(f"  Correct Answer: {q.get('correct_answer', 'N/A')}\n")
            f.write(f"  Selected Agent: {q.get('selected_agent', 'N/A')}\n")
            f.write(f"  Correct: {'✅' if q.get('is_correct', False) else '❌'}\n")
            f.write(f"  Success: {q.get('success', False)}\n")
            f.write("-" * 40 + "\n\n")
    
    print(f"✅ Results saved to extracted_benchmark_results.json")
    print(f"✅ Report saved to extracted_benchmark_report.txt")

def main():
    """Main function to extract and save results."""
    print("�� Extracting results from coordination logs...")
    
    results = extract_results_from_coordination_logs()
    
    print(f"\n📊 Summary:")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Correct: {results['summary']['correct_answers']}")
    print(f"Incorrect: {results['summary']['incorrect_answers']}")
    print(f"Accuracy: {results['summary']['accuracy']:.3f} ({results['summary']['accuracy']*100:.1f}%)")
    
    save_results_to_files(results)

if __name__ == "__main__":
    main()