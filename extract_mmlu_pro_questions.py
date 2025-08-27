import json
import requests
from datasets import load_dataset
from typing import List, Dict, Any
import pandas as pd

def extract_question_ids_from_benchmark(benchmark_file: str) -> List[int]:
    """
    Extract all unique question IDs from the benchmark results file.
    """
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    question_ids = set()
    
    # Extract from single models
    for model_name, model_data in data.get('single_models', {}).items():
        for response in model_data.get('responses', []):
            question_ids.add(response.get('question_id'))
    
    # Extract from multi-agent
    for response in data.get('multi_agent', {}).get('responses', []):
        question_ids.add(response.get('question_id'))
    
    return sorted(list(question_ids))

def load_mmlu_pro_dataset():
    """
    Load the MMLU-Pro dataset from Hugging Face.
    """
    try:
        # Load the dataset
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def get_full_questions_by_ids(dataset, question_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Get full questions from the dataset by question IDs.
    """
    full_questions = []
    
    # Get the test split (or whichever split contains the questions)
    test_data = dataset.get('test', dataset.get('validation', dataset.get('train')))
    
    # Create a mapping of question_id to full question data
    id_to_question = {}
    for item in test_data:
        if 'question_id' in item:
            id_to_question[item['question_id']] = item
        elif 'id' in item:
            id_to_question[item['id']] = item
    
    # Extract questions by ID
    for qid in question_ids:
        if qid in id_to_question:
            full_questions.append({
                'question_id': qid,
                'full_question': id_to_question[qid]
            })
        else:
            print(f"Warning: Question ID {qid} not found in dataset")
            full_questions.append({
                'question_id': qid,
                'full_question': None
            })
    
    return full_questions

def create_comprehensive_report(benchmark_file: str, output_file: str = "mmlu_pro_full_questions.json"):
    """
    Create a comprehensive report with full questions and benchmark results.
    """
    print("Loading benchmark results...")
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    
    print("Extracting question IDs...")
    question_ids = extract_question_ids_from_benchmark(benchmark_file)
    print(f"Found {len(question_ids)} unique question IDs: {question_ids}")
    
    print("Loading MMLU-Pro dataset...")
    dataset = load_mmlu_pro_dataset()
    if dataset is None:
        print("Failed to load dataset. Please check your HF_API_KEY and internet connection.")
        return
    
    print("Extracting full questions...")
    full_questions = get_full_questions_by_ids(dataset, question_ids)
    
    # Create comprehensive report
    report = {
        "metadata": {
            "total_questions": len(question_ids),
            "question_ids": question_ids,
            "dataset_source": "TIGER-Lab/MMLU-Pro",
            "benchmark_file": benchmark_file
        },
        "questions": []
    }
    
    # Combine benchmark results with full questions
    for qid in question_ids:
        # Find benchmark results for this question
        benchmark_results = {
            "single_models": {},
            "multi_agent": None
        }
        
        # Extract single model results
        for model_name, model_data in benchmark_data.get('single_models', {}).items():
            for response in model_data.get('responses', []):
                if response.get('question_id') == qid:
                    benchmark_results["single_models"][model_name] = {
                        "response": response.get('response'),
                        "extracted_answer": response.get('extracted_answer'),
                        "is_correct": response.get('is_correct'),
                        "response_time": response.get('response_time')
                    }
                    break
        
        # Extract multi-agent results
        for response in benchmark_data.get('multi_agent', {}).get('responses', []):
            if response.get('question_id') == qid:
                benchmark_results["multi_agent"] = {
                    "response": response.get('response'),
                    "extracted_answer": response.get('extracted_answer'),
                    "is_correct": response.get('is_correct'),
                    "response_time": response.get('response_time'),
                    "judge_evaluation": response.get('judge_evaluation')
                }
                break
        
        # Find full question data
        full_question_data = None
        for fq in full_questions:
            if fq['question_id'] == qid:
                full_question_data = fq['full_question']
                break
        
        # Combine into final entry
        question_entry = {
            "question_id": qid,
            "full_question": full_question_data,
            "benchmark_results": benchmark_results
        }
        
        report["questions"].append(question_entry)
    
    # Save comprehensive report
    print(f"Saving comprehensive report to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Create a simplified CSV version
    csv_data = []
    for q in report["questions"]:
        full_q = q["full_question"]
        if full_q:
            row = {
                "question_id": q["question_id"],
                "question": full_q.get("question", ""),
                "options": full_q.get("options", []),
                "answer": full_q.get("answer", ""),
                "answer_index": full_q.get("answer_index", ""),
                "category": full_q.get("category", ""),
                "src": full_q.get("src", "")
            }
            
            # Add benchmark results
            for model_name, results in q["benchmark_results"]["single_models"].items():
                row[f"{model_name}_answer"] = results.get("extracted_answer", "")
                row[f"{model_name}_correct"] = results.get("is_correct", False)
                row[f"{model_name}_time"] = results.get("response_time", 0)
            
            # Debug: Print the actual value to see what's happening
            multi_agent_value = q["benchmark_results"]["multi_agent"]
            print(f"DEBUG: multi_agent_value = {multi_agent_value}, type = {type(multi_agent_value)}")
            
            try:
                if q["benchmark_results"]["multi_agent"] is not None and isinstance(q["benchmark_results"]["multi_agent"], dict):
                    ma_results = q["benchmark_results"]["multi_agent"]
                    # ma_results is already the individual response object, not a container
                    row["multi_agent_answer"] = ma_results.get("extracted_answer", "")
                    row["multi_agent_correct"] = ma_results.get("is_correct", False)
                    row["multi_agent_time"] = ma_results.get("response_time", 0)
                    row["judge_evaluation"] = ma_results.get("judge_evaluation", {}).get("is_correct", False)
                else:
                    row["multi_agent_answer"] = ""
                    row["multi_agent_correct"] = False
                    row["multi_agent_time"] = 0
                    row["judge_evaluation"] = False
            except Exception as e:
                print(f"ERROR processing multi_agent for question {q['question_id']}: {e}")
                row["multi_agent_answer"] = ""
                row["multi_agent_correct"] = False
                row["multi_agent_time"] = 0
                row["judge_evaluation"] = False
            
            csv_data.append(row)
    
    # Save CSV
    csv_file = output_file.replace('.json', '.csv')
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Saved CSV report to {csv_file}")
    
    print(f"Report generation complete!")
    print(f"JSON file: {output_file}")
    print(f"CSV file: {csv_file}")
    
    return report

def create_summary_table(report: Dict[str, Any], output_file: str = "mmlu_pro_summary.md"):
    """
    Create a markdown summary table of the questions and results.
    """
    markdown_content = """# MMLU-Pro Benchmark Questions Summary

## Dataset Information
- **Source**: TIGER-Lab/MMLU-Pro
- **Total Questions**: {total_questions}
- **Question IDs**: {question_ids}

## Questions and Results

| Question ID | Question | Correct Answer | Options | Category | GPT-5 Nano | GLM-4.5 | Grok-4 | Gemini 2.5 Flash | Multi-Agent | Judge Correct |
|-------------|----------|----------------|---------|----------|------------|---------|--------|------------------|-------------|---------------|
""".format(
        total_questions=report["metadata"]["total_questions"],
        question_ids=", ".join(map(str, report["metadata"]["question_ids"]))
    )
    
    for q in report["questions"]:
        full_q = q["full_question"]
        if not full_q:
            continue
            
        question_text = full_q.get("question", "")[:100] + "..." if len(full_q.get("question", "")) > 100 else full_q.get("question", "")
        options = full_q.get("options", [])
        options_str = " | ".join([f"{chr(65+i)}: {opt[:30]}..." for i, opt in enumerate(options)]) if options else ""
        
        # Get benchmark results
        benchmark = q["benchmark_results"]
        
        gpt5nano = benchmark["single_models"].get("gpt5nano", {})
        glm45 = benchmark["single_models"].get("glm45", {})
        grok4 = benchmark["single_models"].get("grok4", {})
        gemini25flash = benchmark["single_models"].get("gemini25flash", {})
        multi_agent = benchmark.get("multi_agent") or {}
        
        row = f"| {q['question_id']} | {question_text} | {full_q.get('answer', '')} | {options_str} | {full_q.get('category', '')} | "
        row += f"{gpt5nano.get('extracted_answer', 'N/A')} ({'✅' if gpt5nano.get('is_correct') else '❌'}) | "
        row += f"{glm45.get('extracted_answer', 'N/A')} ({'✅' if glm45.get('is_correct') else '❌'}) | "
        row += f"{grok4.get('extracted_answer', 'N/A')} ({'✅' if grok4.get('is_correct') else '❌'}) | "
        row += f"{gemini25flash.get('extracted_answer', 'N/A')} ({'✅' if gemini25flash.get('is_correct') else '❌'}) | "
        row += f"{multi_agent.get('extracted_answer', 'N/A')} ({'✅' if multi_agent.get('is_correct') else '❌'}) | "
        row += f"{'✅' if multi_agent.get('judge_evaluation', {}).get('is_correct') else '❌'} |"
        
        markdown_content += row + "\n"
    
    markdown_content += "\n## Notes\n- ✅ = Correct answer\n- ❌ = Wrong answer\n- N/A = No response or timeout\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Summary table saved to {output_file}")

if __name__ == "__main__":
    # Set your HF API key if not already set
    import os
    if not os.getenv("HF_API_KEY"):
        print("Warning: HF_API_KEY not set. Please set it to access the dataset.")
        print("You can set it with: export HF_API_KEY=your_key_here")
    
    # Generate the comprehensive report
    report = create_comprehensive_report("benchmark_results.json")
    
    if report:
        # Create summary table
        create_summary_table(report)
        
        print("\n" + "="*50)
        print("SCRIPT COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Generated files:")
        print("1. mmlu_pro_full_questions.json - Complete data with full questions")
        print("2. mmlu_pro_full_questions.csv - CSV format for easy analysis")
        print("3. mmlu_pro_summary.md - Markdown summary table")
        print("\nYou can now analyze the full questions and compare with benchmark results.")