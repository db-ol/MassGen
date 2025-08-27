import json
import csv
import pandas as pd
from datetime import datetime

def convert_benchmark_to_csv(json_file, csv_file):
    """Convert benchmark results JSON to CSV format."""
    
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract data for CSV
    csv_data = []
    
    # Process single models
    for model_name, model_data in data['single_models'].items():
        for response in model_data['responses']:
            csv_data.append({
                'Model': model_name,
                'Question_ID': response.get('question_id', 'N/A'),
                'Question': response.get('question', 'N/A')[:100] + '...',
                'Response': response.get('response', 'N/A')[:200] + '...',
                'Extracted_Answer': response.get('extracted_answer', 'N/A'),
                'Correct_Answer': response.get('correct_answer', 'N/A'),
                'Is_Correct': response.get('is_correct', False),
                'Response_Time': response.get('response_time', 0),
                'Accuracy': model_data.get('accuracy', 0)
            })
    
    # Process multi-agent
    for response in data['multi_agent']['responses']:
        csv_data.append({
            'Model': 'multi_agent',
            'Question_ID': response.get('question_id', 'N/A'),
            'Question': response.get('question', 'N/A')[:100] + '...',
            'Response': response.get('response', 'N/A')[:200] + '...',
            'Extracted_Answer': response.get('extracted_answer', 'N/A'),
            'Correct_Answer': response.get('correct_answer', 'N/A'),
            'Is_Correct': response.get('is_correct', False),
            'Response_Time': response.get('response_time', 0),
            'Accuracy': data['multi_agent'].get('accuracy', 0)
        })
    
    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
    
    print(f"✅ CSV file created: {csv_file}")

def create_accuracy_table(json_file, table_file):
    """Create accuracy summary table similar to the YAML format."""
    
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create table content
    table_content = f"""# MMLU-PRO BENCHMARK RESULTS TABLE
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Dataset: MMLU-Pro
# Total Questions: {data['summary']['total_questions']}

================================================================================
ACCURACY SUMMARY TABLE
================================================================================

Question_ID | Correct | GPT-5 Nano | GLM-4.5 | Grok-4 | Gemini 2.5 Flash | Multi-Agent | Multi-Agent Selected
------------|---------|------------|---------|--------|------------------|-------------|---------------------
"""
    
    # Get all question IDs from multi-agent responses
    question_ids = []
    for response in data['multi_agent']['responses']:
        qid = response.get('question_id', 'N/A')
        if qid not in question_ids:
            question_ids.append(qid)
    
    # Create table rows
    for i, qid in enumerate(question_ids, 1):
        # Find responses for this question
        gpt_response = None
        glm_response = None
        grok_response = None
        gemini_response = None
        multi_response = None
        
        # Find single model responses
        for model_name, model_data in data['single_models'].items():
            for response in model_data['responses']:
                if response.get('question_id') == qid:
                    if model_name == 'gpt5nano':
                        gpt_response = response
                    elif model_name == 'glm45':
                        glm_response = response
                    elif model_name == 'grok4':
                        grok_response = response
                    elif model_name == 'gemini25flash':
                        gemini_response = response
        
        # Find multi-agent response
        for response in data['multi_agent']['responses']:
            if response.get('question_id') == qid:
                multi_response = response
                break
        
        # Get correct answer
        correct_answer = multi_response.get('correct_answer', 'N/A') if multi_response else 'N/A'
        
        # Format responses
        gpt_result = "✅" if gpt_response and gpt_response.get('is_correct') else "❌"
        gpt_answer = f"({gpt_response.get('extracted_answer', 'N/A')})" if gpt_response else "(N/A)"
        
        glm_result = "✅" if glm_response and glm_response.get('is_correct') else "❌"
        glm_answer = f"({glm_response.get('extracted_answer', 'N/A')})" if glm_response else "(N/A)"
        
        grok_result = "✅" if grok_response and grok_response.get('is_correct') else "❌"
        grok_answer = f"({grok_response.get('extracted_answer', 'N/A')})" if grok_response else "(N/A)"
        
        gemini_result = "✅" if gemini_response and gemini_response.get('is_correct') else "❌"
        gemini_answer = f"({gemini_response.get('extracted_answer', 'N/A')})" if gemini_response else "(N/A)"
        
        multi_result = "✅" if multi_response and multi_response.get('is_correct') else "❌"
        multi_answer = f"({multi_response.get('extracted_answer', 'N/A')})" if multi_response else "(N/A)"
        
        # Determine multi-agent selection (simplified)
        multi_selected = "Consensus"
        if multi_response and multi_response.get('extracted_answer'):
            # Check if all models agreed
            answers = []
            if gpt_response: answers.append(gpt_response.get('extracted_answer'))
            if glm_response: answers.append(glm_response.get('extracted_answer'))
            if grok_response: answers.append(grok_response.get('extracted_answer'))
            if gemini_response: answers.append(gemini_response.get('extracted_answer'))
            
            if len(set(answers)) == 1 and answers[0] == multi_response.get('extracted_answer'):
                multi_selected = "Consensus"
            else:
                multi_selected = "Majority"
        
        # Add row to table (removed Topic column)
        table_content += f"{qid:<11} | {correct_answer:<7} | {gpt_result} {gpt_answer:<8} | {glm_result} {glm_answer:<8} | {grok_result} {grok_answer:<8} | {gemini_result} {gemini_answer:<8} | {multi_result} {multi_answer:<8} | {multi_selected}\n"
    
    # Add summary statistics
    table_content += f"""
================================================================================
SUMMARY STATISTICS
================================================================================

Model/System        | Correct | Total | Accuracy | Response Time
-------------------|---------|-------|----------|---------------
"""
    
    for model_name, model_data in data['single_models'].items():
        correct = model_data.get('correct', 0)
        total = model_data.get('total', 0)
        accuracy = model_data.get('accuracy', 0)
        response_time = model_data.get('response_time', 0)
        table_content += f"{model_name:<17} | {correct:<7} | {total:<5} | {accuracy:<8.3f} | {response_time:<.2f}s\n"
    
    # Add multi-agent stats
    multi_data = data['multi_agent']
    table_content += f"{'Multi-Agent':<17} | {multi_data.get('correct', 0):<7} | {multi_data.get('total', 0):<5} | {multi_data.get('accuracy', 0):<8.3f} | {multi_data.get('response_time', 0):<.2f}s\n"
    
    # Add notes
    table_content += f"""
================================================================================
NOTES
================================================================================

- ✅ = Correct answer
- ❌ = Wrong answer
- N/A = No response or timeout
- Consensus = All agents provided the same answer
- Majority = Multi-agent chose the most common answer
- Response Time = Total time for all questions
- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Write table to file
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write(table_content)
    
    print(f"✅ Accuracy table created: {table_file}")

def main():
    json_file = "benchmark_results.json"
    csv_file = "benchmark_results.csv"
    table_file = "benchmark_accuracy_table.txt"
    
    print("🔄 Converting benchmark results...")
    
    # Convert to CSV
    convert_benchmark_to_csv(json_file, csv_file)
    
    # Create accuracy table
    create_accuracy_table(json_file, table_file)
    
    print(f"\n✅ Conversion completed!")
    print(f"📊 CSV file: {csv_file}")
    print(f"📋 Accuracy table: {table_file}")

if __name__ == "__main__":
    main()