import json
import sys
from datetime import datetime

def generate_report(results_file):
    """Generate a simple markdown report from benchmark results."""
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Generate markdown report
    report = f"""# Benchmark Results Report

## Overview
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Questions**: {data.get('total', 0)}
- **Dataset**: MMLU-Pro

## Results Summary

| Model | Correct | Total | Accuracy |
|-------|---------|-------|----------|
"""
    
    # Add model results
    for model, stats in data.get('models', {}).items():
        correct = stats.get('correct', 0)
        total = stats.get('total', 0)
        accuracy = (correct / total * 100) if total > 0 else 0
        report += f"| {model} | {correct} | {total} | {accuracy:.1f}% |\n"
    
    report += "\n## Details\n"
    
    # Add question details if available
    if 'responses' in data:
        report += "\n### Question Details\n\n"
        for i, response in enumerate(data['responses'][:10]):  # Show first 10
            report += f"**Question {i+1}**: {response.get('question', 'N/A')[:100]}...\n"
            report += f"- Response: {response.get('response', 'N/A')[:100]}...\n\n"
    
    # Save report
    output_file = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Report generated: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_report_generator.py <results_file.json>")
        sys.exit(1)
    
    generate_report(sys.argv[1])