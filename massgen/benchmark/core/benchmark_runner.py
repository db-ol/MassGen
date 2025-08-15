"""
Benchmark Runner for HLE Lite Multiple Choice Questions
Single models benchmarked directly, multi-agent uses existing CLI system
"""

import asyncio
import json
import time
import subprocess
import sys
from typing import List, Dict, Any
from pathlib import Path
import os

from massgen.cli import create_backend, create_agents_from_config
from .load_dataset import HLEDatasetLoader

class HLEBenchmarkRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.results = {}
        self.logs = []
        self.current_questions = []
        
    def _log(self, message: str):
        """Add message to logs and print to console."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
        
    def _save_logs(self):
        """Save logs to benchmark.txt file."""
        try:
            with open("agent_outputs/benchmark.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(self.logs))
            self._log("💾 Logs saved to agent_outputs/benchmark.txt")
        except Exception as e:
            print(f"❌ Failed to save logs: {e}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load benchmark configuration."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_hle_dataset(self, token: str) -> List[Dict]:
        """Load and preprocess HLE dataset for benchmarking."""
        self._log("📚 Loading HLE dataset...")
        
        # Use the shared dataset loader with only multiple choice questions
        loader = HLEDatasetLoader(token)
        questions = loader.load_multiple_choice_only()  # Only load multiple choice questions
        
        # Limit to max_questions if specified
        max_q = self.config['benchmark'].get('max_questions', len(questions))
        limited_questions = questions[:max_q]
        
        self._log(f"📊 Loaded {len(limited_questions)} multiple choice questions")
        
        return limited_questions
    
    async def benchmark_single_model(self, model_config: Dict, questions: List[Dict]) -> Dict[str, Any]:
        """Benchmark a single model on the questions."""
        self._log(f"🧪 Benchmarking {model_config['name']}...")
        
        results = {
            'model': model_config['name'],
            'correct': 0,
            'total': len(questions),
            'responses': [],
            'confidence_scores': [],
            'response_time': 0.0,
            'calibration_error': 0.0  # Add missing calibration_error
        }
        
        total_start_time = time.time()
        
        for i, question in enumerate(questions):
            self._log(f"  Question {i+1}/{len(questions)}")
            
            try:
                # Use the formatted question
                question_text = question['formatted_question']
                
                # Create a simple config for single model
                single_config = {
                    "agent": {
                        "id": model_config['name'],
                        "backend": model_config['backend'],
                        "system_message": model_config.get('system_message', 'You are a helpful AI assistant. Answer multiple choice questions accurately and provide confidence scores. **Always end your response with "The answer is: X" where X is the letter (A, B, C, D, or E) of your chosen option.**')
                    }
                }
                
                # Create agent using the existing CLI infrastructure
                agents = create_agents_from_config(single_config)
                agent = next(iter(agents.values()))
                
                # Get response using the agent's chat method
                messages = [{"role": "user", "content": question_text}]
                response_content = ""
                
                self._log(f"    Getting response from {model_config['name']}...")
                
                start_time = time.time()  # Track individual question time
                try:
                    async for chunk in agent.chat(messages):
                        if chunk.type == "content" and chunk.content:
                            response_content += chunk.content
                        elif chunk.type == "error":
                            self._log(f"    ❌ Error chunk received: {chunk.error}")
                            raise Exception(f"Model error: {chunk.error}")
                        elif chunk.type == "done":
                            break
                except Exception as chat_error:
                    self._log(f"    ❌ Chat stream error: {chat_error}")
                    raise chat_error
                
                question_time = time.time() - start_time
                self._log(f"    Question time: {question_time:.2f}s")
                
                self._log(f"    Response length: {len(response_content)} characters")
                
                if not response_content:
                    self._log(f"    Warning: No response generated")
                    response_content = "No response generated"
                
                # Extract confidence score
                confidence = self._extract_confidence(response_content)
                
                # Extract answer and check correctness (FIXED)
                extracted_answer = self._extract_answer_from_response(response_content)
                correct_answer = question['answer']
                is_correct = extracted_answer == correct_answer
                
                results['responses'].append({
                    'question_id': question['id'],
                    'question': question['original_question'],
                    'correct_answer': question['answer'],
                    'extracted_answer': extracted_answer,  # Add extracted answer
                    'response': response_content,
                    'confidence': confidence,
                    'correct': is_correct,
                    'response_time': question_time  # Add individual response time
                })
                
                if is_correct:
                    results['correct'] += 1
                
                # Log with the same format as multi-agent
                self._log(f"    Answer: {extracted_answer} (Correct: {correct_answer}) {'✅' if is_correct else '❌'}")
                
                results['confidence_scores'].append(confidence)
                
            except Exception as e:
                self._log(f"    ❌ Error: {e}")
                import traceback
                error_trace = traceback.format_exc()
                self._log(f"    Traceback: {error_trace}")
                
                results['responses'].append({
                    'question_id': question['id'],
                    'question': question['original_question'],
                    'correct_answer': question['answer'],
                    'error': str(e),
                    'response': "Error occurred",
                    'confidence': 0.0,
                    'correct': False,
                    'response_time': 0.0
                })
        
        # Calculate total response time
        results['response_time'] = time.time() - total_start_time
        
        # Calculate metrics (only for successful responses)
        if results['confidence_scores']:
            results['accuracy'] = results['correct'] / results['total']
            results['calibration_error'] = self._calculate_calibration_error(results)
        else:
            results['accuracy'] = 0.0
            results['calibration_error'] = 0.0
        
        self._log(f"  Final accuracy: {results['accuracy']:.3f}")
        self._log(f"  Total response time: {results['response_time']:.2f}s")
        
        return results
    
    def _resolve_config_path(self, ma_config_path: str) -> Path:
        """Resolve multi-agent config path."""
        if not os.path.isabs(ma_config_path):
            # Try relative to current working directory
            current_dir = Path.cwd()
            resolved_path = current_dir / ma_config_path
            
            if not resolved_path.exists():
                # Try relative to benchmark config directory
                benchmark_dir = Path(self.config_path).parent
                resolved_path = benchmark_dir / ma_config_path
                
                if not resolved_path.exists():
                    # Try relative to massgen configs directory
                    massgen_configs = Path(__file__).parent.parent.parent.parent / "massgen" / "configs"
                    resolved_path = massgen_configs / Path(ma_config_path).name
        
        return resolved_path

    def benchmark_multi_agent_cli(self, questions: List[Dict]) -> Dict[str, Any]:
        """Benchmark multi-agent system using CLI."""
        self._log(" Benchmarking Multi-Agent System using CLI...")
        
        # Get multi-agent config path
        ma_config_path = self.config['benchmark']['multi_agent']['config_file']
        resolved_path = self._resolve_config_path(ma_config_path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Multi-agent config not found: {ma_config_path}")
        
        self._log(f"  Using multi-agent config: {resolved_path}")
        
        results = {
            'model': 'Multi-Agent System',
            'correct': 0,
            'total': len(questions),
            'responses': [],
            'response_time': 0.0,
            'calibration_error': 0.0  # Add missing calibration_error
        }
        
        total_start_time = time.time()
        
        for i, question in enumerate(questions):
            self._log(f"  Question {i+1}/{len(questions)}")
            
            try:
                # Use the formatted question
                question_text = question['formatted_question']
                
                # Run CLI command
                start_time = time.time()
                cmd = [
                    sys.executable, "-m", "massgen.cli",
                    "--config", str(resolved_path),
                    "--no-display",
                    question_text
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',  # Explicitly set encoding
                    errors='ignore',   # Ignore encoding errors
                    timeout=180,  # 3 minute timeout
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONLEGACYWINDOWSSTDIO': 'utf-8'}
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if result.returncode != 0:
                    self._log(f"    ❌ CLI command failed: {result.stderr}")
                    results['responses'].append({
                        'question_id': question['id'],
                        'error': f"CLI failed: {result.stderr}",
                        'is_correct': False,
                        'response_time': response_time
                    })
                    continue
                
                # Extract final response
                output = result.stdout
                if output is None:
                    output = ""
                extracted_answer = self._extract_final_response(output)
                
                # Check correctness
                correct_answer = question['answer']
                is_correct = extracted_answer == correct_answer
                
                if is_correct:
                    results['correct'] += 1
                
                # Store results
                results['responses'].append({
                    'question_id': question['id'],
                    'question': question['original_question'],
                    'response': output,
                    'extracted_answer': extracted_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'response_time': response_time
                })
                
                self._log(f"    Answer: {extracted_answer} (Correct: {correct_answer}) {'✅' if is_correct else '❌'}")
                
            except subprocess.TimeoutExpired:
                self._log(f"    ❌ Timeout after 180 seconds")
                results['responses'].append({
                    'question_id': question['id'],
                    'error': 'Timeout',
                    'is_correct': False,
                    'response_time': 180.0
                })
            except Exception as e:
                self._log(f"    ❌ Error: {e}")
                results['responses'].append({
                    'question_id': question['id'],
                    'error': str(e),
                    'is_correct': False,
                    'response_time': 0.0
                })
        
        # Calculate total response time
        results['response_time'] = time.time() - total_start_time
        
        # Calculate accuracy
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
        
        self._log(f"✅ Multi-Agent System completed: {results['correct']}/{results['total']} correct ({results['accuracy']:.3f})")
        return results

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response."""
        # Simple confidence extraction - look for patterns like "confidence: 0.8" or "95%"
        import re
        
        # Look for percentage patterns
        percent_match = re.search(r'(\d+)%', response, re.IGNORECASE)
        if percent_match:
            return float(percent_match.group(1)) / 100.0
        
        # Look for decimal patterns
        decimal_match = re.search(r'confidence[:\s]*(\d*\.?\d+)', response, re.IGNORECASE)
        if decimal_match:
            confidence = float(decimal_match.group(1))
            if confidence > 1.0:  # If it's a percentage (e.g., 95.0)
                return confidence / 100.0
            return confidence
        
        # Default confidence
        return 0.7

    def _check_answer_correctness(self, response: str, question: Dict) -> bool:
        """Check if the response contains the correct answer."""
        if response == "No response generated" or response == "Error occurred":
            return False
            
        correct_answer = question['answer']
        
        # Look for answer patterns in the response
        response_lower = response.lower()
        
        # Check for direct answer mentions
        if f"answer: {correct_answer.lower()}" in response_lower:
            return True
        if f"option {correct_answer.lower()}" in response_lower:
            return True
        if f"choice {correct_answer.lower()}" in response_lower:
            return True
        
        # Check for answer at the end
        if response_lower.strip().endswith(correct_answer.lower()):
            return True
        
        # Check for answer in parentheses
        if f"({correct_answer.lower()})" in response_lower:
            return True
        
        return False

    def _calculate_calibration_error(self, results: Dict) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        if not results.get('confidence_scores'):
            return 0.0
        
        # Simple ECE calculation
        confidence_scores = results['confidence_scores']
        correct_predictions = [1 if r['correct'] else 0 for r in results['responses']]
        
        # Calculate average confidence vs accuracy
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        accuracy = sum(correct_predictions) / len(correct_predictions)
        
        return abs(avg_confidence - accuracy)

    async def run_benchmark(self, token: str) -> Dict[str, Any]:
        """Run the complete benchmark."""
        self._log("🚀 Starting HLE Lite Benchmark...")
        
        try:
            # Load dataset
            questions = self.load_hle_dataset(token)
            self.current_questions = questions
            
            # Initialize results
            self.results = {
                'single_models': {},
                'multi_agent': {},
                'summary': {}
            }
            
            # Benchmark single models
            for model_config in self.config['benchmark']['single_models']:
                self._log(f"🧪 Benchmarking single model: {model_config['name']}")
                results = await self.benchmark_single_model(model_config, questions)
                self.results['single_models'][model_config['name']] = results
            
            # Benchmark multi-agent system
            self._log(" Benchmarking multi-agent system...")
            ma_results = self.benchmark_multi_agent_cli(questions)
            self.results['multi_agent'] = ma_results
            
            # Generate summary
            self._generate_summary()
            
            # Print results
            self.print_results_table()
            
            # Save detailed results
            with open("benchmark_results.json", "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            self._log("✅ Benchmark completed successfully!")
            return self.results
            
        except Exception as e:
            self._log(f"❌ Benchmark error: {e}")
            raise
        finally:
            self._save_logs()
    
    def _generate_summary(self):
        """Generate summary statistics."""
        summary = {
            'total_questions': len(self.current_questions),
            'single_models': {},
            'multi_agent': {}
        }
        
        # Single models summary
        for model_name, results in self.results['single_models'].items():
            accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
            summary['single_models'][model_name] = {
                'accuracy': accuracy,
                'calibration_error': results['calibration_error'],
                'response_time': results['response_time']
            }
        
        # Multi-agent summary
        if self.results['multi_agent']:
            ma_results = self.results['multi_agent']
            accuracy = ma_results['correct'] / ma_results['total'] if ma_results['total'] > 0 else 0
            summary['multi_agent'] = {
                'accuracy': accuracy,
                'calibration_error': ma_results.get('calibration_error', 0.0),
                'response_time': ma_results['response_time']
            }
        
        self.results['summary'] = summary
    
    def print_results_table(self):
        """Print and save results table."""
        # Get multi-agent components
        ma_components = self._get_multi_agent_components()
        
        # Create table
        table_lines = []
        table_lines.append("=" * 80)
        table_lines.append("HLE LITE BENCHMARK RESULTS")
        table_lines.append("=" * 80)
        table_lines.append(f"{'Model/System':<25} {'Accuracy':<10} {'Calibration':<12} {'Response Time':<15}")
        table_lines.append("-" * 80)
        
        # Single models
        for model_name, results in self.results['single_models'].items():
            accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
            table_lines.append(f"{model_name:<25} {accuracy:.3f}      {results['calibration_error']:.3f}        {results['response_time']:.2f}s")
        
        # Multi-agent
        if self.results['multi_agent']:
            ma_results = self.results['multi_agent']
            accuracy = ma_results['correct'] / ma_results['total'] if ma_results['total'] > 0 else 0
            table_lines.append(f"{'Multi-Agent':<25} {accuracy:.3f}      {ma_results.get('calibration_error', 0.0):.3f}        {ma_results['response_time']:.2f}s")
            table_lines.append(f"  Components: {ma_components}")
        
        table_lines.append("=" * 80)
        
        # Print to console
        for line in table_lines:
            print(line)
        
        # Save to file (overwrite)
        with open("benchmark.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(table_lines))
        
        self._log("📊 Results table saved to benchmark.txt")
    
    def _get_multi_agent_components(self) -> str:
        """Get multi-agent system components."""
        try:
            ma_config_path = self.config['benchmark']['multi_agent']['config_file']
            resolved_path = self._resolve_config_path(ma_config_path)
            
            if resolved_path.exists():
                import yaml
                with open(resolved_path, 'r') as f:
                    ma_config = yaml.safe_load(f)
                
                agent_names = []
                for agent in ma_config.get('agents', []):
                    backend_type = agent.get('backend', {}).get('type', 'unknown')
                    model = agent.get('backend', {}).get('model', 'unknown')
                    agent_names.append(f"{backend_type}-{model}")
                
                return ", ".join(agent_names)
        except Exception as e:
            self._log(f"Warning: Could not get multi-agent components: {e}")
        
        return "Unknown"

    def _print_detailed_analysis(self):
        """Print detailed question-by-question analysis."""
        print("\n" + "=" * 80)
        print(" 📝 DETAILED QUESTION ANALYSIS")
        print("=" * 80)
        
        for i, question in enumerate(self.current_questions):
            print(f"\nQuestion {i+1}:")
            print(f"  Question: {question['original_question'][:100]}...")
            print(f"  True Answer: {question['answer']}")
            print(f"  Category: {question.get('category', 'Unknown')}")
            
            # Single model responses
            for result in self.results.get('single_models', []):
                if i < len(result['responses']):
                    response = result['responses'][i]
                    print(f"  {result['model']}:")
                    print(f"    Response: {response['response'][:100]}...")
                    print(f"    Confidence: {response['confidence']:.2f}")
                    print(f"    Correct: {'✅' if response['correct'] else '❌'}")
            
            # Multi-agent response
            ma_result = self.results.get('multi_agent')
            if ma_result and i < len(ma_result['responses']):
                ma_response = ma_result['responses'][i]
                print(f"  Multi-Agent (CLI):")
                if 'error' in ma_response:
                    print(f"    Error: {ma_response['error']}")
                else:
                    print(f"    Response: {ma_response['response'][:100]}...")
                    print(f"    Correct: {'✅' if ma_response['is_correct'] else '❌'}")

    def _extract_answer_from_response(self, response: str) -> str:
        """Extract answer from a single model response with multiple fallback patterns."""
        if not response:
            return "No answer found"
        
        # Primary pattern: "The answer is: X"
        if "The answer is:" in response:
            answer_part = response.split("The answer is:")[1].strip()
            import re
            match = re.search(r'\b([A-E])\b', answer_part)
            if match:
                return match.group(1)
        
        # Fallback patterns for single models that don't follow the format
        
        # Pattern 1: Look for "Answer: X" or "Option X" or "Choice X"
        import re
        answer_patterns = [
            r'[Aa]nswer:\s*([A-E])',
            r'[Oo]ption\s*([A-E])',
            r'[Cc]hoice\s*([A-E])',
            r'[Ss]elect\s*([A-E])',
            r'[Cc]hoose\s*([A-E])',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # Pattern 2: Look for "X." at the beginning of lines (common in explanations)
        lines = response.split('\n')
        for line in lines:
            match = re.search(r'^([A-E])\.\s*', line.strip())
            if match:
                return match.group(1)
        
        # Pattern 3: Look for "**X. Description**" format
        match = re.search(r'\*\*([A-E])\.\s*([^*]+)\*\*', response)
        if match:
            return match.group(1)
        
        # Pattern 4: Look for LaTeX boxed answers: \boxed{X}
        match = re.search(r'\\boxed\{([A-E])\}', response)
        if match:
            return match.group(1)
        
        # Pattern 5: Look for standalone letters that are likely answers
        letters = re.findall(r'\b([A-E])\b', response)
        if letters:
            # Take the last occurrence as it's likely the final answer
            return letters[-1]
        
        # Pattern 6: Look for specific content-based answers
        if "Weak Quality Addition" in response or "Weak Quality Addition (E)" in response:
            return "E"
        if "Weak Non-Sadism" in response or "Weak Non-Sadism (D)" in response:
            return "D"
        if "Egalitarian Dominance" in response or "Egalitarian Dominance (A)" in response:
            return "A"
        if "General Non-Extreme Priority" in response or "General Non-Extreme Priority (B)" in response:
            return "B"
        if "Non-Elitism" in response or "Non-Elitism (C)" in response:
            return "C"
        
        return "No answer found"

    def _extract_final_response(self, output: str) -> str:
        """Extract the final response from multi-agent CLI output."""
        # Handle None or empty output
        if not output:
            return "No answer found"
        
        lines = output.split('\n')
        
        # Primary patterns: "The answer is:" or "The final answer is:"
        for i, line in enumerate(lines):
            if "The answer is:" in line or "The final answer is:" in line:
                # Extract everything after either pattern
                if "The answer is:" in line:
                    answer_part = line.split("The answer is:")[1].strip()
                else:
                    answer_part = line.split("The final answer is:")[1].strip()
                
                import re
                match = re.search(r'\b([A-E])\b', answer_part)
                if match:
                    return match.group(1)
                
                # Check next line if current line doesn't have the letter
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    match = re.search(r'\b([A-E])\b', next_line)
                    if match:
                        return match.group(1)
                
                # Look for LaTeX boxed answers: \boxed{X}
                match = re.search(r'\\boxed\{([A-E])\}', answer_part)
                if match:
                    return match.group(1)
        
        # Multi-agent specific patterns - look for the selected agent's response
        for i, line in enumerate(lines):
            if "✅ Selected by:" in line:
                selected_agent = line.split("Selected by:")[1].strip()
                
                # Look backwards to find this agent's response content
                for j in range(i-1, max(0, i-200), -1):
                    # Look for the agent's detailed response content
                    if f"[{selected_agent}]" in lines[j]:
                        # Extract the content from this line and surrounding lines
                        content_lines = []
                        for k in range(max(0, j-10), min(len(lines), j+10)):
                            if f"[{selected_agent}]" in lines[k]:
                                content_lines.append(lines[k])
                        
                        # Join the content and extract answer
                        content = "\n".join(content_lines)
                        extracted = self._extract_answer_from_response(content)
                        if extracted != "No answer found":
                            return extracted
        
        # Look for LaTeX boxed answers in any line
        for line in lines:
            if "\\boxed{" in line:
                import re
                match = re.search(r'\\boxed\{([A-E])\}', line)
                if match:
                    return match.group(1)
        
        return "No answer found"
