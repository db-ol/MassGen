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
import re

from massgen.cli import create_backend, create_agents_from_config
from .load_dataset import HLEDatasetLoader

class HLEBenchmarkRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.results = {}
        self.logs = []
        self.current_questions = []
        self.judge_agent = None
        
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
    
    async def _create_judge_agent(self):
        """Create the judge agent for evaluating responses."""
        if self.judge_agent is not None:
            return self.judge_agent
            
        judge_config = self.config['benchmark'].get('judge_model', [])
        if not judge_config:
            self._log("⚠️ No judge model configured, using default evaluation")
            return None
            
        # Use the first judge model in the list
        judge_model_config = judge_config[0]
        
        try:
            # Create judge agent using the same infrastructure as single models
            single_config = {
                "agent": {
                    "id": judge_model_config['name'],
                    "backend": judge_model_config['backend'],
                    "system_message": judge_model_config.get('system_message', '')
                }
            }
            
            agents = create_agents_from_config(single_config)
            self.judge_agent = next(iter(agents.values()))
            self._log(f"✅ Judge agent created: {judge_model_config['name']}")
            return self.judge_agent
            
        except Exception as e:
            self._log(f"❌ Failed to create judge agent: {e}")
            return None
    
    async def _evaluate_response_with_judge(self, question: Dict, response: str, correct_answer: str) -> Dict[str, Any]:
        """Use judge model to evaluate if the response is correct and extract the answer."""
        judge_agent = await self._create_judge_agent()
        
        if judge_agent is None:
            return {
                'is_correct': False,
                'judge_reasoning': 'No judge model available',
                'extracted_answer': 'No answer found',
                'confidence': 0.0
            }
        
        # Create evaluation prompt for the judge
        question_type = self.config['benchmark'].get('question_type', 'multipleChoice')
        
        if question_type == 'exactMatch':
            evaluation_prompt = f"""Evaluate this response and return ONLY a JSON object:

Question: {question['original_question']}
Correct Answer: {correct_answer}
Response: {response}

Return this exact JSON format:
{{
    "extracted_answer": "the specific answer from the response",
    "is_correct": true/false,
    "reasoning": "brief explanation"
}}"""
        else:
            evaluation_prompt = f"""Evaluate this response and return ONLY a JSON object:

Question: {question['original_question']}
Correct Answer: {correct_answer}
Response: {response}

Return this exact JSON format:
{{
    "extracted_answer": "the answer letter (A, B, C, D, E, etc.)",
    "is_correct": true/false,
    "reasoning": "brief explanation"
}}"""
        
        try:
            messages = [{"role": "user", "content": evaluation_prompt}]
            judge_response = ""
            
            async for chunk in judge_agent.chat(messages):
                if chunk.type == "content" and chunk.content:
                    judge_response += chunk.content
                elif chunk.type == "error":
                    self._log(f"    ❌ Judge evaluation error: {chunk.error}")
                    break
            
            # Try to parse JSON response
            try:
                # Extract JSON from the response (handle cases where there's extra text)
                import re
                json_match = re.search(r'\{.*\}', judge_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    evaluation_data = json.loads(json_str)
                    
                    return {
                        'is_correct': evaluation_data.get('is_correct', False),
                        'judge_reasoning': evaluation_data.get('reasoning', 'No reasoning provided'),
                        'extracted_answer': evaluation_data.get('extracted_answer', 'No answer found'),
                        'confidence': 0.8 if evaluation_data.get('is_correct', False) else 0.2
                    }
                else:
                    return {
                        'is_correct': False,
                        'judge_reasoning': 'No JSON found in judge response',
                        'extracted_answer': 'No answer found',
                        'confidence': 0.0
                    }
                    
            except json.JSONDecodeError as e:
                self._log(f"    ❌ Failed to parse judge JSON response: {e}")
                return {
                    'is_correct': False,
                    'judge_reasoning': f'JSON parse error: {e}',
                    'extracted_answer': 'No answer found',
                    'confidence': 0.0
                }
            
        except Exception as e:
            self._log(f"    ❌ Judge evaluation failed: {e}")
            return {
                'is_correct': False,
                'judge_reasoning': f'Judge error: {e}',
                'extracted_answer': 'No answer found',
                'confidence': 0.0
            }
    
    def load_hle_dataset(self, token: str) -> List[Dict]:
        """Load and preprocess HLE dataset for benchmarking."""
        self._log("📚 Loading HLE dataset...")
        
        # Get question type from config (default to multipleChoice for backward compatibility)
        question_type = self.config['benchmark'].get('question_type', 'multipleChoice')
        self._log(f"📋 Question type to benchmark: {question_type}")
        
        # Use the shared dataset loader
        loader = HLEDatasetLoader(token)
        
        if question_type == 'exactMatch':
            questions = loader.load_exact_match_only()
            self._log(f"📊 Loaded {len(questions)} exact match questions")
        else:
            # Default to multiple choice (existing behavior)
            questions = loader.load_multiple_choice_only()
            self._log(f"📊 Loaded {len(questions)} multiple choice questions")
        
        # Limit to max_questions if specified
        max_q = self.config['benchmark'].get('max_questions', len(questions))
        limited_questions = questions[:max_q]
        
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
            'calibration_error': 0.0
        }
        
        total_start_time = time.time()
        
        for i, question in enumerate(questions):
            self._log(f"  Question {i+1}/{len(questions)}")
            
            try:
                # Use the formatted question
                question_text = question['formatted_question']
                
                # Create a simple config for single model
                question_type = self.config['benchmark'].get('question_type', 'multipleChoice')
                
                if question_type == 'exactMatch':
                    default_system_message = 'You are a helpful AI assistant. For exact match questions, provide the exact answer and end with "The answer is: [your answer]".'
                else:
                    # Default to multiple choice (existing behavior)
                    default_system_message = 'You are a helpful AI assistant. For multiple choice questions, provide your analysis and reasoning, but always end your response with ONLY the letter of your chosen option in this exact format: The answer is: X, where X is A, B, C, D, ...Z .Do not include any additional text after the answer letter and provide confidence scores.'
                
                system_message = model_config.get('system_message', default_system_message)
                
                single_config = {
                    "agent": {
                        "id": model_config['name'],
                        "backend": model_config['backend'],
                        "system_message": system_message
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
                
                # Use pattern matching to extract answer from single model response
                correct_answer = question['answer']
                question_type = self.config['benchmark'].get('question_type', 'multipleChoice')
                extracted_answer = self._extract_answer_with_patterns(response_content, question_type)

                # Simple comparison for single models
                is_correct = extracted_answer.strip().lower() == correct_answer.strip().lower()

                if is_correct:
                    results['correct'] += 1

                # Store results
                results['responses'].append({
                    'question_id': question['id'],
                    'question': question['original_question'],
                    'response': response_content,
                    'correct_answer': correct_answer,
                    'extracted_answer': extracted_answer,
                    'is_correct': is_correct,
                    'response_time': question_time
                })

                # Log the extracted answer vs correct answer
                self._log(f"    Answer: {extracted_answer}, Correct: {correct_answer} {'✅' if is_correct else '❌'}")

                results['confidence_scores'].append(0.8 if is_correct else 0.2)
                
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
                    'judge_evaluation': {'is_correct': False, 'judge_reasoning': 'Error occurred', 'confidence': 0.0},
                    'is_correct': False,
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
    
    def _extract_answer_with_patterns(self, response: str, question_type: str = 'multipleChoice') -> str:
        """Extract answer from single model response using pattern matching."""
        if not response:
            return "No answer found"
        
        response = response.strip()
        
        if question_type == 'exactMatch':
            # Pattern 1: Look for "The answer is:" patterns
            answer_patterns = [
                r'[Tt]he answer is:\s*(.+)',
                r'[Aa]nswer:\s*(.+)',
                r'[Ee]xact answer:\s*(.+)',
                r'[Ff]inal answer:\s*(.+)'
            ]
            
            for pattern in answer_patterns:
                matches = re.findall(pattern, response)
                if matches:
                    answer = matches[-1].strip()
                    # Remove trailing punctuation and clean up
                    answer = re.sub(r'[.!?]+$', '', answer)
                    return answer
            
            # Pattern 2: Look for LaTeX boxed answers: \boxed{content}
            boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', response)
            if boxed_matches:
                return boxed_matches[-1].strip()
            
            return "No answer found"
        
        else:  # multipleChoice
            # Pattern 1: Look for "The answer is:" patterns
            answer_patterns = [
                r'[Tt]he answer is:\s*([A-Z])',
                r'[Aa]nswer:\s*([A-Z])',
                r'[Oo]ption\s*([A-Z])',
                r'[Cc]hoice\s*([A-Z])'
            ]
            
            for pattern in answer_patterns:
                matches = re.findall(pattern, response)
                if matches:
                    return matches[-1].strip()
            
            # Pattern 2: Look for LaTeX boxed answers: \boxed{X}
            boxed_matches = re.findall(r'\\boxed\{([A-Z])\}', response)
            if boxed_matches:
                return boxed_matches[-1].strip()
            
            return "No answer found"
    
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

    async def benchmark_multi_agent_cli(self, questions: List[Dict]) -> Dict[str, Any]:
        """Benchmark multi-agent system using CLI."""
        self._log(" Benchmarking Multi-Agent System using CLI...")
        
        # Get multi-agent config path
        ma_config_path = self.config['benchmark']['multi_agent']['config_file']
        resolved_path = self._resolve_config_path(ma_config_path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Multi-agent config not found: {ma_config_path}")
        
        self._log(f"  Using multi-agent config: {resolved_path}")
        
        # Get output format from benchmark config
        output_format = self.config['benchmark'].get('output', {}).get('format', 'text')
        
        results = {
            'model': 'Multi-Agent System',
            'correct': 0,
            'total': len(questions),
            'responses': [],
            'response_time': 0.0,
            'calibration_error': 0.0
        }
        
        total_start_time = time.time()
        
        for i, question in enumerate(questions):
            self._log(f"  Question {i+1}/{len(questions)}")
            
            try:
                # Use the formatted question
                question_text = question['formatted_question']
                
                # Run CLI command with output format
                multi_agent_start_time = time.time()
                cmd = [
                    sys.executable, "-m", "massgen.cli",
                    "--config", str(resolved_path),
                    "--no-display",
                ]
                
                # Add output format if specified
                if output_format == "json":
                    cmd.append("--json")
                
                cmd.append(question_text)
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=600,
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONLEGACYWINDOWSSTDIO': 'utf-8'}
                )
                
                multi_agent_end_time = time.time()
                multi_agent_response_time = multi_agent_end_time - multi_agent_start_time
                
                # Log multi-agent response time
                self._log(f"    Multi-agent response time: {multi_agent_response_time:.2f}s")
                
                if result.returncode != 0:
                    self._log(f"    ❌ CLI command failed: {result.stderr}")
                    results['responses'].append({
                        'question_id': question['id'],
                        'error': f"CLI failed: {result.stderr}",
                        'is_correct': False,
                        'response_time': multi_agent_response_time
                    })
                    continue
                
                # Parse response based on output format
                if output_format == "json":
                    try:
                        json_response = json.loads(result.stdout)
                        response_content = json_response.get('response', '')
                        selected_agent = json_response.get('selected_agent', 'unknown')
                    except json.JSONDecodeError as e:
                        self._log(f"    ❌ Failed to parse JSON response: {e}")
                        response_content = result.stdout
                        selected_agent = 'unknown'
                else:
                    # Fallback to text parsing
                    response_content = result.stdout
                    selected_agent = 'unknown'
                
                # Use judge model to evaluate the response
                correct_answer = question['answer']
                judge_start_time = time.time()
                evaluation = await self._evaluate_response_with_judge(question, response_content, correct_answer)
                judge_end_time = time.time()
                judge_evaluation_time = judge_end_time - judge_start_time
                
                # Log judge evaluation time
                self._log(f"    Judge evaluation time: {judge_evaluation_time:.2f}s")
                
                if evaluation['is_correct']:
                    results['correct'] += 1
                
                # Store results with both times
                results['responses'].append({
                    'question_id': question['id'],
                    'question': question['original_question'],
                    'response': response_content,
                    'correct_answer': correct_answer,
                    'judge_evaluation': evaluation,
                    'is_correct': evaluation['is_correct'],
                    'response_time': multi_agent_response_time,
                    'judge_evaluation_time': judge_evaluation_time,
                    'selected_agent': selected_agent
                })
                
                # Updated logging to show extracted answer vs correct answer
                extracted_answer = evaluation.get('extracted_answer', 'No answer extracted')
                self._log(f"    Answer: {extracted_answer}, Correct: {correct_answer} {'✅' if evaluation['is_correct'] else '❌'}")
                if selected_agent != 'unknown':
                    self._log(f"    Selected Agent: {selected_agent}")
                
            except subprocess.TimeoutExpired:
                self._log(f"    ❌ Timeout after 600 seconds")
                results['responses'].append({
                    'question_id': question['id'],
                    'error': 'Timeout',
                    'is_correct': False,
                    'response_time': 600.0,
                    'judge_evaluation_time': 0.0
                })
            except Exception as e:
                self._log(f"    ❌ Error: {e}")
                results['responses'].append({
                    'question_id': question['id'],
                    'error': str(e),
                    'is_correct': False,
                    'response_time': 0.0,
                    'judge_evaluation_time': 0.0
                })
        
        # Calculate total response time
        results['response_time'] = time.time() - total_start_time
        
        # Calculate accuracy
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
        
        self._log(f"✅ Multi-Agent System completed: {results['correct']}/{results['total']} correct ({results['accuracy']:.3f})")
        return results

    def _calculate_calibration_error(self, results: Dict) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        if not results.get('confidence_scores'):
            return 0.0
        
        # Simple ECE calculation
        confidence_scores = results['confidence_scores']
        correct_predictions = [1 if r['is_correct'] else 0 for r in results['responses']]
        
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
            ma_results = await self.benchmark_multi_agent_cli(questions)
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
                    print(f"    Judge Evaluation: {response.get('judge_evaluation', {}).get('judge_reasoning', 'N/A')}")
                    print(f"    Correct: {'✅' if response['is_correct'] else '❌'}")
            
            # Multi-agent response
            ma_result = self.results.get('multi_agent')
            if ma_result and i < len(ma_result['responses']):
                ma_response = ma_result['responses'][i]
                print(f"  Multi-Agent (CLI):")
                if 'error' in ma_response:
                    print(f"    Error: {ma_response['error']}")
                else:
                    print(f"    Response: {ma_response['response'][:100]}...")
                    print(f"    Judge Evaluation: {ma_response.get('judge_evaluation', {}).get('judge_reasoning', 'N/A')}")
                    print(f"    Correct: {'✅' if ma_response['is_correct'] else '❌'}")
