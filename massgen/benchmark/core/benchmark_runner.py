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
import warnings
import logging
from logging.handlers import RotatingFileHandler

# Suppress all warnings and set logging to ERROR only
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress specific warnings
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

from massgen.cli import create_backend, create_agents_from_config
from .load_dataset import HLEDatasetLoader, get_dataset_loader

class HLEBenchmarkRunner:
    def __init__(self, config_path: str, dataset_name: str = "hle-lite"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.results = {}
        self.logs = []
        self.current_questions = []
        self.judge_agent = None
        self.dataset_name = dataset_name
        
        # Set up rotating log handler
        log_file = "agent_outputs/benchmark.log"
        handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Get the logger and add handler
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
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

    def _extract_answer_from_response(self, response: str, question_type: str) -> str:
        """Extract answer from model response with better error handling."""
        if not response or response.strip() == "":
            return "No answer found"
        
        response = response.strip()
        
        try:
            if question_type == "multipleChoice":
                # Look for "The answer is: X" pattern
                pattern = r"The answer is:\s*([A-Z])"
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    answer = match.group(1).upper()
                    # Accept any letter A-Z (MMLU-Pro can have up to 26 options)
                    if answer in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        return answer
                
                # Fallback: look for single letter at the end
                lines = response.split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if len(line) == 1 and line.isalpha():
                        return line.upper()
                    # Check for patterns like "Answer: A" or "A)" or "A."
                    match = re.search(r'[A-Z]\)?\.?$', line)
                    if match:
                        return match.group(0)[0].upper()
                
                # Look for LaTeX boxed format: $\boxed{X}$
                pattern = r"\\boxed\{([A-Z])\}"
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    answer = match.group(1).upper()
                    if answer in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        return answer
                
                return "No answer found"
                
            elif question_type == "exactMatch":
                # Look for "The answer is: [answer]" pattern
                pattern = r"The answer is:\s*(.+)"
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
                
                # Fallback: return the last non-empty line
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                if lines:
                    return lines[-1]
                
                return "No answer found"
            else:
                return "No answer found"
                
        except Exception as e:
            print(f"Warning: Error extracting answer: {e}")
            return "No answer found"
    
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

Return ONLY a JSON object with these fields:
- "is_correct": true/false
- "judge_reasoning": brief explanation
- "extracted_answer": the answer extracted from the response
- "confidence": 0.0-1.0

JSON:"""
        else:  # multipleChoice
            evaluation_prompt = f"""Evaluate this response and return ONLY a JSON object:

Question: {question['original_question']}
Correct Answer: {correct_answer}
Response: {response}

Return ONLY a JSON object with these fields:
- "is_correct": true/false
- "judge_reasoning": brief explanation
- "extracted_answer": the letter (A, B, C, D, etc.) extracted from the response
- "confidence": 0.0-1.0

JSON:"""
        
        try:
            # Get judge evaluation
            judge_response = await judge_agent.get_response(evaluation_prompt)
            
            # Try to parse JSON response
            try:
                # Look for JSON in the response
                json_start = judge_response.find('{')
                json_end = judge_response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = judge_response[json_start:json_end]
                    evaluation = json.loads(json_str)
                else:
                    # Fallback: extract answer manually
                    extracted_answer = self._extract_answer_from_response(judge_response, question_type)
                    evaluation = {
                        'is_correct': extracted_answer.upper() == correct_answer.upper(),
                        'judge_reasoning': 'Manual extraction',
                        'extracted_answer': extracted_answer,
                        'confidence': 0.5
                    }
            except json.JSONDecodeError:
                # Fallback: extract answer manually
                extracted_answer = self._extract_answer_from_response(judge_response, question_type)
                evaluation = {
                    'is_correct': extracted_answer.upper() == correct_answer.upper(),
                    'judge_reasoning': 'Manual extraction due to JSON parse error',
                    'extracted_answer': extracted_answer,
                    'confidence': 0.5
                }
            
            return evaluation
            
        except Exception as e:
            self._log(f"❌ Judge evaluation failed: {e}")
            # Fallback: extract answer manually
            extracted_answer = self._extract_answer_from_response(response, question_type)
            return {
                'is_correct': extracted_answer.upper() == correct_answer.upper(),
                'judge_reasoning': f'Judge failed: {e}',
                'extracted_answer': extracted_answer,
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
                extracted_answer = self._extract_answer_from_response(response_content, question_type)

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
                        # Debug: log the raw stdout
                        self._log(f"    Raw CLI output length: {len(result.stdout)}")
                        if len(result.stdout) > 200:
                            self._log(f"    Raw CLI output preview: {result.stdout[:200]}...")
                        else:
                            self._log(f"    Raw CLI output: {result.stdout}")
                        
                        # Clean the output to remove control characters that break JSON parsing
                        cleaned_output = result.stdout
                        # Remove control characters except newlines, tabs, and carriage returns
                        cleaned_output = ''.join(char for char in cleaned_output if ord(char) >= 32 or char in '\n\r\t')
                        
                        # Try to extract JSON from the cleaned output
                        json_start = cleaned_output.find('{')
                        json_end = cleaned_output.rfind('}') + 1
                        
                        if json_start != -1 and json_end > json_start:
                            json_content = cleaned_output[json_start:json_end]
                            # Try to parse the JSON
                            try:
                                response_data = json.loads(json_content)
                                response = response_data.get('response', '')
                                self._log(f"    Successfully parsed JSON response: {len(response)} chars")
                            except json.JSONDecodeError as json_err:
                                self._log(f"    ❌ JSON decode error: {json_err}")
                                # Try to extract answer directly from the content
                                response = json_content
                                self._log(f"    Using JSON content as response: {len(response)} chars")
                        else:
                            # Fallback: try to extract answer from text
                            response = cleaned_output.strip()
                            self._log(f"    No JSON found, using raw output: {len(response)} chars")
                            
                    except Exception as e:
                        self._log(f"    ❌ Error parsing response: {e}")
                        # Fallback to raw output
                        response = result.stdout.strip()
                        self._log(f"    Using fallback response: {len(response)} chars")
                else:
                    response = result.stdout.strip()
                
                # Judge evaluation
                try:
                    judge_agent = await self._create_judge_agent()
                    
                    # Create judge prompt
                    judge_prompt = f"""Evaluate if the following answer is correct for the given question.

Question: {question['original_question']}
Correct Answer: {question['answer']}
Model Answer: {response}

Please respond with ONLY "CORRECT" or "INCORRECT" based on whether the model answer matches the correct answer."""

                    self._log(f"     Judge prompt: {judge_prompt}")
                    self._log(f"    🔍 Model response being judged: {response[:200]}...")

                    # Get judge response using chat method
                    messages = [{"role": "user", "content": judge_prompt}]
                    judge_response_content = ""
                    
                    async for chunk in judge_agent.chat(messages):
                        if chunk.type == "content" and chunk.content:
                            judge_response_content += chunk.content
                        elif chunk.type == "error":
                            raise Exception(f"Judge error: {chunk.error}")
                        elif chunk.type == "done":
                            break
                    
                    judge_response = judge_response_content.strip()
                    
                    # FIXED LOGIC: Check for exact match, not substring
                    is_correct = judge_response.upper() == "CORRECT"
                    
                    self._log(f"    ✅ Judge evaluation: {judge_response}")
                    self._log(f"    🔍 Is correct: {is_correct}")
                    
                except Exception as e:
                    self._log(f"    ❌ Judge evaluation failed: {e}")
                    # Fallback: use pattern matching with correct field name
                    is_correct = self._check_answer_pattern(response, question['answer'])
                    self._log(f"    🔍 Fallback is_correct: {is_correct}")
                
                # Extract answer from response for logging
                question_type = self.config['benchmark'].get('question_type', 'multipleChoice')
                extracted_answer = self._extract_answer_from_response(response, question_type)
                
                # Store results with both times
                results['responses'].append({
                    'question_id': question['id'],
                    'question': question['original_question'],
                    'response': response,
                    'correct_answer': question['answer'],
                    'extracted_answer': extracted_answer,
                    'judge_evaluation': {'is_correct': is_correct, 'judge_reasoning': judge_response.strip(), 'confidence': 0.0},
                    'is_correct': is_correct,
                    'response_time': multi_agent_response_time,
                    'judge_evaluation_time': 0.0, # Judge evaluation time is not directly available from CLI output
                    'selected_agent': 'unknown' # No direct agent selection in CLI output
                })
                
                # Updated logging to show extracted answer vs correct answer
                self._log(f"    Answer: {extracted_answer}, Correct: {question['answer']} {'✅' if is_correct else '❌'}")
                
                if is_correct:
                    results['correct'] += 1
                
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
        """Run the benchmark with the specified dataset."""
        self._log(f"🚀 Starting {self.dataset_name} Benchmark...")
        
        # Get the appropriate dataset loader
        try:
            dataset_loader = get_dataset_loader(self.dataset_name, token)
        except Exception as e:
            self._log(f"❌ Failed to get dataset loader: {e}")
            return {}
        
        # Load questions based on question type
        question_type = self.config['benchmark'].get('question_type', 'multipleChoice')
        
        try:
            if question_type == 'multipleChoice':
                if self.dataset_name.lower() in ['hle', 'hle-lite']:
                    questions = dataset_loader.load_multiple_choice_only()
                else:  # MMLU-Pro
                    questions = dataset_loader.load_mcq_only()
            elif question_type == 'exactMatch':
                if self.dataset_name.lower() in ['hle', 'hle-lite']:
                    questions = dataset_loader.load_exact_match_only()
                else:  # MMLU-Pro
                    questions = dataset_loader.load_exact_match_only()
            else:
                # Load all types
                questions = dataset_loader.load_dataset()
        except Exception as e:
            self._log(f"❌ Failed to load dataset: {e}")
            return {}
        
        # Limit questions if specified
        max_questions = self.config['benchmark'].get('max_questions', len(questions))
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        self.current_questions = questions
        self._log(f"📋 Question type to benchmark: {question_type}")
        
        # Initialize results
        self.results = {
            'single_models': {},
            'multi_agent': None,
            'summary': {}
        }
        
        # Benchmark single models
        single_models = self.config['benchmark'].get('single_models', [])
        for model_config in single_models:
            model_name = model_config['name']
            self._log(f"🧪 Benchmarking single model: {model_name}")
            results = await self.benchmark_single_model(model_config, questions)
            self.results['single_models'][model_name] = results
        
        # Benchmark multi-agent system
        multi_agent_config = self.config['benchmark'].get('multi_agent')
        if multi_agent_config:
            self._log(" Benchmarking multi-agent system...")
            results = await self.benchmark_multi_agent_cli(questions)
            self.results['multi_agent'] = results
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self._save_results()
        
        return self.results

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

    def _save_results(self):
        """Save benchmark results to JSON file."""
        try:
            # Get results file path from config
            results_file = self.config['benchmark'].get('results_file', 'benchmark_results.json')
            
            # Save results
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            self._log(f"💾 Results saved to {results_file}")
            
        except Exception as e:
            self._log(f"❌ Failed to save results: {e}")

    def _check_answer_pattern(self, response: str, correct_answer: str) -> bool:
        """Check if the response contains the correct answer pattern."""
        if not response or not correct_answer:
            return False
        
        # Extract answer from response
        extracted_answer = self._extract_answer_from_response(response, "multipleChoice")
        
        # Compare with correct answer
        return extracted_answer.strip().lower() == correct_answer.strip().lower()
