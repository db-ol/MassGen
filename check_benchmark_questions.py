from massgen.benchmark.core.load_dataset import DatasetLoader

# Create DatasetLoader instance
loader = DatasetLoader("maveriq/bigbenchhard")

# Load BigBenchHard dataset
questions = loader.load_bigbenchhard_dataset()

# Print first 10 questions
print(f"Number of loaded questions: {len(questions)}")
print("\nFirst 10 questions:")
for i, q in enumerate(questions[:10]):
    print(f"\nQuestion {i+1}:")
    print(f"Original question: {q['original_question']}")
    print(f"Formatted question: {q['formatted_question']}")
    print(f"Correct answer: {q['answer']}")