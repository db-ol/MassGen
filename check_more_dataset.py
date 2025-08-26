from datasets import load_dataset
import random

# Load boolean_expressions dataset
dataset = load_dataset('maveriq/bigbenchhard', 'boolean_expressions')

# Randomly select 10 questions
indices = random.sample(range(len(dataset['train'])), 10)

print(f"Total questions: {len(dataset['train'])}")
print("\nRandomly selected 10 questions:")
for i, idx in enumerate(indices):
    print(f"\nQuestion {i+1} (index {idx}):")
    print(f"Input: {dataset['train'][idx]['input']}")
    print(f"Answer: {dataset['train'][idx]['target']}")

# Check answer distribution
true_count = sum(1 for item in dataset['train'] if item['target'] == 'True')
false_count = sum(1 for item in dataset['train'] if item['target'] == 'False')

print(f"\nAnswer distribution:")
print(f"True: {true_count} ({true_count/len(dataset['train'])*100:.2f}%)")
print(f"False: {false_count} ({false_count/len(dataset['train'])*100:.2f}%)")