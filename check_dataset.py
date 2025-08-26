from datasets import load_dataset

dataset = load_dataset('maveriq/bigbenchhard', 'boolean_expressions')

for i in range(5):
    print(f"\nQuestion {i+1}:")
    print(f"Input: {dataset['train'][i]['input']}")
    print(f"Target: {dataset['train'][i]['target']}")