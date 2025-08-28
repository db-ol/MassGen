from datasets import load_dataset
import json

# 加载MuSR数据集
print("Loading MuSR dataset...")
dataset = load_dataset("TAUR-Lab/MuSR")

# 打印数据集的键
print(f"Dataset keys: {dataset.keys()}")

# 检查train分割是否存在
if 'train' in dataset:
    print(f"Train split exists with {len(dataset['train'])} samples")
    
    # 打印第一个样本的结构
    print("\nFirst sample structure:")
    sample = dataset['train'][0]
    for key, value in sample.items():
        if isinstance(value, list) and len(value) > 5:
            print(f"{key}: {value[:5]}... (truncated)")
        else:
            print(f"{key}: {value}")
    
    # 保存前5个样本到文件以便检查
    print("\nSaving first 5 samples to musr_samples.json")
    with open("musr_samples.json", "w") as f:
        json.dump([dataset['train'][i] for i in range(min(5, len(dataset['train'])))], f, indent=2)
else:
    print("Train split does not exist!")