import os
import re
import json
import csv
from datasets import load_dataset

# ====== Config ======
LOG_DIR = "massgen_logs"
RESULT_DIR = "ruofan_result"
OUT_CSV = "evaluation_results.csv"

# Load dataset
ds = load_dataset("fingertap/GPQA-Diamond")

# Regex to capture agent answer line
answer_pattern = re.compile(r'"answer":\s*"([A-D])"')

rows = []
total = 198
correct_sum = 0

for x in range(total):
    case = ds["test"][x]
    question = case["question"]
    true_answer = case["answer"]

    # ---- Read agent answer ----
    agent_answer = ""
    # Find the log folder for this question (there may be multiple timestamps)
    log_folders = [f for f in os.listdir(LOG_DIR) if f.startswith(f"log_question_number_{x}_")]
    if log_folders:
        # Use the latest (sorted by name)
        log_folders.sort()
        latest_folder = log_folders[-1]
        agent_outputs_dir = os.path.join(LOG_DIR, latest_folder, "agent_outputs")
        if os.path.isdir(agent_outputs_dir):
            for fname in os.listdir(agent_outputs_dir):
                if fname.startswith("final_presentation_") and fname.endswith(".txt"):
                    path = os.path.join(agent_outputs_dir, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            match = answer_pattern.search(line)
                            if match:
                                agent_answer = match.group(1)
                                break
                    break
    if not agent_answer:
        print(f"[Warning] No agent_answer found for question {x}, leaving blank.")

    # ---- Read result JSON ----
    selected_agent = ""
    result_json_path = os.path.join(RESULT_DIR, f"{x}_result.json")
    if os.path.exists(result_json_path):
        with open(result_json_path, "r", encoding="utf-8") as f:
            try:
                result_data = json.load(f)
                selected_agent = result_data.get("selected_agent", "")
                # Overwrite true_answer from json if available
                true_answer = result_data.get("true_answer", true_answer)
            except Exception as e:
                print(f"[Error] Failed to load {result_json_path}: {e}")

    # ---- Correct or not ----
    correct_or_not = 1 if agent_answer == true_answer and agent_answer else 0
    correct_sum += correct_or_not

    rows.append([
        x,
        question,
        agent_answer,
        true_answer,
        selected_agent,
        correct_or_not
    ])

# ---- Write CSV ----
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question_number", "question", "agent_answer", "true_answer", "selected_agent", "correct_or_not"])
    writer.writerows(rows)

    # summary row
    accuracy = correct_sum / total
    writer.writerow([])
    writer.writerow(["Sum correct_or_not", correct_sum])
    writer.writerow(["Accuracy", accuracy])

print(f"CSV saved to {OUT_CSV}")
