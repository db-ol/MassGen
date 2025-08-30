#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
from datasets import load_dataset

# ====== Config ======
LOG_DIR = "massgen_logs_show_voting_before"   # <- switched
OUT_CSV = "evaluation_results.csv"
TOTAL = 198

# Load dataset (login if needed: `huggingface-cli login`)
ds = load_dataset("fingertap/GPQA-Diamond")

# Regex
answer_pattern = re.compile(r'"answer":\s*"([A-D])"')
final_selected_pattern = re.compile(r'FINAL:\s*([A-Za-z0-9._-]+)\s+selected', re.IGNORECASE)

def extract_answer_from_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = answer_pattern.search(line)
                if m:
                    return m.group(1)
    except Exception as e:
        print(f"[Error] Reading {path}: {e}")
    return ""

def find_selected_agent(agent_outputs_dir: str) -> str:
    status_path = os.path.join(agent_outputs_dir, "system_status.txt")
    if not os.path.exists(status_path):
        return ""
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            for line in f:
                m = final_selected_pattern.search(line)
                if m:
                    return m.group(1).strip()
    except Exception as e:
        print(f"[Error] Reading {status_path}: {e}")
    return ""

def find_agent_answer(agent_outputs_dir: str, selected_agent: str) -> str:
    # Prefer the final_presentation for the selected agent
    if selected_agent:
        candidates = [
            f"final_presentation_{selected_agent}_latest.txt",
            f"final_presentation_{selected_agent}.txt",
        ]
        for c in candidates:
            p = os.path.join(agent_outputs_dir, c)
            if os.path.exists(p):
                ans = extract_answer_from_file(p)
                if ans:
                    return ans

    # Fallback: scan any final_presentation_*.txt
    try:
        for fname in sorted(os.listdir(agent_outputs_dir)):
            if fname.startswith("final_presentation_") and fname.endswith(".txt"):
                p = os.path.join(agent_outputs_dir, fname)
                ans = extract_answer_from_file(p)
                if ans:
                    return ans
    except FileNotFoundError:
        pass

    return ""

rows = []
correct_sum = 0

missing_folder = []
missing_status = []
missing_answer = []

for x in range(TOTAL):
    case = ds["test"][x]
    question = case["question"]
    true_answer = case["answer"]

    # ---- Locate latest log folder for this question ----
    log_folders = [f for f in os.listdir(LOG_DIR) if f.startswith(f"log_question_number_{x}_")]
    if not log_folders:
        print(f"[Warning] No log folder for question {x}")
        missing_folder.append(x)
        selected_agent = ""
        agent_answer = ""
    else:
        log_folders.sort()  # timestamped names sort lexicographically by time
        latest_folder = log_folders[-1]
        agent_outputs_dir = os.path.join(LOG_DIR, latest_folder, "agent_outputs")

        # ---- Parse selected_agent from system_status.txt ----
        selected_agent = ""
        if os.path.isdir(agent_outputs_dir):
            selected_agent = find_selected_agent(agent_outputs_dir)
            if not selected_agent:
                missing_status.append(x)

            # ---- Read agent answer ----
            agent_answer = find_agent_answer(agent_outputs_dir, selected_agent)
            if not agent_answer:
                print(f"[Warning] No agent_answer found for question {x}, leaving blank.")
                missing_answer.append(x)
        else:
            print(f"[Warning] agent_outputs not found for question {x} in {latest_folder}")
            missing_folder.append(x)
            selected_agent = ""
            agent_answer = ""

    # ---- Correct or not ----
    correct_or_not = 1 if agent_answer and agent_answer == true_answer else 0
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

    # summary rows
    accuracy = correct_sum / TOTAL if TOTAL else 0.0
    writer.writerow([])
    writer.writerow(["Sum correct_or_not", correct_sum])
    writer.writerow(["Accuracy", accuracy])

print(f"CSV saved to {OUT_CSV}")
if missing_folder:
    print("[Missing log folders]:", missing_folder)
if missing_status:
    print("[Missing selected_agent in system_status.txt]:", missing_status)
if missing_answer:
    print("[Missing agent_answer in final_presentation files]:", missing_answer)
