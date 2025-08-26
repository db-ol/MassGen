#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan MassGen logs and extract agent answers into a CSV.

- Looks under <root>/log_question_number_<x>_<timestamp>/agent_outputs/
- Reads these files (if present):
    - claudesonnet420250514.txt
    - gemini2.5pro.txt
    - gpt5.txt
    - grok4.txt
- Finds the last occurrence of: "answer": "<A/B/C/D>"
- Outputs CSV with columns:
    index, claudesonnet420250514, gemini2.5pro, gpt5, grok4
- If not found, writes NOT_FOUND and prints the NOT_FOUND index lists.

Usage:
    python extract_agent_answers.py \
      --root "/Users/ruofanz/workspace/mass_gen/MassGen/massgen_logs_back" \
      --out "agent_answers.csv"
"""

import os
import re
import csv
import glob
import argparse

# Ordered columns exactly as requested
AGENT_FILES = [
    ("claudesonnet420250514", "claudesonnet420250514.txt"),
    ("gemini2.5pro",          "gemini2.5pro.txt"),
    ("gpt5",                  "gpt5.txt"),
    ("grok4",                 "grok4.txt"),
]

ANSWER_RE = re.compile(r'"answer"\s*:\s*"([ABCD])"', re.IGNORECASE)

def extract_answer_from_text(text: str) -> str:
    """Return the LAST A/B/C/D found in 'answer' field, else NOT_FOUND."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return "NOT_FOUND"
    return matches[-1].upper()

def find_agent_outputs_dir(root: str, idx: int) -> str | None:
    """
    Find the agent_outputs folder for question idx.
    If multiple timestamped folders exist, pick the lexicographically latest.
    """
    pattern = os.path.join(root, f"log_question_number_{idx}_*", "agent_outputs")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return sorted(candidates)[-1]

def read_file_safe(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract agent answers into CSV.")
    parser.add_argument(
        "--root",
        default="/Users/ruofanz/workspace/mass_gen/MassGen/massgen_logs_show_voting_before",
        help="Root folder containing log_question_number_* subfolders.",
    )
    parser.add_argument(
        "--out",
        default="agent_answers.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive). Default: 0",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=197,
        help="End index (inclusive). Default: 197",
    )
    args = parser.parse_args()

    headers = ["index"] + [name for name, _ in AGENT_FILES]
    rows = []

    for idx in range(args.start, args.end + 1):
        row = {"index": idx}
        agent_dir = find_agent_outputs_dir(args.root, idx)

        for agent_name, filename in AGENT_FILES:
            if agent_dir is None:
                row[agent_name] = "NOT_FOUND"
                continue

            file_path = os.path.join(agent_dir, filename)
            text = read_file_safe(file_path)
            if text is None:
                row[agent_name] = "NOT_FOUND"
            else:
                row[agent_name] = extract_answer_from_text(text)

        rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")

    # Build NOT_FOUND lists
    missing_by_agent = {name: [] for name, _ in AGENT_FILES}
    missing_any = []
    for r in rows:
        any_missing = False
        for name, _ in AGENT_FILES:
            if r[name] == "NOT_FOUND":
                missing_by_agent[name].append(r["index"])
                any_missing = True
        if any_missing:
            missing_any.append(r["index"])

    # Print counts and lists
    print("\n=== NOT_FOUND summary ===")
    for name in [n for n, _ in AGENT_FILES]:
        lst = missing_by_agent[name]
        print(f"{name}: {len(lst)} missing")
        if lst:
            print(f"  indices: {', '.join(map(str, lst))}")

    print(f"\nAny-agent NOT_FOUND: {len(missing_any)}")
    if missing_any:
        print(f"  indices: {', '.join(map(str, missing_any))}")

if __name__ == "__main__":
    main()
