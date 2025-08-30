#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute MassGen metrics over 198 cases under massgen_logs_origin:

(1) Correct Answer totals per model
(2) Self voting: "self votes / total votes" per model
(3) Vote/Improvement Ratio: "votes / answers" per model
(4) Tie count (last line contains 'tie-broken by registration order')
(5) Consensus Round list of size 198:
    [{"case_index": "<i>", "model": "<model_or_tie>", "consensus_round": "<max_count>"}]
    - For ties of the max Answer-provided count, models are joined with "+" (e.g., "gpt5+grok4")
(6) Voting Distribution: total votes received per model across all cases

Notes:
- Lines in system_status.txt can include bullets, emojis, timestamps, etc.
  We normalize each line by stripping special characters and then regex search.
- Model answer is extracted from <model>.txt by pattern:  "answer": "<A/B/C/D>"
- Dataset: "fingertap/GPQA-Diamond" (split="test") must be accessible (huggingface-cli login).
"""

import os
import re
import json
from collections import defaultdict, Counter

# ---------------- Config ----------------
BASE_DIR = "massgen_logs_show_voting_before"
MODELS = ["claudesonnet420250514", "gemini2.5pro", "gpt5", "grok4"]
MODEL_FILE = {m: f"{m}.txt" for m in MODELS}
TOTAL_CASES = 198

# --- auto-install guards ---
try:
    from datasets import load_dataset
except ModuleNotFoundError:
    import sys, subprocess
    print("Installing 'datasets' ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "datasets"])
    from datasets import load_dataset


# ---------------- Helpers ----------------
def normalize_line(s: str) -> str:
    """
    Keep only alphanumerics, whitespace, colon, square brackets, dot, dash, underscore.
    This drops bullets, emojis, and other symbols so regex becomes reliable.
    """
    return re.sub(r"[^A-Za-z0-9\[\]\:\.\-\_\s]", "", s).strip()


# Regex on normalized lines
MODEL_ALT = r"(?:claudesonnet420250514|gemini2\.5pro|gpt5|grok4)"
VOTE_RE = re.compile(
    rf"\b(?P<src>{MODEL_ALT})\s*:\s*Vote\s+recorded\s+for\s*\[(?P<dst>{MODEL_ALT})\]",
    re.IGNORECASE,
)
ANSWER_PROVIDED_RE = re.compile(
    rf"\b(?P<src>{MODEL_ALT})\s*:\s*Answer\s+provided\b",
    re.IGNORECASE,
)

# For extracting model answer from <model>.txt
ANSWER_JSON_RE = re.compile(r'"answer"\s*:\s*"([A-Da-d])"')


def find_latest_case_folder(case_idx: int) -> str | None:
    """Pick lexicographically latest folder matching log_question_number_<x>_*."""
    prefix = f"log_question_number_{case_idx}_"
    try:
        entries = [
            d for d in os.listdir(BASE_DIR)
            if d.startswith(prefix) and os.path.isdir(os.path.join(BASE_DIR, d))
        ]
    except FileNotFoundError:
        return None
    if not entries:
        return None
    entries.sort()
    return entries[-1]


def read_text(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except FileNotFoundError:
        return []


def extract_model_answer(txt_path: str) -> str | None:
    """
    Return 'A'/'B'/'C'/'D' if found; otherwise None.
    If multiple matches exist, take the first.
    """
    if not os.path.isfile(txt_path):
        return None
    content = ""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None
    matches = ANSWER_JSON_RE.findall(content)
    if not matches:
        return None
    return matches[-1].upper()


# ---------------- Main ----------------
def main():
    # Load dataset (answers)
    ds = load_dataset("fingertap/GPQA-Diamond", split="test")
    if len(ds) < TOTAL_CASES:
        print(f"[WARN] Dataset size {len(ds)} < expected {TOTAL_CASES}")

    # (1) Correct counts per model
    correct_counts = Counter()

    # (2) Self voting: per model
    self_votes = Counter()         # model -> self-votes
    total_votes_by_src = Counter() # model -> total votes cast (as src)

    # (3) Vote/Improvement: track votes & answers-provided
    answer_provided_counts = Counter()  # per model

    # (4) Tie ratio: count of cases whose LAST line contains the phrase
    tie_count = 0

    # (5) Consensus rounds list
    consensus_list = []

    # (6) Voting Distribution: total votes RECEIVED by model (dst)
    votes_received = Counter()

    for i in range(TOTAL_CASES):
        case = ds[i]
        true_answer = str(case.get("answer", "")).strip().upper()

        latest = find_latest_case_folder(i)
        if latest is None:
            # No folder for this case — still push consensus element
            consensus_list.append({
                "case_index": str(i),
                "model": "",
                "consensus_round": "0",
            })
            continue

        agent_dir = os.path.join(BASE_DIR, latest, "agent_outputs")
        # --- (1) Correct answer per model ---
        for m in MODELS:
            ans = extract_model_answer(os.path.join(agent_dir, MODEL_FILE[m]))
            if ans is not None and ans == true_answer:
                correct_counts[m] += 1

        # --- system_status parsing for (2)(3)(4)(5)(6) ---
        sys_path = os.path.join(agent_dir, "system_status.txt")
        lines = read_text(sys_path)

        # Track per-case Answer provided counts for consensus (5)
        per_case_answer_provided = Counter()

        last_non_empty_norm = ""
        for raw in lines:
            norm = normalize_line(raw)
            if not norm:
                continue
            last_non_empty_norm = norm

            # Answer provided
            m_ap = ANSWER_PROVIDED_RE.search(norm)
            if m_ap:
                src = m_ap.group("src")
                # Normalize to exact model key (lower/upper issues)
                for model_key in MODELS:
                    if src.lower() == model_key.lower():
                        src = model_key
                        break
                answer_provided_counts[src] += 1
                per_case_answer_provided[src] += 1

            # Vote recorded
            m_vote = VOTE_RE.search(norm)
            if m_vote:
                src = m_vote.group("src")
                dst = m_vote.group("dst")
                # Normalize exact keys
                for model_key in MODELS:
                    if src.lower() == model_key.lower():
                        src = model_key
                    if dst.lower() == model_key.lower():
                        dst = model_key
                total_votes_by_src[src] += 1
                votes_received[dst] += 1
                if src == dst:
                    self_votes[src] += 1

        # (4) Tie ratio by last line
        if last_non_empty_norm and "tie-broken by registration order".lower() in last_non_empty_norm.lower():
            tie_count += 1

        # (5) Consensus round (max # of "Answer provided" among models)
        if per_case_answer_provided:
            max_val = max(per_case_answer_provided.values())
            winners = [m for m, c in per_case_answer_provided.items() if c == max_val]
            winners.sort()
            winner_str = "+".join(winners)
            consensus_list.append({
                "case_index": str(i),
                "model": winner_str,
                "consensus_round": str(max_val),
            })
        else:
            consensus_list.append({
                "case_index": str(i),
                "model": "",
                "consensus_round": "0",
            })

    # ---------------- Print Results ----------------
    print("\n=== (1) Correct Answer totals (across 198 cases) ===")
    for m in MODELS:
        print(f"{m}: {correct_counts[m]}")

    print("\n=== (2) Self voting totals (self / voting recorded as src) ===")
    for m in MODELS:
        print(f"{m}: {self_votes[m]} / {total_votes_by_src[m]}")

    print("\n=== (3) Vote/Improvement Ratio totals (votes / answers-provided) ===")
    for m in MODELS:
        print(f"{m}: {total_votes_by_src[m]} / {answer_provided_counts[m]}")

    print("\n=== (4) Tie count (last line contains 'tie-broken by registration order') ===")
    print(tie_count)

    print("\n=== (5) Consensus Round list (size should be 198) ===")
    # Print as JSON list (readable)
    print(json.dumps(consensus_list, ensure_ascii=False, indent=2))

    print("\n=== (6) Voting Distribution totals (votes RECEIVED by model) ===")
    for m in MODELS:
        print(f"{m}: {votes_received[m]}")


if __name__ == "__main__":
    main()
