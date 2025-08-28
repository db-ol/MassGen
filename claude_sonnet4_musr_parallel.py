import os
import re
import csv
import json
import anthropic
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ====== Config ======
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # export ANTHROPIC_API_KEY=...
MODEL = "claude-3-5-sonnet-20241022"               # Claude Sonnet 4 model
START_ID = 0                                       # start from ID 0
END_ID = 100                                       # end at ID 100 (first 100 questions per split)
N_QUESTIONS = END_ID - START_ID                    # total questions to process per split
OUT_CSV_PREFIX = "claude_sonnet4_musr"             # output CSV prefix
MAX_WORKERS = 8                                    # parallel workers (reduced from 20)
SPLITS = ["murder_mysteries", "object_placements", "team_allocation"]  # all MuSR splits

# ====== Helpers ======
def row_to_item(row):
    narrative = (row.get("narrative") or "").strip()
    question = (row.get("question") or "").strip()
    choices = (row.get("choices") or "").strip()
    answer_index = row.get("answer_index", -1)
    answer_choice = (row.get("answer_choice") or "").strip()
    
    # Combine narrative and question for the full prompt
    full_question = f"{narrative}\n\n{question}\n\n{choices}"
    
    return {
        "question": full_question,
        "correct_answer_index": answer_index,
        "correct_answer_choice": answer_choice
    }

def call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            temperature=0,
            system="You are a helpful AI assistant. For narrative questions, provide your analysis and reasoning. Please give your answer in the following JSON format: ```json {\"answer\": <str>} ``` Please don't generate anything except JSON format.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return (response.content[0].text or "").strip()
    except Exception as e:
        return f"[error] {e}"

def extract_answer_from_json(response_text: str) -> str:
    """Extract answer from JSON format response"""
    try:
        # Try to find JSON block
        json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            return data.get("answer", "")
        
        # Try to parse the entire response as JSON
        data = json.loads(response_text)
        return data.get("answer", "")
    except:
        # Fallback: try to extract any letter A-Z
        match = re.search(r'\b([A-Z])\b', response_text)
        return match.group(1) if match else ""

def process_question(idx, item, client):
    """Process a single question and return the result dict, with logs."""
    print(f"\n[START] Q{idx}: {item['question'][:100]}...")  # log beginning

    answer_text = call_claude(client, item["question"])
    print(f"[ANSWER] Q{idx}: {answer_text}")

    pred_answer = extract_answer_from_json(answer_text)
    correct = (
        int(pred_answer == item["correct_answer_choice"])
        if item["correct_answer_choice"] else ""
    )

    print(f"[RESULT] Q{idx}: Pred={pred_answer}, Correct={item['correct_answer_choice']}, Match={correct==1}")

    return {
        "idx": idx,
        "question": item["question"],
        "correct_answer_index": item["correct_answer_index"],
        "correct_answer_choice": item["correct_answer_choice"],
        "model_answer": answer_text,
        "pred_answer": pred_answer,
        "correct": correct,
    }

def main():
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Please set ANTHROPIC_API_KEY in your environment.")

    print("Initializing Anthropic client...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    all_results = []
    total_correct = 0
    total_questions = 0

    for split in SPLITS:
        print(f"\nProcessing split: {split}")
        print(f"Loading MuSR dataset ({split})...")
        
        ds = load_dataset("TAUR-Lab/MuSR", split=split)
        end_idx = min(END_ID, len(ds))
        selected_ds = ds.select(range(START_ID, end_idx))
        items = [(START_ID + i, row_to_item(row)) for i, row in enumerate(selected_ds)]

        print(f"Processing {len(items)} questions from {split} in parallel (max_workers={MAX_WORKERS})...")

        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_question, idx, item, client): idx for idx, item in items}

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split}"):
                try:
                    result = future.result()
                    result["split"] = split  # Add split information
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    print(f"[ERROR] Q{idx}: {e}")
                    # Find the corresponding item
                    item_data = next((item for item_idx, item in items if item_idx == idx), (idx, {}))[1]
                    results.append({
                        "idx": idx,
                        "split": split,
                        "question": item_data.get("question", ""),
                        "correct_answer_index": item_data.get("correct_answer_index", ""),
                        "correct_answer_choice": item_data.get("correct_answer_choice", ""),
                        "model_answer": f"[error] {e}",
                        "pred_answer": "",
                        "correct": "",
                    })

        results.sort(key=lambda x: x["idx"])
        all_results.extend(results)
        
        # Save results for this split
        split_csv = f"{OUT_CSV_PREFIX}_{split}.csv"
        print(f"\nSaving {split} results to {split_csv}...")
        with open(split_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "split", "question", "correct_answer_index", "correct_answer_choice", "model_answer", "pred_answer", "correct"])
            for r in results:
                writer.writerow([
                    r["idx"], r["split"], r["question"], r["correct_answer_index"], r["correct_answer_choice"],
                    r["model_answer"], r["pred_answer"], r["correct"]
                ])
        
        split_correct = sum(1 for r in results if r["correct"] == 1)
        total_correct += split_correct
        total_questions += len(results)
        print(f"Split {split}: {split_correct}/{len(results)} correct ({split_correct/len(results)*100:.1f}%)")

    # Save combined results
    combined_csv = f"{OUT_CSV_PREFIX}_all_splits.csv"
    print(f"\nSaving combined results to {combined_csv}...")
    with open(combined_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "split", "question", "correct_answer_index", "correct_answer_choice", "model_answer", "pred_answer", "correct"])
        for r in all_results:
            writer.writerow([
                r["idx"], r["split"], r["question"], r["correct_answer_index"], r["correct_answer_choice"],
                r["model_answer"], r["pred_answer"], r["correct"]
            ])

    print(f"\nOverall Summary: {total_correct}/{total_questions} correct ({total_correct/total_questions*100:.1f}%)")
    print(f"Results saved to individual split files and {combined_csv}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    except Exception as e:
        print(f"Script failed: {e}")