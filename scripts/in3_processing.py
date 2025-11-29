from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import random
import json
from copy import deepcopy
from time import sleep
from typing import Dict, List, NotRequired, Tuple, TypedDict
import dotenv
import dashscope
from concurrent.futures import as_completed

class Detail(TypedDict):
    description: str
    importance: str
    inquiry: str
    options: List[str]

class KnownDetail(TypedDict):
    description: str
    value: str
    source_detail: Detail

class DataEntry(TypedDict):
    category: str
    task: str
    vague: bool
    thought: str
    missing_details: List[Detail]
    known_details: NotRequired[List[KnownDetail]] # only for modified entries

def split_entries(filename: str, output_dir: str | Path) -> None:
    '''
    split the dataset by the number of details missing and store them in separate files within the output directory
    '''
    output_dir = Path(output_dir)
    with open(filename, "r") as f:
        data: List[DataEntry] = [json.loads(i) for i in f]
    splits = {}
    for entry in data:
        n_missing = len(entry["missing_details"])
        splits.setdefault(n_missing, []).append(entry)
    output_dir.mkdir(parents=True, exist_ok=True)
    for n_missing, entries in splits.items():
        out_path = output_dir / f"entries_missing_{n_missing}.jsonl"
        with out_path.open("w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

def pick_entry_with_knowns(candidates: List[DataEntry], n_missing: int, m_known: int) -> DataEntry:
    """
    Sample a datum whose total details are n_missing + m_known, then mark m_known as known (g_a analogue).
    Returns the mutated entry (with at least n_missing remaining) and the list of known details with chosen values.
    """
    base = deepcopy(random.choice(candidates))
    total_details = base["missing_details"].copy()
    random.shuffle(total_details)
    known_raw: List[Detail] = total_details[:m_known]
    known: List[KnownDetail] = [{
        "description": det['description'],
        "value": random.choice(det["options"]) if det["options"] else "",
        "source_detail": det,
    } for det in known_raw]
    remaining: List[Detail] = total_details[m_known:]
    base["missing_details"] = remaining
    base["vague"] = n_missing > 0
    base["known_details"] = known
    return base

def mutate_known_details(entry: DataEntry) -> DataEntry:
    """
    Mutate known details by switching to a different available option when possible (g_m analogue).
    """
    mutated: List[KnownDetail] = []
    entry = deepcopy(entry)
    for kd in entry.get("known_details", []):
        options = kd["source_detail"]["options"]
        current = kd["value"]
        alternatives = [opt for opt in options if opt != current]
        new_val = random.choice(alternatives) if alternatives else current
        mutated.append({**kd, "value": new_val})
    entry["known_details"] = mutated
    return entry

rewrite_prompt = '''
You are a task polishing expert. Your job is to integrate detailed requirements into a vague task description to make it clearer and more specific. In this task, you will be provided with a vague task description and a list of missing details that need to be incorporated into the task. Your goal is to rewrite the task description by seamlessly integrating the provided missing details into the original description. You should not add any new information beyond what is provided in the missing details, and you should maintain the original intent of the vague task description. You should only output the rewritten task description without any additional commentary or explanation.
Here is the vague task description and the missing details:
'''

def _rewrite_with_llm(prompt: str) -> str:
    """
    Try using the OpenAI client from proactest.llms; fallback to empty string on failure.
    """
    while True:
        response = _get_response(prompt)
        if response.status_code == 429:
            print('rate limited, sleeping...')
            sleep(random.uniform(30, 60))
            continue
        elif response.status_code != 200:
            return ""
        try:
            result = response['output']["text"].strip()
        except Exception as e:
            print(response)
            print(e)
        print('response received:', result)
        return result

def _get_response(prompt):
    # use dashscope Qwen-plus for rewriting
    return dashscope.Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        model="qwen-plus",
        messages=[
            {"role": "system", "content": rewrite_prompt},
            {"role": "user", "content": prompt}
        ],
    )

def synthesize_task_with_knowns(entry: DataEntry) -> DataEntry:
    """
    Integrate known details into a task description using a provided LLM callable; fallback to appending facts.
    """
    prompt = [
        f"Original task:\n{entry['task']}",
        "",
        "Known details:"
    ]
    for kd in entry.get("known_details", []):
        prompt.append(f"- {kd['description']}: {kd['value']}")
    prompt_text = "\n".join(prompt)

    new_task = _rewrite_with_llm(prompt_text)
    if not new_task:
        known_text = "; ".join(f"{kd['description']}: {kd['value']}" for kd in entry.get("known_details", []))
        new_task = f"{entry['task']} (Details provided: {known_text})"

    out = deepcopy(entry)
    out["task"] = new_task
    return out

def generate_entry(n_missing: int, m_known: int, candidates: List[DataEntry]) -> Tuple[DataEntry, DataEntry]:
    """
    Generate a tuple of data entry with specified numbers of missing and known details with slight mutation.
    """
    entry = pick_entry_with_knowns(candidates, n_missing, m_known)
    if m_known == 0:
        return entry, entry
    mutated_entry = mutate_known_details(entry)
    entry = synthesize_task_with_knowns(entry)
    mutated_entry = synthesize_task_with_knowns(mutated_entry)
    return entry, mutated_entry

def generate_dataset(
    dir: str | Path = "./data/in3/test_splits",
    out_path: str | Path = "./data/in3/test_processed",
) -> None:
    """
    Generate datasets with varying numbers of missing and known details.
    """
    dir = Path(dir)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    group_1 = out_path / "group_1"
    group_1.mkdir(parents=True, exist_ok=True)
    group_2 = out_path / "group_2"
    group_2.mkdir(parents=True, exist_ok=True)
    entries_by_missing: Dict[int, List[DataEntry]] = {}
    for n in range(6):
        path = dir / f"entries_missing_{n}.jsonl"
        if path.exists():
            with path.open("r") as f:
                entries_by_missing[n] = [json.loads(line) for line in f]
    for n_missing in range(3, 6):
        for m_known in range(6 - n_missing):
            if n_missing == 3 and m_known < 2:
                continue  # skip small configs to save time
            out_1 = group_1 / f"test_missing_{n_missing}_known_{m_known}.jsonl"
            out_2 = group_2 / f"test_missing_{n_missing}_known_{m_known}.jsonl"
            with out_1.open("w") as out_f1, out_2.open("w") as out_f2:
                candidates = entries_by_missing.get(n_missing + m_known, [])
                if not candidates:
                    print(f"No candidates for missing {n_missing} + known {m_known}, skipping.")
                    continue
                num_samples = 20
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(generate_entry, n_missing, m_known, candidates)
                        for _ in range(num_samples)
                    ]
                    for fut in as_completed(futures):
                        entry1, entry2 = fut.result()
                        out_f1.write(json.dumps(entry1) + "\n")
                        out_f2.write(json.dumps(entry2) + "\n")

if __name__ == "__main__":
    dotenv.load_dotenv()
    # split_entries("./data/in3/raw/test.jsonl", "./data/in3/test_splits")
    # split_entries("./data/in3/raw/train.jsonl", "./data/in3/train_splits")
    generate_dataset()
