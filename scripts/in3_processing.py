from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import random
import json
from copy import deepcopy
from typing import Dict, List, NotRequired, Tuple, TypedDict
import dotenv
import requests

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

def pick_entry_with_knowns(candidates: List[DataEntry], n_missing: int, m_known: int) -> Tuple[DataEntry, List[KnownDetail]]:
    """
    Sample a datum whose total details are n_missing + m_known, then mark m_known as known (g_a analogue).
    Returns the mutated entry (with at least n_missing remaining) and the list of known details with chosen values.
    """
    # path = os.path.join(dir, f"entries_missing_{n_missing + m_known}.jsonl")
    # if not os.path.exists(path):
    #     raise ValueError(f"No entries with {n_missing + m_known} missing details found in {dir}.")
    # with open(path, "r") as f:
    #     candidates: List[DataEntry] = [json.loads(line) for line in f]
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
    return base, known


def mutate_known_details(entry: DataEntry) -> List[KnownDetail]:
    """
    Mutate known details by switching to a different available option when possible (g_m analogue).
    """
    mutated: List[KnownDetail] = []
    for kd in entry.get("known_details", []):
        options = kd["source_detail"]["options"]
        current = kd["value"]
        alternatives = [opt for opt in options if opt != current]
        new_val = random.choice(alternatives) if alternatives else current
        mutated.append({**kd, "value": new_val})
    entry["known_details"] = mutated
    return mutated

rewrite_prompt = '''
You are a task polishing expert. Your job is to integrate detailed requirements into a vague task description to make it clearer and more specific. In this task, you will be provided with a vague task description and a list of missing details that need to be incorporated into the task. Your goal is to rewrite the task description by seamlessly integrating the provided missing details, ensuring that the final task description is comprehensive and unambiguous. You should not add any new information beyond what is provided in the missing details, and you should maintain the original intent of the vague task description. You should only output the rewritten task description without any additional commentary or explanation.
Here is the vague task description and the missing details:
'''

def _rewrite_with_llm(prompt: str) -> str:
    """
    Try using the OpenAI client from proactest.llms; fallback to empty string on failure.
    """
    try:
        response = requests.post(
            url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen/qwen3-4b:free",
                "messages": [
                    {"role": "system", "content": rewrite_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
            },
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

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
    entries_by_missing: Dict[int, List[DataEntry]] = {}
    for n in range(1, 6):
        path = dir / f"entries_missing_{n}.jsonl"
        if path.exists():
            with path.open("r") as f:
                entries_by_missing[n] = [json.loads(line) for line in f]
    for n_missing in range(1, 6):
        for m_known in range(6 - n_missing):
            out = out_path / f"test_missing_{n_missing}_known_{m_known}.jsonl"
            with out.open("w") as out_f:
                candidates = entries_by_missing.get(n_missing + m_known, [])
                if not candidates:
                    print(f"No candidates for missing {n_missing} + known {m_known}, skipping.")
                    continue
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for _ in range(20):  # generate 20 samples per configuration
                        futures.append(executor.submit(pick_entry_with_knowns, candidates, n_missing, m_known))
                    for future in futures:
                        entry, knowns = future.result()
                        if m_known > 0:
                            entry_with_task = synthesize_task_with_knowns(entry)
                        else:
                            entry_with_task = entry
                        out_f.write(json.dumps(entry_with_task) + "\n")

if __name__ == "__main__":
    dotenv.load_dotenv()
    # split_entries("./data/in3/raw/test.jsonl", "./data/in3/test_splits")
    # split_entries("./data/in3/raw/train.jsonl", "./data/in3/train_splits")
    generate_dataset()
