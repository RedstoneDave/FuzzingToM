from __future__ import annotations
import os
import random
from copy import deepcopy
from typing import List, Sequence, Tuple, TypedDict


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
    known_details: List[KnownDetail]

def pick_entry_with_knowns(entries: Sequence[DataEntry], n_missing: int, m_known: int) -> Tuple[DataEntry, List[KnownDetail]]:
    """
    Sample a datum whose total details are n_missing + m_known, then mark m_known as known (g_a analogue).
    Returns the mutated entry (with only n_missing remaining) and the list of known details with chosen values.
    """
    candidates = [d for d in entries if len(d["missing_details"]) >= n_missing + m_known]
    if not candidates:
        raise ValueError("No entries have enough missing details for the requested split.")
    base = deepcopy(random.choice(candidates))
    total_details = base["missing_details"]
    chosen_known_idx = set(random.sample(range(len(total_details)), m_known)) if m_known else set()
    known: List[KnownDetail] = []
    remaining: List[Detail] = []
    for idx, det in enumerate(total_details):
        if idx in chosen_known_idx:
            value = random.choice(det["options"]) if det["options"] else ""
            known.append({"description": det["description"], "value": value, "source_detail": det})
        else:
            remaining.append(det)
    base["missing_details"] = remaining[:n_missing]
    base["vague"] = len(base["missing_details"]) > 0
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

def _rewrite_with_llm(prompt: str) -> str:
    """
    Try using the OpenAI client from proactest.llms; fallback to empty string on failure.
    """
    try:
        from proactest.llms.openai_client import OpenAIClient
        model = os.getenv("PROACTEST_LLM_MODEL", "gpt-4o-mini")
        client = OpenAIClient(model=model)
        resp = client.chat([{"role": "system", "content": "Rewrite tasks with provided details embedded."}, {"role": "user", "content": prompt}], temperature=0.2)
        return resp.get("text", "").strip()
    except Exception:
        return ""

def synthesize_task_with_knowns(entry: DataEntry) -> DataEntry:
    """
    Integrate known details into a task description using a provided LLM callable; fallback to appending facts.
    """
    prompt = [
        "Rewrite the task to explicitly include the provided known details.",
        "Keep the intent and category unchanged.",
        "Output only the rewritten task text.",
        "",
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
