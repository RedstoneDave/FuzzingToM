from __future__ import annotations
import json, random
from pathlib import Path

SLOTS = {
    "origin": ["origin","from","depart","leaving"],
    "dest": ["to","destination","going"],
    "date": ["when","date","day","time"],
    "budget": ["budget","price","cost","spend"],
    "seat": ["seat","class","economy","business"],
}

def synth_case(cid: str, t: int, unresolved):
    conv = [
        {"role":"user","text":"Book me a flight"},
        {"role":"agent","text":"Sure—I can help. What details can you share?"},
        {"role":"user","text":"I want to go to Tokyo in November."}
    ]
    resolved = {"dest": "Tokyo", "month": "November"}
    slot_keywords = {**SLOTS, "month": ["month","when","date"]}
    return {
        "conversation_id": cid,
        "turn_index": t,
        "conversation": conv,
        "unresolved_items": list(unresolved),
        "resolved_items": resolved,
        "slot_keywords": slot_keywords,
    }

def main():
    random.seed(0)
    out_dir = Path("data/toy")
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train","dev","test"]:
        path = out_dir / f"{split}.jsonl"
        with path.open("w") as f:
            for i in range(200 if split=="train" else 64):
                cid = f"toy-{split}-{i:04d}"
                ks = list(SLOTS.keys()); random.shuffle(ks)
                unresolved = ks[: random.randint(1,3)]
                obj = synth_case(cid, 1, unresolved)
                f.write(json.dumps(obj) + "\n")
    print("[ok] toy dataset written to data/toy/")

if __name__ == "__main__":
    main()
