from __future__ import annotations
import json, re
from pathlib import Path

def normalize_slot(desc: str) -> str:
    s = desc.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    toks = [t for t in s.split() if t not in {"of","the","for","to","and","a","an","in","on"}]
    return toks[0] if toks else s.strip().split()[0]

def convert_split(raw_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open() as fin, out_path.open("w") as fout:
        for i, line in enumerate(fin):
            if not line.strip(): continue
            obj = json.loads(line)
            text = obj.get("task") or obj.get("instruction") or obj.get("query") or ""
            vague = bool(obj.get("vague", False))
            md = obj.get("missing_details") or []
            unresolved = []
            slot_keywords = {}
            if vague and isinstance(md, list):
                for item in md:
                    desc = item.get("description") if isinstance(item, dict) else str(item)
                    if not desc: continue
                    unresolved.append(desc)
                    key = normalize_slot(desc)
                    slot_keywords[key] = [key]
            tc = {
                "conversation_id": f"IN3_{raw_path.stem}_{i:05d}",
                "turn_index": 0,
                "conversation": f"User: {text}",
                "unresolved_items": unresolved,
                "resolved_items": {},
                "slot_keywords": slot_keywords
            }
            fout.write(json.dumps(tc, ensure_ascii=False) + "\n")
    print("[ok] wrote", out_path)

def main():
    base = Path("data/in3")
    raw = base / "raw"
    out = base
    pairs = []
    for split in ["train","test","dev","validation"]:
        rp = raw / f"{split}.jsonl"
        if rp.exists():
            pairs.append((rp, out / f"proactest_{split}.jsonl"))
    if not pairs:
        print("[warn] No IN3 raw files found under data/in3/raw/.")
        return 0
    for rp, op in pairs:
        convert_split(rp, op)
    return 0

if __name__ == "__main__":
    main()
