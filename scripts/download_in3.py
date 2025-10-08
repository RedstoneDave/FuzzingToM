from __future__ import annotations
import sys, json
from pathlib import Path

def main():
    out_dir = Path("data/in3/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except Exception as e:
        print("[warn] Install `datasets` or place files manually in data/in3/raw/", file=sys.stderr)
        return 0

    candidates = ["hbx/IN3", "IN3", "in3"]
    ds, last_err = None, None
    for name in candidates:
        try:
            ds = load_dataset(name)
            print(f"[ok] loaded dataset: {name}")
            break
        except Exception as e:
            last_err = e
    if ds is None:
        print(f"[warn] could not load IN3 via HF: {last_err}", file=sys.stderr)
        return 0

    for split in ["train", "test", "validation", "dev"]:
        if split in ds:
            path = out_dir / f"{split}.jsonl"
            with path.open("w") as f:
                for ex in ds[split]:
                    f.write(json.dumps(ex) + "\n")
            print("[ok] wrote", path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
