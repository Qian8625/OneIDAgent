#!/usr/bin/env python3
"""Write the task-4 subset of RSTIBench test.jsonl.

Only task 4 (optical-SAR cross-modal interpretation) can be affected by the
inference-module ablation: 281/281 task-4 reference trajectories call
CrossModalInterpret, and 0/1133 of the other four task groups do.  Running the
arms on this subset is therefore half the cost of a full pass with none of the
statistical power lost.

idx note: task.py re-indexes by enumeration (idx = "rsagent_<i>"), so the
subset gets fresh indices.  All arms share the same subset file, so idx stays
comparable across arms; join gold by position, never across different files.

Usage
-----
    python scripts/exp/make_task4_subset.py \
        --src $ONEID_DATA_ROOT/test.jsonl \
        --out $ONEID_DATA_ROOT/test_task4.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def is_task4(rec: dict) -> bool:
    imgs = rec.get("images") or [""]
    return "task4" in pathlib.PurePosixPath(imgs[0]).name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N rows (smoke test)")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    rows = []
    total = 0
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            if is_task4(rec):
                rows.append(line)
                if args.limit and len(rows) >= args.limit:
                    break

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[ok] {out}  kept {len(rows)}/{total} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
