#!/usr/bin/env python3
"""Score the Question-1 module ablation: final-answer accuracy, paired.

The run's own metric file only reports answer_acc aggregated per task family,
and its LLM_scores.jsonl has no sample id.  This script re-scores the *saved
predictions* (ckpt jsonl, keyed by idx) against the dataset gold answer with the
same judge prompt used by tool_server/tf_eval/tasks/rsagent/task.py, so every
arm is judged once, identically, and can be paired sample by sample.

Outputs (markdown + json):
  1. per-arm outcome table  (answered rate, answer_acc, acc@0.5, selective acc)
  2. paired deltas vs a reference arm (McNemar exact on acc@0.5, bootstrap CI)
  3. answer-content vs commitment confusion table
  4. module exposure from armlog.jsonl (how often CrossModalInterpret ran and
     whether it degraded) -- a low exposure rate invalidates the comparison

Usage
    export JUDGE_API_KEY=...
    python scripts/exp/analyze_module_ablation.py \
        --root outputs/module_ablation/20261107_1400 \
        --dataset $ONEID_DATA_ROOT/test_task4.jsonl \
        --reference A0_hard
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import sys
import urllib.request

# --- judge prompt: import the real one when the repo is importable --------- #
EVAL_PROMPT_FALLBACK = None
try:  # pragma: no cover - depends on the run environment
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from tool_server.tf_eval.tasks.rsagent.task import EVAL_PROMPT as EVAL_PROMPT  # type: ignore

    PROMPT_SOURCE = "tool_server.tf_eval.tasks.rsagent.task"
except Exception:  # pragma: no cover
    EVAL_PROMPT = """
You are an evaluation assistant responsible for measuring the Answer Accuracy Score (a value between 0 and 1) for a geospatial agent's prediction.
You are given: A Question, A Ground Truth Answer, A Predicted Answer.
Judge numerical agreement within +-10%, and semantic equivalence for categorical answers.
Reply strictly as {"Score": <float 0..1>, "Justification": "<1-2 sentences>"}.
"""
    PROMPT_SOURCE = "inline-fallback (WARNING: may drift from task.py)"

JUDGE_URL = os.environ.get("JUDGE_API_URL", "https://api.deepseek.com/chat/completions")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v4-flash")

# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #


def load_dataset(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_answer(rec: dict) -> str:
    ans = ""
    for msg in rec.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        try:
            obj = json.loads(msg.get("content", ""))
        except Exception:
            continue
        for act in obj.get("actions", []) or []:
            if act.get("name") == "Terminate":
                value = (act.get("arguments") or {}).get("ans")
                if isinstance(value, str) and value.strip():
                    ans = value.strip()
    return ans


def question_text(rec: dict) -> str:
    for msg in rec.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "").replace("<image>", "").strip()
    return ""


def task_family(rec: dict) -> str:
    imgs = rec.get("images") or [""]
    name = pathlib.PurePosixPath(imgs[0]).name
    return name.split("_")[0] if "_" in name else "unknown"


def load_ckpt(path: pathlib.Path) -> dict:
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            res = rec.get("results") or {}
            idx = res.get("idx")
            if idx is None:
                continue
            out[idx] = res  # last write wins (resume appends)
    return out


# --------------------------------------------------------------------------- #
# judging
# --------------------------------------------------------------------------- #
class Judge:
    def __init__(self, cache_path: pathlib.Path):
        self.cache_path = cache_path
        self.cache: dict[str, float] = {}
        self.calls = 0
        self.hits = 0
        if cache_path.exists():
            with cache_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    self.cache[rec["k"]] = float(rec["s"])
        self._fh = cache_path.open("a", encoding="utf-8")

    @staticmethod
    def key(question: str, gt: str, pred: str) -> str:
        return hashlib.sha1(f"{JUDGE_MODEL}\x00{question}\x00{gt}\x00{pred}".encode()).hexdigest()

    def score(self, question: str, gt: str, pred: str) -> float:
        pred = (pred or "").strip()
        if not pred:
            return 0.0
        k = self.key(question, gt, pred)
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        prompt = (
            f"{EVAL_PROMPT}\n\nQuestion: {question}\n"
            f"Ground Truth Answer: {gt}\nPredicted Answer: {pred}\n"
        )
        payload = {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 150,
        }
        req = urllib.request.Request(
            JUDGE_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('JUDGE_API_KEY', '')}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
        score = extract_score(text)
        self.calls += 1
        self.cache[k] = score
        self._fh.write(json.dumps({"k": k, "s": score, "q": question[:200], "p": pred[:300]},
                                  ensure_ascii=False) + "\n")
        self._fh.flush()
        return score

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def extract_score(text: str) -> float:
    head = text.split("Justification")[0]
    num = ""
    for ch in head:
        if ch.isdigit() or ch in ".-+":
            num += ch
        elif num:
            break
    try:
        return max(0.0, min(1.0, float(num.strip("."))))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_diff(a: list[float], b: list[float], iters: int = 10000, seed: int = 42):
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="OUT_ROOT of run_module_ablation.sh")
    ap.add_argument("--dataset", required=True, help="same jsonl the arms were run on")
    ap.add_argument("--reference", default="A0_hard")
    ap.add_argument("--threshold", type=float, default=0.5, help="acc@threshold for paired tests")
    ap.add_argument("--out", default=None, help="write markdown here too")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    dataset = load_dataset(pathlib.Path(args.dataset))
    gold = {f"rsagent_{i}": gold_answer(rec) for i, rec in enumerate(dataset)}
    question = {f"rsagent_{i}": question_text(rec) for i, rec in enumerate(dataset)}
    family = {f"rsagent_{i}": task_family(rec) for i, rec in enumerate(dataset)}

    judge = Judge(root / "judge_cache.jsonl")

    arms = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not arms:
        print(f"no arm directories under {root}", file=sys.stderr)
        return 2

    per_arm: dict[str, dict] = {}
    for arm in arms:
        ckpt = load_ckpt(root / arm / "ckpt" / f"{root.name}_{arm}_rsagent_e2e_ckpt.jsonl")
        if not ckpt:  # tolerate a different RESULT_NAME prefix
            candidates = sorted((root / arm / "ckpt").glob("*.jsonl"))
            if candidates:
                ckpt = load_ckpt(candidates[0])
        rows = {}
        for idx, gt in gold.items():
            res = ckpt.get(idx, {})
            pred = res.get("final_answer") or ""
            pred = pred.strip() if isinstance(pred, str) else ""
            score = judge.score(question[idx], gt, pred) if gt else 0.0
            rows[idx] = {
                "family": family[idx],
                "answered": bool(pred),
                "score": score,
                "correct": score >= args.threshold,
                "cmi": sum(
                    1
                    for c in (res.get("tool_cfg") or [])
                    if isinstance(c, dict) and c.get("API_name") == "CrossModalInterpret"
                ),
            }
        per_arm[arm] = rows

    judge.close()
    print(f"[judge] prompt={PROMPT_SOURCE} calls={judge.calls} cache_hits={judge.hits}")

    # ---- table 1: outcome per arm ----
    lines = ["## 1. Outcome per arm", "",
             "| arm | n | answered | answer_acc | acc@%.2f | selective acc | empty |" % args.threshold,
             "|---|---|---|---|---|---|---|"]
    for arm in arms:
        rows = per_arm[arm]
        n = len(rows)
        answered = sum(r["answered"] for r in rows.values())
        mean = sum(r["score"] for r in rows.values()) / n
        acc = sum(r["correct"] for r in rows.values()) / n
        sel = (sum(r["correct"] for r in rows.values() if r["answered"]) / answered) if answered else 0.0
        lines.append(f"| {arm} | {n} | {answered / n:.3f} | {mean:.4f} | {acc:.4f} | {sel:.4f} | "
                     f"{1 - answered / n:.3f} |")

    # ---- table 2: paired vs reference ----
    ref = args.reference if args.reference in per_arm else arms[0]
    lines += ["", f"## 2. Paired comparison vs `{ref}`", "",
              "| arm | d answer_acc | 95% CI | McNemar b/c | p |", "|---|---|---|---|---|"]
    for arm in arms:
        if arm == ref:
            continue
        a = [per_arm[arm][i]["score"] for i in gold]
        b = [per_arm[ref][i]["score"] for i in gold]
        lo, hi = bootstrap_diff(a, b)
        bb = sum(1 for i in gold if per_arm[arm][i]["correct"] and not per_arm[ref][i]["correct"])
        cc = sum(1 for i in gold if per_arm[ref][i]["correct"] and not per_arm[arm][i]["correct"])
        p = mcnemar_exact(bb, cc)
        lines.append(f"| {arm} | {sum(a) / len(a) - sum(b) / len(b):+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | {bb}/{cc} | {p:.4f} |")

    # ---- table 3: answer content x commitment ----
    lines += ["", "## 3. Answer content vs commitment", "",
              "| arm | committed+correct | committed+wrong | silent, answerable | silent, gold empty |",
              "|---|---|---|---|---|"]
    for arm in arms:
        rows = per_arm[arm]
        cc_ = sum(1 for i in gold if rows[i]["answered"] and rows[i]["correct"])
        cw = sum(1 for i in gold if rows[i]["answered"] and not rows[i]["correct"])
        sa = sum(1 for i in gold if not rows[i]["answered"] and gold[i])
        se = sum(1 for i in gold if not rows[i]["answered"] and not gold[i])
        lines.append(f"| {arm} | {cc_} | {cw} | {sa} | {se} |")

    # ---- table 4: module exposure ----
    # Exposure from the saved trajectories is the gate that matters: if an arm
    # never calls CrossModalInterpret, the arm cannot differ from any other.
    lines += ["", "## 4. Module exposure", "",
              "| arm | 调用 CMI 的轨迹占比 | 平均调用数 | armlog 调用数 | degraded | 主要原因 |",
              "|---|---|---|---|---|---|"]
    for arm in arms:
        rows = per_arm[arm]
        covered = sum(1 for i in gold if rows[i]["cmi"] > 0)
        mean_calls = sum(rows[i]["cmi"] for i in gold) / max(1, len(rows))
        log = root / arm / "armlog.jsonl"
        calls = degraded = 0
        reasons: dict[str, int] = {}
        if log.exists():
            with log.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    calls += 1
                    if rec.get("degraded"):
                        degraded += 1
                        reasons[str(rec.get("degrade_reason"))] = reasons.get(
                            str(rec.get("degrade_reason")), 0) + 1
        top = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else "-"
        lines.append(f"| {arm} | {covered / max(1, len(rows)):.3f} | {mean_calls:.2f} | "
                     f"{calls} | {degraded} | {top} |")

    lines += ["", f"_judged with `{JUDGE_MODEL}`; paired tests need identical samples per arm._"]
    report = "\n".join(lines)
    print()
    print(report)

    payload = {
        "reference": ref,
        "threshold": args.threshold,
        "prompt_source": PROMPT_SOURCE,
        "arms": {
            arm: {
                "n": len(rows),
                "answered_rate": sum(r["answered"] for r in rows.values()) / len(rows),
                "answer_acc": sum(r["score"] for r in rows.values()) / len(rows),
                "acc_at_threshold": sum(r["correct"] for r in rows.values()) / len(rows),
                "cmi_coverage": sum(1 for r in rows.values() if r["cmi"] > 0) / len(rows),
                "cmi_mean_calls": sum(r["cmi"] for r in rows.values()) / len(rows),
            }
            for arm, rows in per_arm.items()
        },
    }
    (root / "module_ablation_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    if args.out:
        pathlib.Path(args.out).write_text(report + "\n", encoding="utf-8")
    print(f"\n[ok] {root / 'module_ablation_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
