"""记忆探针与 OneID 因果探针。

为什么需要它们：SFT 与 base 在"能拿到证据"之外还差"先验/记忆"，只看 2x2 无法区分
"会查证据"和"背过分布"。三个探针把"不给证据也能答对多少"量出来：

    m1 闭卷文本  无工具、无图片，只给原始问法            -> 记忆 pair/问法
    m2 换图      工具可用，图片换成另一 pair              -> 答的是不是 A 的答案
    m3 假 pair   无工具，pair_id 改成不存在的 id          -> 是否按标识符回忆

以及一个因果探针：

    splice      问题是 A，图片给 A_RGB + B_SAR（检测+OneID 可用）
                -> 若仍输出 A 的正常计数，说明它没在看证据

统一口径：所有探针都按"是否等于原始样本的 GT"打分（answer_ok），因此 m2/m3/splice
的低分恰恰说明模型在跟随证据。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.e2e_task45 import data as harness_data  # noqa: E402
from scripts.exp_1_1 import run as run_mod  # noqa: E402

FAKE_ID = "999999"


def _fake_pair_question(question: str) -> str:
    return re.sub(r"(task[45])_\d{6}", rf"\1_{FAKE_ID}", question)


def _derive(sample: Dict[str, Any], kind: str) -> Dict[str, Any]:
    derived = dict(sample)
    derived["sample_id"] = f"{sample['sample_id']}::{kind}"
    if kind == "m3":
        derived["question"] = _fake_pair_question(sample["question"])
        derived["pair_id"] = _fake_pair_question(sample["pair_id"])
    return derived


def _wrong_images(samples: List[Dict[str, Any]], index: int) -> Dict[str, str]:
    other = samples[(index + 1) % len(samples)]
    images = [(Path(harness_data.__file__).parents[0], None)]  # placeholder, replaced below
    return {"_wrong_sample": other}


def _registry_with(sample: Dict[str, Any], data_root: Path, images: List[str]) -> Dict[str, Any]:
    registry = run_mod.registry_for(sample, data_root)
    if sample["family"] == "task4":
        registry["optical_image"] = str((data_root / images[0]).resolve())
        registry["sar_image"] = str((data_root / images[1]).resolve())
    else:
        registry["time1_image"] = str((data_root / images[0]).resolve())
        registry["time2_image"] = str((data_root / images[1]).resolve())
    return registry


def run_memory(samples: List[Dict[str, Any]], cells: List[str], seed: int,
               out_dir: Path, data_root: Path, max_rounds: int = 8) -> List[Path]:
    """m1/m2/m3：每条样本一份，分别写到 probe 子目录。"""
    paths: List[Path] = []
    for cell in cells:
        weights, condition = cell.split("_", 1)

        # m1 / m3：闭卷（none），只改输入可见性/标识符
        none_cell = f"{weights}_none"
        for kind in ("m1", "m3"):
            derived = [_derive(s, kind) for s in samples]
            paths.append(run_mod.run_samples(
                none_cell, derived, seed=seed,
                trace_path=out_dir / "memory" / f"{cell}_{kind}_seed{seed}.jsonl",
                data_root=data_root, max_rounds=max_rounds,
                extra_fields={"probe": kind, "probe_cell": cell},
                registry_fn=lambda s, root=data_root: {"sample_id": s["sample_id"]},
            ))

        # m2：工具可用（det），图片换成下一个 pair
        det_cell = f"{weights}_det"
        derived = [_derive(s, "m2") for s in samples]
        index = {id(s): i for i, s in enumerate(derived)}
        wrong = {id(s): samples[(i + 1) % len(samples)] for i, s in enumerate(derived)}
        paths.append(run_mod.run_samples(
            det_cell, derived, seed=seed,
            trace_path=out_dir / "memory" / f"{cell}_m2_seed{seed}.jsonl",
            data_root=data_root, max_rounds=max_rounds,
            extra_fields={"probe": "m2", "probe_cell": cell},
            registry_fn=lambda s, root=data_root, w=wrong: _registry_with(
                s, root, w[id(s)]["images"]),
        ))
    return paths


def run_splice(samples: List[Dict[str, Any]], cells: List[str], seed: int,
               out_dir: Path, data_root: Path, max_rounds: int = 16) -> List[Path]:
    """splice：A 的问题 + A_RGB + B_SAR（仅对 task4；task5 用 time1 对调）。"""
    paths: List[Path] = []
    for cell in cells:
        derived = [_derive(s, "splice") for s in samples]
        swapped = {id(s): samples[(i + 1) % len(samples)] for i, s in enumerate(derived)}

        def registry(s, root=data_root, sw=swapped):
            other = sw[id(s)]
            registry = run_mod.registry_for(s, root)
            if s["family"] == "task4":
                registry["sar_image"] = str((root / other["images"][1]).resolve())
            else:
                registry["time2_image"] = str((root / other["images"][1]).resolve())
            return registry

        paths.append(run_mod.run_samples(
            cell, derived, seed=seed,
            trace_path=out_dir / "splice" / f"{cell}_splice_seed{seed}.jsonl",
            data_root=data_root, max_rounds=max_rounds,
            extra_fields={"probe": "splice", "probe_cell": cell},
            registry_fn=registry,
        ))
    return paths


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("memory", "splice"):
        p = sub.add_parser(name)
        p.add_argument("--data", default="/home/ubuntu/00_Data/RSTI")
        p.add_argument("--cells", default="base_det,sft_det")
        p.add_argument("--pairs", type=int, default=100)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--out", default="outputs/exp_1_1/probes")
        p.add_argument("--max-rounds", type=int, default=8)

    args = parser.parse_args(argv)
    data_root = Path(args.data).expanduser().resolve()
    samples = harness_data.load_samples(data_root / "test.jsonl", ("task4", "task5"),
                                        per_family=max(1, args.pairs // 2))
    print(f"[data] samples={len(samples)}")
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    out_dir = Path(args.out).expanduser().resolve()

    if args.cmd == "memory":
        paths = run_memory(samples, cells, args.seed, out_dir, data_root, args.max_rounds)
    else:
        paths = run_splice(samples, cells, args.seed, out_dir, data_root, args.max_rounds)
    print("[done]", ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
