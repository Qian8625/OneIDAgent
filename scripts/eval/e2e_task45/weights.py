#!/usr/bin/env python3
"""权重解析：不加载到显存也能说清"这份 checkpoint 是什么、和基座差在哪"。

    python scripts/eval/e2e_task45/weights.py report --ckpt-root cpk
    python scripts/eval/e2e_task45/weights.py delta  --a cpk/base/Qwen3-4B-Instruct-2507 \
                                                     --b cpk/sft/qwen3_4b_rs_text_sft_14336
    python scripts/eval/e2e_task45/weights.py load   --path cpk/base/Qwen3-4B-Instruct-2507 --cuda 0

读的是 safetensors 头（每 shard 前 8 字节 + JSON），不打满 8 GB；``delta`` 用
``safe_open`` 按张量惰性读取，只碰需要对比的几十个张量。

结论会直接服务于 2×2 的横轴：若 SFT 只动了 transformer 层、嵌入与 norm 与基座逐位相同，
那"SFT 相对基座"的改动就是可复现的（合并后的全量权重，不是外挂 adapter）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def _header(path: Path) -> Dict[str, dict]:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size))


def _index_map(model_dir: Path) -> Dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    # 单 shard checkpoint 没有 index.json：直接由该 shard 的头部建映射。
    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) == 1:
        return {name: shards[0].name for name in _header(shards[0]) if name != "__metadata__"}
    return {}


def describe(model_dir: Path) -> Dict[str, object]:
    shards = sorted(model_dir.glob("*.safetensors"))
    dtypes: Counter = Counter()
    tensors = 0
    for shard in shards:
        for name, meta in _header(shard).items():
            if name == "__metadata__":
                continue
            tensors += 1
            dtypes[meta["dtype"]] += 1

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    index = model_dir / "model.safetensors.index.json"
    total_parameters = None
    if index.exists():
        total_parameters = json.loads(index.read_text(encoding="utf-8")).get("metadata", {}).get("total_parameters")

    tokenizer_config = {}
    tok_path = model_dir / "tokenizer_config.json"
    if tok_path.exists():
        tokenizer_config = json.loads(tok_path.read_text(encoding="utf-8"))

    template_source = "none"
    if tokenizer_config.get("chat_template"):
        template_source = "tokenizer_config"
    elif (model_dir / "chat_template.jinja").exists():
        template_source = "chat_template.jinja"

    template_sha = None
    if template_source == "tokenizer_config":
        template_sha = hashlib.sha256(tokenizer_config["chat_template"].encode()).hexdigest()[:10]
    elif template_source == "chat_template.jinja":
        template_sha = hashlib.sha256((model_dir / "chat_template.jinja").read_bytes()).hexdigest()[:10]

    size_bytes = sum(shard.stat().st_size for shard in shards)
    params = total_parameters or int(size_bytes / DTYPE_BYTES.get(list(dtypes)[0] if dtypes else "BF16", 2))
    return {
        "path": str(model_dir),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "shards": [shard.name for shard in shards],
        "bytes": size_bytes,
        "tensors": tensors,
        "dtypes": dict(dtypes),
        "parameters": params,
        "parameters_b": round(params / 1e9, 3),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": config.get("vocab_size"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "tokenizer_class": tokenizer_config.get("tokenizer_class"),
        "chat_template": template_source,
        "template_sha": template_sha,
        "eos_token": tokenizer_config.get("eos_token"),
    }


def _read(model_dir: Path, key: str, weight_map: Dict[str, str]):
    from safetensors import safe_open

    shard = weight_map.get(key)
    if shard is None:
        return None
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _diff(a, b) -> Dict[str, object]:
    import torch

    af, bf = a.float(), b.float()
    delta = bf - af
    return {
        "identical": bool(torch.equal(a, b)),
        "rel_l2": round(float(delta.norm() / (af.norm() + 1e-12)), 6),
        "max_abs": round(float(delta.abs().max()), 6),
        "mean_abs": round(float(delta.abs().mean()), 8),
    }


def delta(a_dir: Path, b_dir: Path, layers: int = 36) -> Dict[str, object]:
    map_a, map_b = _index_map(a_dir), _index_map(b_dir)
    probe = ["model.embed_tokens.weight", "model.norm.weight"]
    probe += [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(layers)]

    rows: List[Dict[str, object]] = []
    for key in probe:
        tensor_a, tensor_b = _read(a_dir, key, map_a), _read(b_dir, key, map_b)
        if tensor_a is None or tensor_b is None:
            rows.append({"tensor": key, "status": "missing"})
            continue
        rows.append({"tensor": key, **_diff(tensor_a, tensor_b)})

    layer_rows = [row for row in rows if "layers." in row["tensor"]]
    changed = [row for row in layer_rows if not row["identical"]]
    return {
        "a": str(a_dir),
        "b": str(b_dir),
        "layers_probed": len(layer_rows),
        "layers_changed": len(changed),
        "embed_identical": rows[0].get("identical"),
        "norm_identical": rows[1].get("identical"),
        "rel_l2_range": [min(row["rel_l2"] for row in changed), max(row["rel_l2"] for row in changed)]
        if changed else None,
        "rows": rows,
    }


def load_check(path: Path, cuda: str = "0") -> Dict[str, object]:
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(int(cuda) if cuda.isdigit() else 0)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16,
                                                 trust_remote_code=True).to("cuda")
    load_seconds = time.time() - started
    messages = [{"role": "system", "content": "You are a remote-sensing tool-use agent."},
                {"role": "user", "content": 'Reply with exactly one JSON object: {"thought": "ok"}'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    gen_started = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    gen_seconds = time.time() - gen_started
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    result = {"path": str(path), "load_seconds": round(load_seconds, 1),
              "gen_seconds": round(gen_seconds, 2), "sample": text.strip()[:200],
              "json_ok": text.strip().startswith("{")}
    del model
    torch.cuda.empty_cache()
    return result


def _fmt(report: Dict[str, object]) -> str:
    lines = [
        f"path          {report['path']}",
        f"arch          {report['architectures']} ({report['model_type']})",
        f"params        {report['parameters_b']} B  dtype={report['dtypes']}  tensors={report['tensors']}",
        f"size          {report['bytes'] / 1e9:.2f} GB / {len(report['shards'])} shard(s)",
        f"shape         hidden={report['hidden_size']} layers={report['num_hidden_layers']} vocab={report['vocab_size']}",
        f"embeddings    tie={report['tie_word_embeddings']} max_pos={report['max_position_embeddings']}",
        f"tokenizer     {report['tokenizer_class']} eos={report['eos_token']}",
        f"chat_template {report['chat_template']} sha={report['template_sha']}",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_report = sub.add_parser("report", help="解析一个目录下的所有 checkpoint")
    p_report.add_argument("--ckpt-root", default="cpk")
    p_report.add_argument("--json", default="")
    p_report.add_argument("--load", action="store_true", help="额外做一次真实加载+生成")
    p_report.add_argument("--cuda", default="0")

    p_delta = sub.add_parser("delta", help="逐张量对比两份权重")
    p_delta.add_argument("--a", required=True)
    p_delta.add_argument("--b", required=True)
    p_delta.add_argument("--layers", type=int, default=36)
    p_delta.add_argument("--json", default="")

    p_load = sub.add_parser("load", help="加载并生成一次")
    p_load.add_argument("--path", required=True)
    p_load.add_argument("--cuda", default="0")

    args = parser.parse_args(argv)

    if args.cmd == "report":
        root = Path(args.ckpt_root)
        model_dirs = sorted({p.parent for p in root.rglob("config.json")})
        reports = []
        for model_dir in model_dirs:
            try:
                report = describe(model_dir)
            except Exception as exc:
                print(f"[skip] {model_dir}: {type(exc).__name__}: {exc}")
                continue
            print("\n" + "=" * 70)
            print(_fmt(report))
            reports.append(report)
        if args.load:
            for report in reports:
                print("\n" + "=" * 70)
                print("[load]", report["path"])
                print(json.dumps(load_check(Path(report["path"]), args.cuda), ensure_ascii=False, indent=1))
        if args.json:
            Path(args.json).write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[json] {args.json}")
        return 0

    if args.cmd == "delta":
        result = delta(Path(args.a), Path(args.b), args.layers)
        print(f"a={result['a']}\nb={result['b']}")
        print(f"probed layers={result['layers_probed']} changed={result['layers_changed']}")
        print(f"embed identical={result['embed_identical']}  norm identical={result['norm_identical']}")
        print(f"relL2 range over changed layers={result['rel_l2_range']}")
        for row in result["rows"][:4]:
            print(" ", row)
        if args.json:
            Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[json] {args.json}")
        return 0

    print(json.dumps(load_check(Path(args.path), args.cuda), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
