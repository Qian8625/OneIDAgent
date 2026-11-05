# OneID

OneID is an agent-based interpretation system for paired optical (RGB) and SAR
remote-sensing imagery. A language-model agent drives a pool of registered tools
(detection, segmentation, change detection, GIS, plotting, ...) and a set of
skills; the cross-modal answers are produced by **DS+VI**, a shared-identity
engine that pairs RGB and SAR detections, reasons about object-level identity,
and assembles a single answer.

```
agent (app/*.py  or  tool_server/tf_eval/agent_runtime)
   │
   ├── tool_server/tool_workers/online_workers   # one HTTP worker per tool
   ├── tool_server/tool_workers/skills/plugins   # multi-tool skills
   └── DS_VI/                                    # pairing + evidence + assembly
```

## Repository layout

| Path | Contents |
|---|---|
| `app/` | Gradio application (`app/app.py`) |
| `tool_server/tool_workers/online_workers/` | Tool workers: `ObjectDetection`, `DenoDetV2`, `SmallObjectDetection`, `ChangeDetection`, `CrossModalInterpret`, `SegmentObjectPixels`, `RegionAttributeDescription`, `TextToBbox`, `AddPoisLayer`, `ComputeDistance`, `DrawBox`, `AddText`, `Plot`, `GetAreaBoundary`, `Calculator`, `Solver`, `CountGivenObject`, `ImageDescription`, `GoogleSearch`, `BaseModel`, `SkillExecutor` |
| `tool_server/tool_workers/skills/plugins/` | Skills: `cross_modal_match`, `cross_modal_task`, `object_detect_select`, `object_measure`, `object_visual_output`, `sar_evidence_extract` |
| `tool_server/tf_eval/` | Agent runtime, prompt building, evaluation tasks |
| `DS_VI/` | DS+VI engine: `models/ds_fusion` (D-S pairing), `models/shared_identity` (evidence inference), `models/cascade` (assembly), `models/cross_modal` (interpreter), `models/opt` (optional OT association) |
| `scripts/eval/` | Evaluation entry points and the frozen GT protocol (`gt_protocol.py`) |
| `scripts/tools_test/` | Smoke tests for every registered tool |
| `tests/contract/` | Contract tests for the cross-modal API and evidence pipeline |
| `docs/` | Extra developer documentation |

## Requirements

- Python 3.10
- `requirements/` splits dependencies per runtime role:
  `inference_requirements.txt` (agent + UI), `tool_server_e{1,2,3}_requirements.txt`
  (tool workers, GPU groups), `train_requirements.txt` (fine-tuning).

```bash
pip install -r requirements/inference_requirements.txt
pip install -r requirements/tool_server_e1_requirements.txt
```

Model weights are **not** distributed with this repository. Set
`CHECKPOINT_ROOT` / `LOCAL_MODEL_ROOT` (see `.env.example`) and place the
checkpoints you actually use under those directories.

## Quick start

```bash
cp .env.example .env        # then edit the paths
export ONEID_DATA_ROOT=/path/to/your/data
```

1. Start the tool service (controller + workers):

```bash
python tool_server/tool_workers/scripts/launch_scripts/start_server_local.py \
    --config tool_server/tool_workers/scripts/launch_scripts/config/all_service_example_local.yaml
```

2. Smoke-test the tools (another terminal):

```bash
python scripts/tools_test/0_tools_test_all.py --all
```

3. Launch the agent UI:

```bash
python app/app.py
```

4. Terminal chat loop (same prompt and tool loop as the UI):

```bash
python scripts/chat/chat.py \
    --pretrained /path/to/agent-model \
    --question "Use Calculator to compute 17*23+5" \
    --image ""        # omit both flags for an interactive REPL
```

`--pretrained` defaults to `$PRETRAINED_PATH`. Both the app and the chat loop
need the tool service from step 1.

5. Single cross-modal call, without the service:

```bash
python tool_server/tool_workers/online_workers/CrossModalInterpret_worker.py \
    --test-json payload.json
```

## Cross-modal interpretation (DS+VI)

`CrossModalInterpret` pairs RGB and SAR detections for one target class and
answers one of five tasks: `crosscheck_exists`, `select_largest`,
`select_highest_score`, `joint_scene`, `distance_largest_smallest`.

| `mode` | What runs | When |
|---|---|---|
| `shared_identity` (default) | D-S pairing -> shared-identity evidence -> cascade assembly | normal operation |
| `vi_only` | shared-identity graph only | D-S stage failed, or too many observations |
| `candidate_graph` | candidate graph + physical priors, no inference | diagnostics |
| `legacy_hard` | legacy geometric matching | baseline / fallback |

The evidence layer reports its verdict per object with the four-valued state of the
paper — `Confirmed`, `Weak Support`, `Conflict`, `Uncertain` — surfaced to callers as
`vi_status`.

Degrading is always explicit: the response carries `mode`, `degraded` and
`degrade_reason`. `ONEID_DEFAULT_MODE=legacy_hard` switches the default back.

Online capability is intentionally narrower than the offline experiments:
only `category`-level evidence is available (no colour/size/orientation
attributes), the temporal (static/moved) family is not part of the online path,
and registration is fixed to `identity`. These limits are stated in the tool
description and in `structured.capability`.

Configuration lives in
`tool_server/tool_workers/online_workers/configs/ds_vi/`:
`online_match.json` (detection-fusion side) and `runtime_online.yaml`
(shared-identity side, a copy of `DS_VI/configs/shared_identity/runtime_best.yaml`
with contextual reliability disabled).

## Data

The repository ships **no** imagery, detections or ground truth. Evaluation
data is expected under one directory:

```
$ONEID_DATA_ROOT/
  match_images/   task4_*.png            RGB/SAR pairs
                  0_tool_json/           {"detections": [...], ...}
                  0_truth_json/          YOLO-format truth (.txt)
  time_images/    task5_*.png, 0_tool_json/, 0_truth_json/
```

`ONEID_DATA_ROOT` defaults to `<repo>/data` (git-ignored).

## Evaluation

```bash
export ONEID_DATA_ROOT=/path/to/data
python scripts/eval/eval_multisource_detection.py \
    --targets ship --modalities cross_modal --image-size 1024 \
    --output-dir /tmp/eval_smoke
```

The evaluator writes `metrics_summary.json`, `per_class_metrics.json`,
`summary.md` and `per_sample_results.jsonl`. Ground-truth construction is frozen
in `scripts/eval/gt_protocol.py` and stamped into every run as
`gt_protocol_version`; two runs on the same input must produce identical
metrics apart from `run_id`.

## Tests

```bash
# contract tests (no pytest required)
python scripts/ci/run_contract_tests.py

# DS+VI engine tests
pip install pytest
pytest -q DS_VI/tests -k "not opt"     # opt/ needs torch

# end-to-end checks (see below)
python scripts/ci/run_e2e.py
```

### End-to-end checks

`scripts/ci/run_e2e.py` drives four independent suites; each can also run on its
own and exits non-zero on failure.

| Suite | What it proves |
|---|---|
| `e2e_static.py` | bundled CJK font renders, every sample image referenced by the UI exists, header is OneID, exactly one UI entry point |
| `e2e_worker.py` | `AddText` renders Chinese through the bundled font; `CrossModalInterpret` runs the shared-identity chain over real HTTP |
| `e2e_skill_chain.py` | skill -> controller -> tool: `CrossModalMatchSkill` resolves the interpret worker through the controller and returns VI-scored pairs |
| `e2e_ui.py` | `app/app.py` starts and serves the OneID header (needs a Python with `gradio`) |

Workers are launched with the interpreter in `ONEID_E2E_PYTHON`; the UI uses
`ONEID_E2E_UI_PYTHON`. Ports are overridable (`ONEID_E2E_*_PORT`), see
`scripts/ci/e2e_common.py`. A `gradio`-less interpreter makes the UI suite skip
instead of fail.

## Determinism

`PYTHONHASHSEED=0` is set by the launch scripts for every worker; evaluation
runs should export it too. Same input is expected to give field-identical
output across processes.

## License

MIT — see [LICENSE](LICENSE). Third-party notices are in [NOTICE.md](NOTICE.md).

## 中文摘要

OneID 是面向光学（RGB）/ SAR 双源遥感影像的智能体解译系统：大模型智能体调度
一个工具池（检测、分割、变化检测、GIS、绘图等）与若干技能，跨模态答案由 DS+VI
共享身份引擎给出（先 D-S 配对，再做对象级身份推断，最后 cascade 装配）。

仓库不包含任何影像、检测结果或真值；模型权重也不随仓库分发。评测数据按
`ONEID_DATA_ROOT` 组织，默认 `<repo>/data`。

- 快速验证：`python scripts/ci/run_contract_tests.py`
- 引擎自测：`pytest -q DS_VI/tests -k "not opt"`
- 在线能力边界：只有类别级证据，不含颜色/尺寸/朝向属性，不含时相（static/moved）判断。
