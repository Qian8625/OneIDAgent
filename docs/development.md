# Development notes

## Layout of a request

```
agent  --HTTP-->  controller  --HTTP-->  tool worker  (one process per tool)
```

Every worker exposes `POST /worker_generate`, `POST /tool_instruction`,
`POST /worker_get_status`. `CrossModalInterpret` is the only CPU-only worker and
is a good first target when debugging:

```bash
python tool_server/tool_workers/online_workers/CrossModalInterpret_worker.py \
    --no-register --port 20035
curl -s -X POST localhost:20035/worker_generate \
     -H 'Content-Type: application/json' -d @payload.json
```

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `ONEID_DATA_ROOT` | evaluation data root | `<repo>/data` |
| `ONEID_DEVICE` | GPU index (`0`, `1`, ...) or `cpu` for the UI agent model | `0` |
| `ONEID_DEFAULT_MODE` | default `CrossModalInterpret` mode | `shared_identity` |
| `ONEID_DSVI_BUDGET_SECONDS` | per-request soft budget | `120` |
| `ONEID_DSVI_MAX_OBSERVATIONS` | observation cap before `vi_only` | `800` |
| `CHECKPOINT_ROOT` | model checkpoints | `../../00_CPK` |
| `LOCAL_MODEL_ROOT` | tool-level weights | `models` |
| `LOGO_PATH` | optional UI logo | empty (hidden) |

## Tests

```bash
python scripts/ci/run_contract_tests.py          # no pytest needed
pytest -q DS_VI/tests -k "not opt"               # engine tests, torch-free
```

`tests/contract/` covers the public `CrossModalInterpret` contract, the
RGB/SAR evidence pipeline, the match/task skill status mapping, the frozen GT
protocol, and a static check that the six app variants are wired identically.

## Running the agent without the UI

```bash
python scripts/chat/chat.py --pretrained <model-dir> --question "<question>" --image ""
```

It shares the prompt and the tool loop with the UI. Keep worker outputs inside
the repository: the launcher creates `tool_server/tool_workers/tools_output`, and
direct worker starts accept `--save_path`. Gradio only serves files under its
working directory, so an artifact written outside it is rejected at the UI.

## End-to-end suites

```bash
export ONEID_E2E_PYTHON=/path/to/python-with-fastapi   # workers
export ONEID_E2E_UI_PYTHON=/path/to/python-with-gradio  # UI (optional)
export ONEID_DATA_ROOT=/path/to/data
python scripts/ci/run_e2e.py                # everything
python scripts/ci/run_e2e.py --only worker  # one suite
```

`e2e_skill_chain.py` starts a controller plus two workers and calls the skill
over HTTP, so it covers the real `skill -> controller -> tool` path rather than
importing functions directly. All suites write process logs to `.e2e_tmp/`
(git-ignored).

## Determinism

Workers are started with `PYTHONHASHSEED=0` by the launch scripts. Export it in
evaluation shells as well. Two runs on identical input must produce identical
output (the contract suite asserts this for one payload).

## Adding a tool

1. Drop `YourTool_worker.py` under `tool_server/tool_workers/online_workers/`,
   subclass `BaseToolWorker`, implement `generate()` and `get_tool_instruction()`.
2. Register it in
   `tool_server/tool_workers/scripts/launch_scripts/config/all_service_example_local.yaml`.
3. Add a case under `scripts/tools_test/`.

## Known environment quirks

- PROJ/GDAL data directories differ per environment. The evaluation scripts and
  `ComputeDistance_worker.py` probe `sys.prefix` for `rasterio/proj_data` and
  fall back to `share/proj`; override `PROJ_DATA` / `PROJ_LIB` if a worker fails
  with "proj.db not found".
- `DS_VI/tests/opt/` needs PyTorch; it is excluded with `-k "not opt"` on
  CPU-only machines.
