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
RGB/SAR evidence pipeline, the frozen GT
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

All suites write process logs to `.e2e_tmp/` (git-ignored).

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


## Workbook attribute replay

Replay is a **hybrid** run: detection is live, attributes come from the frozen
workbook. `ObjectDetection` / `DenoDetV2` produce the boxes, labels and scores
through the normal registry gate (exactly like online); the workbook contributes
only the requested target's `subtype` / `color` / `size_bin` / `orientation`
(plus the acquisition `visibility` context). No box, score, pairing, change
label or selection is read from the workbook. Start the worker with:

```bash
export ONEID_ENABLE_WORKBOOK_REPLAY=1
export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI
```

The server reads only tool_task4_property.xlsx or tool_task5_property.xlsx from
that root. An optional ONEID_REPLAY_CONFIG server JSON allowlists root-relative
paths with a datasets mapping, for example
`{"datasets": {"task4": "tool_task4_property.xlsx", "task5": "tool_task5_property.xlsx"}}`.
The request cannot supply a workbook path or override inference configuration.

Run the detectors on the pair's images first, then request:

```json
{"source_mode":"workbook_replay","dataset_id":"task4","family":"match","pair_id":"task4_000001","target":"harbor","task":"joint_scene","output_mode":"full","rgb_detections":[...],"sar_detections":[...],"evidence_provenance":{...}}
```

Task5 uses the same shape with `dataset_id=task5`, `family=time` and
`task=temporal_changes` (the two detection arrays carry the time1 / time2 boxes).

Binding rule: an `instance_id` ends with the 0-based position of the box in the
raw detector JSON order for its view (`task4_000001:rgb:0007`), and the staging
step keeps that same position as `detection["index"]`. A live box therefore
binds to the workbook row with the same `(view, index)` — positional binding, no
geometric threshold. The frozen row bbox is kept only to *audit* the binding:
`replay_diagnostics.attribute_binding` reports bound / unbound counts, the min
IoU, how many bound boxes fall below 0.5 IoU, and how the snapshot's row count
compares with the live payload (`snapshot_rows`, `live_payload`,
`index_space_matches` — equal lengths do not prove the order is shared, but a
difference proves it is not). A run where nothing binds by index, or where no
bound box overlaps at all, fails instead of mis-attributing attributes.

Only rows of the requested target are read. Other classes are skipped without
being validated — their category vocabulary and attribute values are not this
mode's business — while box and dimension checks stay global, because they tie
the whole sheet to the pair's rasters. Identities, is_real_object, association_score and temporal answers are never
injected, and no truth workbook is opened. Empty attributes do not become
unknown claims — an empty attribute row, or a live box with no workbook row at
its index, simply carries no attribute evidence. Visibility is recorded as
observation context, but contextual theta is disabled. Fixed reliability
settings are experimental priors, not calibrated claims about these workbook
versions. Calibrate on a separate split before reporting results.

Outputs identify the workbook SHA256, dataset/pair, live detection counts,
binding counts and frozen configuration hashes. Full mode includes the
instance-to-observation binding map, attribute claims, context and attribute
posteriors. Time results derive change labels from the live t0/t1 geometry with
identity registration; inputs must already be aligned. Missing detectors,
target-less detections, a drifted snapshot or malformed attributes fail
explicitly, never silently fall back to an attribute-free estimator.
