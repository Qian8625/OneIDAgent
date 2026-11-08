# Question 1 — is the agent-level gain SFT or the shared-identity module?

Everything here runs on **one checkpoint**, no retraining. Only two env vars differ
between arms:

| env | set in | effect |
|---|---|---|
| `ONEID_FORCE_MODE` | `CrossModalInterpret_worker.py` | forces the chain behind the tool; wins over any `mode` the model passes |
| `ONEID_VI_CONFIG` | `ds_vi_adapter.py` (already existed) | picks the shared-identity YAML |
| `ONEID_ARM_LOG` | `CrossModalInterpret_worker.py` | one jsonl line per tool call (mode, degraded, payload) |
| `ONEID_DATASET_PATH` | `tf_eval/utils/task_utils.py` | points the rsagent task at a subset file |

All four are no-ops when unset, so the shipped behaviour is unchanged.

## Why only task 4 matters

`CrossModalInterpret` is the **only** tool that calls the DS+VI chain
(`grep -l ds_vi_adapter tool_server/` -> one file). On the test split:

```
task1  285   with CrossModalInterpret:   0
task2  285                            0
task3  278                            0
task4  281                          281  (100%)
task5  285                            0
```

So 1,133 of 1,414 trajectories are byte-identical across arms by construction —
they double as a no-op control. All the signal is in task4, n=281.

## Arms

| arm | mode | config | corresponds to |
|---|---|---|---|
| `A0_hard` | `legacy_hard` | — | conventional geometry match ("module off" proxy) |
| `A1_prior` | `candidate_graph` | — | candidate graph + physical prior, no inference (paper `S`) |
| `A3_online` | `shared_identity` | `f_online` | **what Table 1 actually ran** |
| `A3_F` | `shared_identity` | `f_ctx` | paper default `F` (contextual reliability on) |
| `A3_F_nobudget` | `shared_identity` | `f_ctx_nobudget` | `F−P` |
| `A3_F_nocap` | `shared_identity` | `f_ctx_nocap` | `J+` rung (no capacity/null projection) |
| `A3_F_nofeedback` | `shared_identity` | `f_ctx_nofeedback` | shared row with prior-only `A` |

`legacy_hard` cannot be removed: the reference trajectory requires the tool call.
The contrast is therefore *which estimator sits behind the tool*, which is exactly
the ablation the reviewers asked for ("仅改变推断模块").

## Run

```bash
export EVAL_MODEL_API_KEY=... JUDGE_API_KEY=... ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI
bash scripts/exp/run_module_ablation.sh                       # 7 arms x 281
ARMS="A0_hard A3_F" DATASET=test.jsonl \
  bash scripts/exp/run_module_ablation.sh                     # full pass, 2 arms
CUDA_VISIBLE_DEVICES=0,1 MODEL_PATH=$PWD/models/0_basemodel/Qwen3-4B-Instruct-2507 \
  RUN_TAG=base ARMS="A0_hard A3_F" bash scripts/exp/run_module_ablation.sh   # SFT-off column
```

Every arm writes `ckpt/`, `eval/`, `tools_output/`, `armlog.jsonl` under
`outputs/module_ablation/<RUN_TAG>/<ARM>/`; `manifest.jsonl` records the setup.
Re-running resumes from the ckpt (`SKIP_EXISTING=1` skips finished arms).

## Analyse

```bash
export JUDGE_API_KEY=...
python scripts/exp/analyze_module_ablation.py \
  --root outputs/module_ablation/<RUN_TAG> \
  --dataset $ONEID_DATA_ROOT/test_task4.jsonl \
  --reference A0_hard
```

Four tables: outcome per arm, paired McNemar + bootstrap CI vs the reference,
answer-content-vs-commitment confusion, module exposure. Judgements are cached in
`judge_cache.jsonl`, so re-runs are free and every arm is scored by one judge pass.

## Pre-registered reading of the result (decide before looking)

Let `Δ = A3_F − A1_prior` on task4 `answer_acc`.

* `Δ ≥ +0.03` and the bootstrap CI excludes 0 → module contribution is real;
  put table 1+2 in §5.2 and claim it.
* `|Δ| < 0.03` or CI includes 0 → report as an upper bound, move the headline to
  the inference level (Truth Acc +1.69, Holm p=0.0048) and call the agent part a
  deployability check.
* `Δ ≤ 0` → check exposure (table 4) and degradation first; a degraded arm is a
  broken run, not a null result.

## Two things the code says that the paper should say too

1. `runtime_online.yaml` ships with `reliability.contextual.enabled: false`
   ("拟合参数未随本仓库发布"), so the agent-level chain ran on the *table*
   reliability, not on the calibrated θ0 of `F`. `A3_F` is the first arm that
   matches the paper's default configuration.
2. The evaluator's E2E score is
   `0.30·final_outcome + 0.25·strict_tool_sequence + 0.20·arg_normalized_match
   + 0.15·tool_success + 0.10·robustness` (`task.py:1015`), i.e. 70 % of it is
   reference-trajectory fidelity. That is why E2E (44.90) sits far below tool-call
   F1 (78.18), and why the ablation must report `answer_acc`, not E2E.
