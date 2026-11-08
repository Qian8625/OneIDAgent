# Q1 执行清单 — 把"模块贡献"从 SFT 里分离出来

> 对应审稿问题：*OneID-Agent 在 RSTIBench 上微调，基线是否也训练了？提升是微调带来的还是共享身份推断带来的？*
>
> 本清单的做法：**同一份 SFT checkpoint，只换 `CrossModalInterpret` 背后的估计器**，其它（prompt / 工具 schema / 解码参数 / 最大轮数 / 数据集）全部冻结。不重训。
>
> 产出：论文新增 Table α（模块消融）、Table β（答案内容 × 承诺）、Table γ（Table 1 定位声明）。

---

## 0. 一分钟速览

| 项 | 值 |
|---|---|
| 仓库 | `~/01_Code/03_OneIDAgent` |
| 数据根 | `/home/ubuntu/00_Data/RSTI` |
| 被测 checkpoint | `models/0_sft_train/qwen3_sft_text_0614`（= OneID-Agent） |
| 基座（补 SFT-off 列用） | `models/0_basemodel/Qwen3-4B-Instruct-2507` |
| 主实验规模 | 7 arm × 281 条（task4 子集） |
| 能否影响结果的任务组 | **只有 task4**（281/281 参考轨迹调用 `CrossModalInterpret`；其余 1,133 条为 0） |
| 是否重训 | 否 |
| 预计总耗时 | 约 1.5 个工作日（含排队） |

## 状态跟踪

| Step | 内容 | 完成 |
|---|---|---|
| 0 | 环境与资产自检 | ☐ |
| 1 | 生成 5 个 arm 配置 + task4 子集 | ☐ |
| 2 | 冒烟：5 条 × 2 臂，验证强制 mode 与通路 | ☐ |
| 3 | 主实验 A 批：`A0_hard` `A1_prior` `A3_F` | ☐ |
| 4 | 主实验 B 批：4 个 config 档 | ☐ |
| 5 | 质检 4 道 gate | ☐ |
| 6 | 判分与统计（出 4 张表） | ☐ |
| 7 | 判定分支 + 论文落点 | ☐ |
| 8 | （可选）全量复核 + SFT-off 列 | ☐ |

---

## Step 0 — 环境与资产自检（10 min）

```bash
cd ~/01_Code/03_OneIDAgent
export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI

python -c "import rasterio, yaml, datasets; print('deps ok')"
wc -l $ONEID_DATA_ROOT/test.jsonl
test -f tool_server/tf_eval/scripts/configs/accelerate_config/deepspeed.yaml && echo "accelerate cfg ok"
test -d models/0_sft_train/qwen3_sft_text_0614 && echo "sft ckpt ok"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo "EVAL=${EVAL_MODEL_API_KEY:-MISSING} JUDGE=${JUDGE_API_KEY:-MISSING}" | sed -E 's/(sk-)[A-Za-z0-9]+/\1***/g'
git status --short | head
```

**预期输出**

```
deps ok
1414 /home/ubuntu/00_Data/RSTI/test.jsonl
accelerate cfg ok
sft ckpt ok
0, 1024 MiB, 24564 MiB
...
EVAL=sk-*** JUDGE=sk-***
 M tool_server/tf_eval/utils/task_utils.py
 M tool_server/tool_workers/online_workers/CrossModalInterpret_worker.py
?? scripts/exp/
```

**判据**

| # | 判据 | 通过 | 不通过 → 动作 |
|---|---|---|---|
| 0.1 | `deps ok` | 继续 | 装依赖；`rasterio` 缺会直接让 `eval_e2e.sh` 退出 |
| 0.2 | `test.jsonl` 恰好 1414 行 | 继续 | 数据根不对，回查 `ONEID_DATA_ROOT` |
| 0.3 | 至少 1 张卡空闲显存 > 30 GB | 继续 | 改 `CUDA_VISIBLE_DEVICES` |
| 0.4 | 两个 key 都非 `MISSING` | 继续 | `export` 后重跑本步 |
| 0.5 | git 状态只含本方案的 2 个 M + `scripts/exp/` | 继续 | 有别的改动 → 先 `git stash`，避免污染 |

---

## Step 1 — 生成 arm 配置与子集（5 min）

```bash
cd ~/01_Code/03_OneIDAgent
export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI

python scripts/exp/make_arm_configs.py
python scripts/exp/make_task4_subset.py --src $ONEID_DATA_ROOT/test.jsonl \
    --out $ONEID_DATA_ROOT/test_task4.jsonl
python scripts/exp/make_task4_subset.py --src $ONEID_DATA_ROOT/test.jsonl \
    --out $ONEID_DATA_ROOT/test_task4_smoke.jsonl --limit 5
cat scripts/exp/arms/manifest.json
```

**预期输出**

```
[ok] .../scripts/exp/arms/f_online.yaml
[ok] .../scripts/exp/arms/f_ctx.yaml
[ok] .../scripts/exp/arms/f_ctx_nobudget.yaml
[ok] .../scripts/exp/arms/f_ctx_nocap.yaml
[ok] .../scripts/exp/arms/f_ctx_nofeedback.yaml
[ok] .../scripts/exp/arms/manifest.json
[ok] /home/ubuntu/00_Data/RSTI/test_task4.jsonl  kept 281/1414 rows
[ok] /home/ubuntu/00_Data/RSTI/test_task4_smoke.jsonl  kept 5/... rows
```

**判据**

| # | 判据 | 通过 | 不通过 → 动作 |
|---|---|---|---|
| 1.1 | 5 个 yaml + manifest 生成 | 继续 | 看报错里的 `config path not found: <dotted>`，说明 base YAML 字段名变了 |
| 1.2 | task4 子集 **恰好 281 行** | 继续 | 行数不符 → 数据版本不对，**停止** |
| 1.3 | 冒烟子集 5 行 | 继续 | — |
| 1.4 | `f_ctx.yaml` 里 `contextual.enabled: true` 且 `theta_path` 非 null | 继续 | 手工核对 `diff scripts/exp/arms/f_ctx.yaml tool_server/tool_workers/online_workers/configs/ds_vi/runtime_online.yaml` |

---

## Step 2 — 冒烟：5 条 × 2 臂（20–40 min）

目的是**在烧掉半天算力之前**证明两件事：强制 mode 真的生效；模块真的能改变最终答案。

```bash
cd ~/01_Code/03_OneIDAgent
export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI
# 注意：这两个 key 必须是真的（下面只是占位），不要覆盖已有的 export
: "${EVAL_MODEL_API_KEY:?please export first}"
: "${JUDGE_API_KEY:?please export first}"

RUN_TAG=smoke DATASET=test_task4_smoke.jsonl \
ARMS="A0_hard A3_F" CUDA_VISIBLE_DEVICES=3,4,5 \
bash scripts/exp/run_module_ablation.sh
```

**预期输出（尾部）**

```
run_tag : smoke
dataset : /home/ubuntu/00_Data/RSTI/test_task4_smoke.jsonl  (5 samples)
---- A0_hard  mode=legacy_hard  config=none ----
... ProcessedSamples / Metrics: {...} ...
---- A3_F  mode=shared_identity  config=f_ctx ----
...
all arms finished.  results under outputs/module_ablation/smoke
```

**自检命令**

```bash
OUT=outputs/module_ablation/smoke
for A in A0_hard A3_F; do
  echo "== $A =="
  wc -l $OUT/$A/ckpt/*.jsonl
  wc -l $OUT/$A/armlog.jsonl
  python - <<PY
import json,glob,collections
f=glob.glob("$OUT/$A/armlog.jsonl")
c=collections.Counter(); d=0
for l in open(f[0]):
    r=json.loads(l); c[r["effective_mode"]]+=1; d+=bool(r["degraded"])
print("effective_mode:",dict(c),"degraded:",d)
PY
  python - <<PY
import json,glob
f=sorted(glob.glob("$OUT/$A/ckpt/*.jsonl"))[0]
for l in open(f):
    r=json.loads(l)["results"]
    print(r["idx"], "|", (r.get("final_answer") or "")[:80].replace("\n"," "))
PY
done
```

**判据**

| # | 判据 | 通过 | 不通过 → 动作 |
|---|---|---|---|
| 2.1 | 两臂 ckpt 都是 **5 行** | 继续 | <5 → 有样本崩溃，看 `logs/eval_e2e_*.log` |
| 2.2 | `A0_hard` 的 `armlog` 里 `effective_mode` **只有** `legacy_hard`；`A3_F` 只有 `shared_identity` | 继续 | 出现混合值 → `ONEID_FORCE_MODE` 没传进 worker（检查是否被 `unset`，或 worker 在别的进程里） |
| 2.3 | 两臂 `degraded` 都是 0 | 继续 | 非 0 → 记下 `degrade_reason`，走附录 E3 |
| 2.4 | 两臂 5 条 `final_answer` **至少 1 条不同** | 继续 | 5 条全同 → 走 **E1**（先扩到 30 条再判；仍全同说明模块不在答案通路上，**停**，别跑主实验） |
| 2.5 | `armlog` 行数 ≥ 4（5 条里多数调用了该工具） | 继续 | 0 行 → 模型没调用工具，检查 prompt 版本；`armlog` 缺失但 ckpt 正常 → 检查 `ONEID_ARM_LOG` 是否被覆盖 |

> 说明：`armlog` 每行是一次工具调用，不含样本 id；**准确的"调用覆盖率"由 Step 6 从 ckpt 的 `tool_cfg` 统计**，这里只做存在性检查。

---

## Step 3 — 主实验 A 批：3 臂 × 281（约 3 h）

只跑三臂先出核心对比（模块关 / 先验 / 完整），确认效应方向后再投 B 批。

```bash
cd ~/01_Code/03_OneIDAgent
export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI

# 前台跑（3h 左右；建议放进 tmux：tmux new -s q1）
RUN_TAG=q1 DATASET=test_task4.jsonl \
ARMS="A0_hard A1_prior A3_F" CUDA_VISIBLE_DEVICES=3,4,5 \
bash scripts/exp/run_module_ablation.sh
```

**预期输出**

```
run_tag : q1
model   : .../models/0_sft_train/qwen3_sft_text_0614
dataset : /home/ubuntu/00_Data/RSTI/test_task4.jsonl  (281 samples)
out     : .../outputs/module_ablation/q1
arms    : A0_hard A1_prior A3_F
```

**快速自检（每臂跑完即查，别等三臂全完）**

```bash
for A in A0_hard A1_prior A3_F; do
  printf "%-10s ckpt=%s\n" "$A" "$(wc -l < outputs/module_ablation/q1/$A/ckpt/*.jsonl 2>/dev/null)"
done
python -c "
import json,glob
for f in sorted(glob.glob('outputs/module_ablation/q1/*/eval/*.json')):
    d=json.load(open(f)); h=d.get('headline_metrics',{})
    print(f.split('/')[-3], 'answer_acc=',h.get('answer_acc'),
          'overall=',h.get('overall_score'), 'steps=',h.get('avg_steps_taken'))
    print('   families:', list(d.get('task_family_metrics',{}).keys()))
"
```

**判据**

| # | 判据 | 通过 | 不通过 → 动作 |
|---|---|---|---|
| 3.1 | 每臂 ckpt ≥ 281 行（等于须为 281） | 继续 | 少于 → 重跑同一命令（`SKIP_EXISTING=1` 会从 ckpt 续跑，不会重算已完成样本） |
| 3.2 | 每臂 `eval/*.json` 存在且 `headline_metrics.answer_acc` 是 **0–100 的有限值** | 继续 | 缺失 → judge 调用失败，查 `JUDGE_API_KEY` / `logs/LLM_scores.jsonl` |
| 3.3 | **首次运行记录 `task_family_metrics` 的键名**（写进下面的记录表） | 继续 | — |
| 3.4 | `A3_F` 的 `answer_acc` 与 `A0_hard` 有明显差异（>1 pt） | 进 B 批 | 差距 <1 pt → 先按 E1 复查通路，仍无差异则直接跳 Step 6 出"零效应"结论 |

### 记录表（跑完填，后面写论文直接用）

| arm | answer_acc | overall_score | avg_steps | 墙钟时间 |
|---|---|---|---|---|
| A0_hard | | | | |
| A1_prior | | | | |
| A3_F | | | | |

| 项 | 值 |
|---|---|
| RUN_TAG | `q1` |
| task_family 键名 | |
| 单样本平均墙钟（s/query） | |
| GPU / 卡数 | |

---

## Step 4 — 主实验 B 批：4 个 config 档（约 4 h）

```bash
cd ~/01_Code/03_OneIDAgent
export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI

RUN_TAG=q1 DATASET=test_task4.jsonl \
ARMS="A3_online A3_F_nobudget A3_F_nocap A3_F_nofeedback" CUDA_VISIBLE_DEVICES=3,4,5 \
bash scripts/exp/run_module_ablation.sh
```

**预期输出**：4 个新 arm 目录；`manifest.jsonl` 追加 4 行（`run_tag` 头不重复写）。

**判据**

| # | 判据 | 通过 | 不通过 → 动作 |
|---|---|---|---|
| 4.1 | 4 臂 ckpt 各 281 行 | 继续 | 同 3.1 |
| 4.2 | `A3_online` 与 `A3_F` 的 `answer_acc` **允许完全相同**（说明该数据上 contextual θ 未改变决策）——这是可接受结果，不算失败 | 继续 | 若差异 >5 pt，检查 `f_ctx.yaml` 的 `theta_path` 是否真的被读到（见附录 E4） |
| 4.3 | `A3_F_nofeedback` 存在且完成 | 继续 | 缺 → 该档 `feedback_damping: 0.0` 可能触发断言，看日志后单独重跑 |

---

## Step 5 — 质检：4 道 gate（15 min）

**任何一道不过，先修再判分**，否则后面所有表格都是废的。

```bash
cd ~/01_Code/03_OneIDAgent
OUT=outputs/module_ablation/q1
ARMS="A0_hard A1_prior A3_F A3_online A3_F_nobudget A3_F_nocap A3_F_nofeedback"

echo "== G1 完整性 =="
for A in $ARMS; do printf "%-18s %s\n" "$A" "$(wc -l < $OUT/$A/ckpt/*.jsonl 2>/dev/null || echo MISSING)"; done

echo "== G2 强制 mode 生效 =="
python - <<'PY'
import json,glob,collections
for f in sorted(glob.glob("outputs/module_ablation/q1/*/armlog.jsonl")):
    c=collections.Counter(); d=0; n=0
    for l in open(f):
        r=json.loads(l); n+=1; c[r["effective_mode"]]+=1; d+=bool(r["degraded"])
    print(f"{f.split('/')[-2]:18s} calls={n:4d} modes={dict(c)} degraded={d}")
PY

echo "== G3 退化率 =="
python - <<'PY'
import json,glob
for f in sorted(glob.glob("outputs/module_ablation/q1/*/armlog.jsonl")):
    rs=[json.loads(l) for l in open(f)]
    bad=[r for r in rs if r["degraded"]]
    print(f"{f.split('/')[-2]:18s} degrade_rate={len(bad)/max(1,len(rs)):.3%}",
          "reasons=", {r["degrade_reason"] for r in bad})
PY

echo "== G4 是否有样本仍在跑/被截断 =="
grep -l "max_rounds" $OUT/*/ckpt/*.jsonl 2>/dev/null | head
python - <<'PY'
import json,glob,collections
for f in sorted(glob.glob("outputs/module_ablation/q1/*/ckpt/*.jsonl")):
    c=collections.Counter(json.loads(l)["results"].get("finish_reason") for l in open(f))
    print(f"{f.split('/')[-2]:18s}", dict(c))
PY
```

**判据**

| Gate | 判据 | 通过 | 不通过 → 动作 |
|---|---|---|---|
| **G1** | 7 臂 ckpt 全部 == 281 | 继续 | 重跑该臂（自动续跑） |
| **G2** | 每臂 `effective_mode` **只有 1 个取值**，且等于该臂声明的 mode | 继续 | 混合 → 附录 E2 |
| **G3** | 每臂 `degrade_rate` < 2% | 继续 | ≥2% → 附录 E3，先把退化清掉再判分 |
| **G4** | `finish_reason` 中 `max_rounds` 占比 < 5%，`None` 为 0 | 继续 | 偏高 → 加大 `MAX_ROUNDS` 或查工具报错率 |

---

## Step 6 — 判分与统计（30 min）

```bash
cd ~/01_Code/03_OneIDAgent
: "${JUDGE_API_KEY:?please export first}"
export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI

python scripts/exp/analyze_module_ablation.py \
  --root outputs/module_ablation/q1 \
  --dataset $ONEID_DATA_ROOT/test_task4.jsonl \
  --reference A1_prior \
  --out outputs/module_ablation/q1/REPORT.md
```

**预期输出**

```
[judge] prompt=tool_server.tf_eval.tasks.rsagent.task calls=1967 cache_hits=0

## 1. Outcome per arm
| arm | n | answered | answer_acc | acc@0.50 | selective acc | empty |
...
## 2. Paired comparison vs `A1_prior`
| arm | d answer_acc | 95% CI | McNemar b/c | p |
...
## 3. Answer content vs commitment
...
## 4. Module exposure
| arm | 调用 CMI 的轨迹占比 | 平均调用数 | armlog 调用数 | degraded | 主要原因 |
...
[ok] .../module_ablation_report.json
```

**判据**

| # | 判据 | 通过 | 不通过 → 动作 |
|---|---|---|---|
| 6.1 | 每臂 `n = 281` | 继续 | n<281 → ckpt 有缺行，回 Step 3/4 |
| 6.2 | `prompt=` 显示的是 `tool_server...task`（不是 fallback） | 继续 | fallback → 说明 repo 不可导入，检查是否在仓库根目录执行 |
| 6.3 | 表 4 中 `A0_hard` 的"调用 CMI 的轨迹占比" **> 0.8** | 继续 | 低 → 模型大量跳过该工具，代表该臂几乎未被触发，结论必须限定（写进 limitation） |
| 6.4 | 表 4 中 `A3_*` 的占比同样 > 0.8 | 继续 | 低 → 与 6.3 同 |
| 6.5 | 表 1 的 `answered` 率各臂差 < 0.15 | 继续 | 差异大 → 说明存在"少答即变准"的混淆，必须同时引用表 3 的 `silent, answerable` 一列 |

---

## Step 7 — 判定分支与论文落点（30 min）

令
```
Δ  = answer_acc(A3_F) − answer_acc(A1_prior)
Δ′ = answer_acc(A3_F) − answer_acc(A0_hard)
```

| 分支 | 条件 | 结论怎么写 | 论文动作 |
|---|---|---|---|
| **S1 成立** | Δ ≥ +0.03 且 95% CI 不含 0 | "在同一 SFT checkpoint 上仅替换推断模块，答案正确率提升 Δ 点" | 表 α + 表 β 进 §5.2.2；Abstract 加一句；rebuttal 直接引 |
| **S2 上界** | −0.03 < Δ < +0.03，或 CI 含 0 | "模块贡献上界 < 3 pt；推断层面的收益（Truth Acc +1.69, Holm p=0.0048）仍是主要贡献" | headline 收到推断层面；§5.2.2 写"agent 层为可部署性验证"；limitation 写清 |
| **S3 反向** | Δ ≤ −0.03 | "在该 checkpoint 与数据上，共享身份推断未改善端到端答案" | 先做附录 E1/E3 排查；确实是负向则在 Discussion 正面讨论，并把主张缩到 Table 2 |
| **S4 无效** | 表 4 覆盖率 < 0.5 | 该对比不成立 | 不报 Δ，改报"模块在该数据上触发率不足，无法评估"，并把这条写进 limitation |

**无论哪个分支都要做的三处文字改动**

1. **Table 1 caption 加一句**：`The inference module is fixed to the full shared-identity configuration for every row; this table isolates supervised fine-tuning.`
2. **§5.1 开头加一句**：`All rows run the same tool harness and the same inference module, so the differences in Table 1 measure fine-tuning, not shared-identity inference.`
3. **新增 §5.2.2**：模块消融，表格编号按论文顺序排（表 α → 新 Table 2，原 Table 2 顺延）。

---

## Step 8 —（可选，建议做）全量复核 + SFT-off 列（约 4–6 h）

**8a. 全量复核 3 臂**：只为证明消融干净、并把 task4 的数字放进完整 1,414 的表里。

```bash
RUN_TAG=q1full DATASET=test.jsonl \
ARMS="A0_hard A1_prior A3_F" bash scripts/exp/run_module_ablation.sh
```

**no-op 对照（最关键的一条）**：非 task4 轨迹在两臂之间必须**逐字相同**。

```bash
python - <<'PY'
import json,glob
ds=[json.loads(l) for l in open('/home/ubuntu/00_Data/RSTI/test.jsonl')]
def fam(i):
    return 'task4' in ds[int(i.split('_')[1])]['images'][0]
def load(arm):
    f=glob.glob(f'outputs/module_ablation/q1full/{arm}/ckpt/*.jsonl')[0]
    return {json.loads(l)['results']['idx']: json.loads(l)['results'].get('final_answer') for l in open(f)}
a,b=load('A0_hard'),load('A3_F')
keys=[k for k in a if k in b]
noc=[k for k in keys if not fam(k)]
print('non-task4:',len(noc),'identical:',sum(a[k]==b[k] for k in noc))
t4=[k for k in keys if fam(k)]
print('task4    :',len(t4),'identical:',sum(a[k]==b[k] for k in t4))
PY
```

| # | 判据 | 通过 | 不通过 → 动作 |
|---|---|---|---|
| 8a.1 | `non-task4` 的 identical == 总数（100%） | 消融干净，可写"其余四组逐字不变" | 不一致 → 有别的随机源（seed / 采样），检查 `generation_config` 与 `batch_size` 是否跨臂不同 |
| 8a.2 | task4 上的 Δ 与 Step 6 的 Δ 相差 < 0.02 | 结论稳健 | 差异大 → 说明子集与全量不是同一批图像，回查映射 |

**8b. SFT-off 列**（回答"基线是否也训练过"的正面证据）

```bash
export MODEL_PATH=$PWD/models/0_basemodel/Qwen3-4B-Instruct-2507
RUN_TAG=base DATASET=test_task4.jsonl \
ARMS="A0_hard A3_F" bash scripts/exp/run_module_ablation.sh
python scripts/exp/analyze_module_ablation.py --root outputs/module_ablation/base \
  --dataset $ONEID_DATA_ROOT/test_task4.jsonl --reference A0_hard
```

> ⚠️ `SKIP_EXISTING=1` + 已存在的 ckpt 会串到 SFT 结果，所以**必须用新的 `RUN_TAG`**（`base`），不要覆盖 `q1`。

| # | 判据 | 通过 |
|---|---|---|
| 8b.1 | 基座两臂 ckpt 各 281 行 | 继续 |
| 8b.2 | 基座 `A0_hard` 的 answer_acc 显著低于 SFT `A0_hard` | 这直接量化"SFT 带来的增益"，与模块增益并列进表 α 脚注 |

---

## 附录 A — 异常处置表

| 编号 | 症状 | 最可能原因 | 处置 |
|---|---|---|---|
| **E1** | 两臂 `final_answer` 完全相同 | 模块输出没进 prompt（`output_mode` 把关键字段 strip 了）或模型没读 | ① 对比两臂同一条样本的 OBSERVATION 文本；② 临时把 `output_mode` 设 `full` 重跑 5 条；③ 仍无差异 → 模块不在通路上，**停止主实验** |
| **E2** | `armlog` 出现混合 `effective_mode` | worker 进程没继承 env（被别处 `unset`），或 `ONEID_FORCE_MODE` 拼写错 | `env \| grep ONEID` 核对；确认 `run_module_ablation.sh` 里 export 的变量名一致 |
| **E3** | `degraded` 比例高 | 适配器预算（`ONEID_DSVI_BUDGET_SECONDS`，默认 120 s）或异常降级 | 看 `degrade_reason`；`budget_exceeded` → 调大到 300；`adapter_failed:*` → 看栈 |
| **E4** | `A3_online` 与 `A3_F` 差异异常大 | `theta_path` 没被读到（相对路径解析失败） | 绝对化 `theta_path` 后用 5 条冒烟复跑对比 |
| **E5** | `answer_acc` 全 0 或全 1 | judge 返回被 `extract_score` 解析成极值 | 看 `judge_cache.jsonl` 的 `s` 分布与原始 `evaluation` 文本 |
| **E6** | 某臂 ckpt 停在非 281 行 | OOM / 工具超时 | 减 `NUM_PROCESSES`、降 `BATCH_SIZE` 到 1，重跑（自动续跑） |
| **E7** | `eval/*.json` 不存在 | `res_log` 只在主进程写 | 用 `NUM_PROCESSES=1` 重跑该臂的评测段 |

## 附录 B — 目录结构（跑完后应长这样）

```
outputs/module_ablation/q1/
├── manifest.jsonl                     # 每次调用一行：arm / mode / config
├── judge_cache.jsonl                  # 判分缓存（可复用，重跑免费）
├── module_ablation_report.json        # 机器可读结果
├── REPORT.md                          # 四张表
├── A0_hard/
│   ├── ckpt/q1_A0_hard_rsagent_e2e_ckpt.jsonl   # 281 行，断点续跑依据
│   ├── eval/q1_A0_hard_rsagent_e2e_ckpt.json    # headline_metrics / task_family_metrics
│   ├── tools_output/                            # 中间产物
│   └── armlog.jsonl                             # 每次工具调用一行
├── A1_prior/ ...
└── A3_F/ ...
```

## 附录 C — 代码改动与回滚

| 文件 | 改动 | 回滚 |
|---|---|---|
| `tool_server/tool_workers/online_workers/CrossModalInterpret_worker.py` | +`ONEID_FORCE_MODE` / `ONEID_ARM_LOG`，`generate` 拆成 wrapper + `_dispatch` | `cp *.bak_armablation` 回原文件 |
| `tool_server/tf_eval/utils/task_utils.py` | +`ONEID_DATASET_PATH` | 同上 |
| `scripts/exp/*` | 新增，无侵入 | 直接删目录 |

全部改动 env 门控：不设任何 `ONEID_*` 时行为与改动前逐字节一致（已验证 `tests/contract/test_cross_modal_api.py` 10/10 通过）。

## 附录 D — 一张表总结"这一套实验到底证明了什么"

| 已被证明 | 本清单补上 |
|---|---|
| SFT 让工具调用与答案大幅提升（Table 1） | SFT 的增益**与模块无关**（Table 1 定位声明） |
| 模块让 pair F1 0.4553→0.7320、Truth Acc +1.69（Table 2） | 模块是否**改善最终答案**（表 α） |
| 模块改变了身份与真值信念 | 模块输出是否**进了答案**（表 4 覆盖率 + 表 β） |
| 决策层 precision/coverage（Tables 12–13） | 承诺的答案**内容**对不对（表 β） |
