# exp_1_1 · SFT 记忆效应 vs OneID 增益

2x3 条件矩阵，用户论文表里的 2x2 是其中四格：

```
                    信息条件
              none(闭卷)   det(仅检测)   oneid(检测+OneID)
weights base   base_none    base_det      base_oneid      ← 右两列 = 2x2 上排
        sft    sft_none     sft_det       sft_oneid       ← 右两列 = 2x2 下排
```

- `none`：提示词去掉全部工具条目 + 硬性说明"本轮无工具"，Policy 全拦，量的是**记忆/先验**；
- `det`：21 个工具（无 `CrossModalInterpret`）；
- `oneid`：22 个工具，并追加一段**取证配方**（先两路检测、再 OneID；不要手写检测）。

## 文件

| 文件 | 作用 |
|---|---|
| `config.py` | 条件定义、模型解析、提示词派生与结构断言（0/21/22 个工具条目） |
| `run.py` | 主跑：cell × seed，落盘 `traces/<cell>_seed<k>.jsonl` |
| `probes.py` | 记忆探针 m1/m2/m3 + OneID 因果探针 splice |
| `analyze.py` | 2x2 主表、G_E/G_O/G_S/DiD、McNemar、按 pair 聚类的 bootstrap CI |
| `selftest.py` | CPU 自检（20 条断言），不需要 GPU/controller |
| `run_all.sh` | P0–P6 全流程 |

## 运行

```bash
cd ~/01_Code/03_OneIDAgent
export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI CHECKPOINT_ROOT=/home/ubuntu/00_CPK

# P0 体检 + 小样本 pilot（每任务 2 条，验证四臂输出与日志格式）
python -m scripts.exp_1_1.run --data $ONEID_DATA_ROOT --check
python -m scripts.exp_1_1.run --data $ONEID_DATA_ROOT --cells arms --per-family 2 \
       --seed 1 --out outputs/exp_1_1/pilot --no-resume

# P2 闭卷基线（分离记忆）
python -m scripts.exp_1_1.run --data $ONEID_DATA_ROOT --cells base_none,sft_none \
       --seed 1 --out outputs/exp_1_1/run_seed1

# P3 主跑：双卡并行，base 两臂 / sft 两臂
CUDA_VISIBLE_DEVICES=0 python -m scripts.exp_1_1.run --data $ONEID_DATA_ROOT \
    --cells base_det,base_oneid --seed 1 --out outputs/exp_1_1/run_seed1
CUDA_VISIBLE_DEVICES=1 python -m scripts.exp_1_1.run --data $ONEID_DATA_ROOT \
    --cells sft_det,sft_oneid --seed 1 --out outputs/exp_1_1/run_seed1

# P4/P5 探针
python -m scripts.exp_1_1.probes memory --data $ONEID_DATA_ROOT --pairs 100
python -m scripts.exp_1_1.probes splice --data $ONEID_DATA_ROOT --pairs 100

# P6 汇总
python -m scripts.exp_1_1.analyze --traces outputs/exp_1_1/run_seed1/traces \
       --probes outputs/exp_1_1/probes --out outputs/exp_1_1/report
```

## 日志格式（一行一样本）

```json
{"cell":"sft_oneid","weights":"sft","condition":"oneid","seed":1,
 "model":".../qwen3_4b_rs_text_sft_14336","prompt_sha":"...",
 "sample_id":"S00000_task4_000001_bridge","family":"task4","pair_id":"task4_000001",
 "target":"bridge","question":"...","label":"...","final_answer":"...",
 "tool_sequence":["ObjectDetection","DenoDetV2","CrossModalInterpret"],
 "oneid_calls":1,"oneid_blocked":[],
 "action_log":[{"round":1,"api_name":"ObjectDetection","error":null,"arg_keys":[...],
                "n_rgb":0,"n_sar":0,"prov":[]}],
 "tool_log":[{"api_name":"ObjectDetection","error_code":0,"n_detections":12,"text":"..."}],
 "trace":{"rounds":4,"raw_tool_calls":3,"tool_errors":0,"error_tags":[],
          "wall_time":41.2,"llm_time":33.1,"tool_time":7.6},
 "score":{"counts_ok":false,"answer_ok":false,"attr_hits":2,"attr_total":4,...},
 "error":null}
```

## 首跑（pilot，4 臂 × 每任务 2 条 = 16 条）

实测：`sft_oneid` 会按配方取证（3/4 调 OneID，2–4 轮完成），`base_oneid` 不会
（重复检测、`invalid_dependency`、打满 16 轮）。这正是 H3（SFT×OneID 交互）要看的信号，
不是装置问题——`oneid_call_rate` 必须与 `G_O` 一起报。

## 本轮修掉的四个装置问题

1. 一视图零目标时门禁判 `invalid_dependency`：新增 `register_empty_detection`（独立负例槽位，
   不改 `register_detections` 契约），并在 `prepare_interpret_args` 里注入空列表 + provenance；
2. 模型把 `rgb_detections` 写成占位符字符串：非"检测字典列表"一律丢弃、交由 registry 注入；
3. 提示词缺取证配方：ON 条件追加三步硬性说明（两臂等量受益）；
4. trace 追加 `action_log` / `tool_log`，用于定位"为什么没调 OneID"。

## 待办

- 主跑完成后补 `seeds 2/3`（`--seed` 已支持）；
- `analyze.py` 目前给的是 per-condition 的 G_E/G_O；接 `none` 条件后自动出记忆分解表；
- 建议补"仅在两臂都调用了 OneID 的样本上"的条件增益（complier average），避免把
  "会不会按配方"混进"OneID 好不好"。
