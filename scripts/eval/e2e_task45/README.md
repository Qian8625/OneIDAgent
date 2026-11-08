# task4/task5 上的 2×2 消融

```
                          相同工具、无 OneID          相同工具、启用 OneID
No SFT   base              Base Agent + Tools          Base Agent + Tools + OneID
SFT      sft               SFT Agent + Tools           SFT Agent + Tools + OneID
```

四个臂之间**只允许**两个维度不同：权重（基座 / SFT）与 OneID（关 / 开）。
工具集、提示词模板、样本顺序、评分、解码参数必须完全一致，否则增益不可归因。

---

## 1. 现有 e2e 代码缺什么

| # | 缺口 | 证据 | 影响 | 本目录对策 |
|---|---|---|---|---|
| 1 | 评测主入口跑不起来 | `tool_server/tf_eval/evaluator.py:16` `from .models import get_model`，而 `tool_server/tf_eval/models/` 不存在（`git log -- tool_server/tf_eval/models` 为空，从未入库）。实测：`python -c "import tool_server.tf_eval.models"` → `ModuleNotFoundError` | `scripts/eval/eval_e2e.sh` / `eval_e2e_api.sh` 调用的 `python -m tool_server.tf_eval` 直接 ImportError，现有"e2e 评测"不可用 | 改走 `agent_runtime.AgentRunner`（UI/CLI 同一条循环），`runner.py` |
| 2 | 没有"臂"的概念 | 无任何 arm/model/prompt 指纹落盘；`scripts/ci` 只有 static/worker/ui 三套 | 事后无法证明某行结果是哪个臂跑的 | `arms.py` + `runner.py` 每行落盘 arm/model/prompt_sha |
| 3 | 提示词不可按臂切换 | `agent_runner.py:777` `system_prompt = RS_AGENT_PROMPT` 硬编码；`RS_AGENT_PROMPT` 被 app/chat/agent_runner 三处共用 | "无 OneID" 臂无法只改提示词 | `arms.py::build_system_prompt` 从唯一提示词派生，并断言两臂工具条目只差 `CrossModalInterpret` |
| 4 | 没有工具白/黑名单 | `ToolManager.__init__` 里 `internal_available_tools` 与 `agent_available_tools` 赋值相同（base_manager.py:49/51），白名单钩子是空的；`call_tool` 对未知工具只回 `error_code=1` | 只能靠"不注册 worker"关 OneID，但提示词仍在推销该工具，模型会反复撞错 | `runner.ToolPolicy` 在派发前硬拦截，并记录 `oneid_blocked` |
| 5 | 数据没有按任务切分 | `test.jsonl` 是 task1–5 扁平一份（1565 行），`task_config` 只有 `num_sample`（前 N 条），无筛选 | 无法"只在 task4/task5 上做" | `data.py` 按图片目录切出 181 + 536 = 717 条 |
| 6 | 数据是围绕 OneID 构造的 | 717/717 条问法都写着 "Use workbook replay for pair ..." | 原文喂给"无 OneID"臂＝给它一条不可执行指令，基线被系统性压低 | `data.py::render_question` 渲染中性问法（不带任何工具名），两臂逐字相同；`--question-mode original` 可对照 |
| 7 | 评分依赖 LLM judge | `task.py::sim_score` 每条样本调一次 DeepSeek，无缓存、无一致性校验；`answer_acc` 是 judge 分数 | 四臂差值会被 judge 抖动污染，且要 API key | `score.py` 字段级确定性评分（计数 + 4 个属性），judge 只作可选复核 |
| 8 | 没有按 target/task 的分桶 | `classify_task_families` 只有 cross_modal/change_detection 等语义族，无 task4/task5 维度 | 报不出 task4 vs task5 的对比表 | `score.py::aggregate` 按 arm × family 出表 |
| 9 | 环境开关配错会单边变差 | workbook replay 需要 **worker 进程级** `ONEID_ENABLE_WORKBOOK_REPLAY=1` + `ONEID_DATA_ROOT`；而 `scripts/ci/e2e_common.py::base_env` 把 `ONEID_DATA_ROOT` 固定成 `<repo>/data` | OneID 臂可能全部 replay 失败而不自知 | `stack.py::preflight` 显式体检 |
| 10 | 起不起得来一套最小栈没人管 | `start_server_local.py` 只能整份 yaml 全起 20 个 worker（含 3 个 GPU 重模型），无法只起 task4/5 需要的 6 个 | 四臂反复起停会互相抢显存 | `stack.py::render_stack_yaml` 裁最小栈 |
| 11 | 没有 agent 循环的回归测试 | `scripts/ci` 三套只测 static/worker/ui，不跑 AgentRunner | 改 `agent_runner` 无测试兜底 | `selftest.py`（CPU/mock，23 条断言） |
| 12 | 权重缺失（阻断性） | `models/0_sft_train/` 不存在；`.env` 指向 `qwen3_sft_text_skill_0611`、`eval_e2e.sh` 默认 `qwen3_sft_text_0614`，两者都不在盘上；`00_CPK` 只有 `OpenEarthAgent`(Qwen3ForCausalLM) 与 `Qwen3-VL-*`/`Qwen2.5-VL-*`/`ZoomEarth-3B`，没有 SFT 的文本基座 | 2×2 的横轴（No SFT）没有可比对权重，实验无法开跑 | `--check` 明确列出哪个臂缺哪个 env；不猜默认值 |

---

## 2. 数据视图（实测）

| 项 | 值 |
|---|---|
| 样本总数 | 717（task4 181 + task5 536） |
| 唯一 pair | task4 96 / task5 362 |
| target 分布 | ship 288, aircraft 163, harbor 118, tank 77, vehicle 41, bridge 30 |
| 带属性的样本 | 471（其余 246 条 GT 是 "no recorded ..."，此时预测也必须留空） |
| 原始问法含 "workbook replay" | 717 / 717 |

一份 pair 下有多个 target 变体（如 `task4_000001` 有 bridge/harbor/ship/vehicle 四问），
所以 `sample_id` 形如 `S00012_task4_000001_bridge`。

---

## 3. 评分口径

* **计数**：GT 给出几个就比几个，全部相等记 `counts_ok`；
* **属性**：`subtype/color/size_bin/orientation` 逐字段比对；GT 为 `None`（"no recorded"）时
  **预测也必须没有值**，编造即扣分——这条对"无 OneID"臂尤其关键，否则"张口就来"会得分；
* **answer_ok** = 所有可比较字段都命中（计数 ∧ 属性），不是"计数对了就算对"；
* 表里同时给 `counts_acc` / `attrs_acc` / `attr_field_acc` / `empty_rate` / `oneid_call_rate`，
  便于区分"没调 OneID"和"调了但没答对"。

---

## 4. 怎么跑

```bash
cd ~/01_Code/03_OneIDAgent
PY=/home/ubuntu/miniconda3/envs/OEA/bin/python

# 0) 体检：数据 / 权重 / 环境开关
$PY scripts/eval/e2e_task45/run_matrix.py --data ~/00_Data/RSTI --check

# 1) 不跑模型，看四臂提示词指纹与样本量
$PY scripts/eval/e2e_task45/run_matrix.py --data ~/00_Data/RSTI --dry-run

# 2) 装机自检（CPU，无 GPU / 无网络 / 无 API key）
$PY scripts/eval/e2e_task45/selftest.py

# 3) 全量 mock 也跑得通（用 GT 自身当答案，应 100%）
$PY scripts/eval/e2e_task45/run_matrix.py --data ~/00_Data/RSTI --mock --arms base,base_oneid

# 4) 真跑
export ONEID_ENABLE_WORKBOOK_REPLAY=1
export ONEID_DATA_ROOT=$HOME/00_Data/RSTI
export CHECKPOINT_ROOT=$HOME/00_CPK
export ONEID_ABLATION_BASE_MODEL=/path/to/base     # 待定：SFT 的基座
export ONEID_ABLATION_SFT_MODEL=/path/to/sft       # 待定：SFT 权重

#   4a. 起最小栈（先落 yaml，再按需前台起）
$PY -c "from pathlib import Path; from scripts.eval.e2e_task45.stack import render_stack_yaml as r; \
print(r(Path('tool_server/tool_workers/scripts/launch_scripts/config/all_service_example_local.yaml'), \
Path('outputs/e2e_task45/stack_task45.yaml'), ['task4','task5']))"
python tool_server/tool_workers/scripts/launch_scripts/start_server_local.py \
       --config outputs/e2e_task45/stack_task45.yaml

#   4b. 四臂（可 --limit 先小跑，断点续跑默认开启）
$PY scripts/eval/e2e_task45/run_matrix.py --data ~/00_Data/RSTI --tasks task4,task5
```

输出：

```
outputs/e2e_task45/
  traces/{base,base_oneid,sft,sft_oneid}.jsonl   # 逐样本，可续跑
  summary.csv / summary.json / REPORT.md          # 2×2 表
```

`--limit N` 是"每个臂前 N 条"，`--no-resume` 关闭续跑，`--arms` 可只跑部分臂。

---

## 5. 已验证 / 未验证

已验证（本机实测）：

* `selftest.py`：23/23。含"有证据时臂策略拦截 / 无证据时依赖门禁先触发"两种顺序；
* 真实数据 717 条 × 2 臂 mock 全跑通，`answer_acc` 两侧均 100%，说明 GT 解析与评分在所有样本上自洽；
* `--dry-run`：两臂提示词工具条目 21 vs 22，指纹 `dcd024afc6b3` / `701dca7127cd`。

未验证（需要你补）：

1. **两份权重**。SFT 的路径与它的文本基座都缺；基座必须与 SFT 同族，否则横轴无意义；
2. **最小栈真起**（`ObjectDetection`/`DenoDetV2`/`SmallObjectDetection` 要 GPU 与检测权重）；
3. **OneID 臂的 workbook replay 是否真的生效**——起栈前导出的两个 env 缺一不可，
   `--check` 只做提示，不代你启动；
4. 真模型的解码参数（temperature/seed/max_new_tokens）目前沿用 `LocalLLM` 默认值
   （贪心 + 256），四臂一致；若你要采样，必须四臂同参。

---

## 6. 两个需要你确认的口径

1. **中性问法**：默认把 "Use workbook replay for pair ..." 改写成不带工具名的问法，
   保证两臂收到同样输入。若你希望保留原文（更接近论文里的原始设定），
   用 `--question-mode original`，但"无 OneID"臂会收到不可执行指令，需在论文里说明。
2. **属性 GT 为 None 的 246 条**：当前口径是"预测留空才得分、编造即扣分"。
   若更希望"这 246 条只比计数"，改 `score.py::score_answer` 里 `attr.*` 的比较策略即可，
   但四臂必须同时改。


---

## 7. 权重与四臂落点（2026-11-08 解析）

`cpk/` 下三份 Qwen3-4B 都是**同构全量权重**（4.022B / BF16 / 398 tensors / tied embedding /
262144 位置），差别只在权重值和模板存放位置：

| 权重 | 用途 | 关键事实 |
|---|---|---|
| `cpk/base/Qwen3-4B-Instruct-2507` | No SFT 臂 | 模板内嵌在 tokenizer_config（sha `64f85b1980`） |
| `cpk/sft/qwen3_4b_rs_text_sft_14336` | SFT 臂 | 模板在 `chat_template.jinja`；渲染结果与 base **逐字符相同** |
| `cpk/OpenEarthAgent` | 参考（非本实验） | 同为 Qwen3-4B，但连 embedding/norm 都变了 |

`weights.py delta` 的实证：base→sft **36/36 层的 q_proj 全变**，而 `embed_tokens` 与
`model.norm` **逐位相同**（relL2 0），层内 relL2 落在 `0.0128–0.0252`。
这正是 ms-swift `lora_rank=16` 合并进全量权重后的指纹：动的是每层线性层，没动嵌入与 norm。

```bash
python scripts/eval/e2e_task45/weights.py report --ckpt-root cpk --json outputs/e2e_task45/weights_report.json
python scripts/eval/e2e_task45/weights.py delta  --a cpk/base/Qwen3-4B-Instruct-2507 \
                                                 --b cpk/sft/qwen3_4b_rs_text_sft_14336
```

四臂的模型来源：

```bash
export ONEID_ABLATION_BASE_MODEL=$PWD/cpk/base/Qwen3-4B-Instruct-2507
export ONEID_ABLATION_SFT_MODEL=$PWD/cpk/sft/qwen3_4b_rs_text_sft_14336
```

## 8. 打通四臂时修掉的三处阻断

1. **检测结果进不了证据层**（`cross_modal_evidence.py` + `agent_runner.py`）
   线上 worker 只回 `"(x1, y1, x2, y2), label, score s"` 文本，没有 `structured`；
   `register_detections` 因此永远拿不到检测，`CrossModalInterpret` 一律被判
   `invalid_dependency`。新增 `parse_text_detections()` / `enrich_detection_response()`，
   并在**按类过滤之前**补齐（带原始 `index`），否则过滤会重排下标。
2. **位置绑定被分数抖动打断**（`workbook_replay_adapter.py`）
   重跑检测时分数有 ~1e-3 抖动，近似并列的框互换位置，`(view, index)` 绑不上，
   报 `Attribute binding failed`。新增**可选** `ONEID_REPLAY_BINDING=iou_assisted`：
   位置绑定失配时改用同视图、未占用、IoU≥0.5 的行，并在 `replay_diagnostics`
   里记 `binding_mode` 与 `iou_assisted_bindings`。**默认仍是纯位置绑定**。
3. **256 token 截断导致 no_action 死循环**（`runner.py`）
   4B 模型的 `thought` 一旦稍长就被截断，JSON 不完整 → 解析失败 → 反复重问。
   矩阵统一设 `--max-new-tokens 512`（四臂同参）。

由此需要三个环境变量（起栈**之前**导出）：

```bash
export ONEID_ENABLE_WORKBOOK_REPLAY=1
export ONEID_REPLAY_BINDING=iou_assisted
export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI
```

起最小栈（本机实测 5 个 worker 约 40s 全部注册）：

```bash
export PATH="$PWD/outputs/e2e_task45/bin:/home/ubuntu/miniconda3/bin:$PATH"   # 需要 python/conda 在 PATH
python tool_server/tool_workers/scripts/launch_scripts/start_server_local.py \
       --config outputs/e2e_task45/stack_task45.yaml
```

## 9. 首跑记录（pilot，4 臂 × 每任务 2 条）

```bash
python scripts/eval/e2e_task45/run_matrix.py --data ~/00_Data/RSTI --arms all \
       --per-family 2 --max-rounds 16 --max-new-tokens 512 --out outputs/e2e_task45/pilot
```

结果见 `outputs/e2e_task45/pilot/REPORT.md`。要点：

* 四臂全部跑通（16 条 trace，无崩溃），说明装置可用；
* 计数全对率全为 0：4B 模型在 16 轮内很难独立完成"两路检测 → 去重计数"；
* OneID 调用率极低（仅 1/4）：ON 臂多数样本**先**调 `CrossModalInterpret` 或检测为空，
  被依赖门禁判 `invalid_dependency`——这不是 OneID 无效，而是模型没按配方先取证；
* 因此正式跑之前要解决"取证配方"：或在 ON 提示词里把
  `ObjectDetection(无 text) → DenoDetV2(无 classes) → CrossModalInterpret(workbook_replay)`
  写成硬性三步，或给 1 个 few-shot；OFF 臂若要能答属性，还需起 `BaseModel` +
  `RegionAttributeDescription`（当前最小栈未含）。


---

## 10. 复现性实测与回归结论（2026-11-08 复查）

回归矩阵（全部通过）：

| 检查 | 结果 |
|---|---|
| `python3 scripts/ci/run_contract_tests.py` | 34/34 |
| `run_e2e.py --only static,worker` | static 8/8、worker 10/10 |
| `e2e_ui.py`（`ONEID_E2E_UI_PYTHON=OEA`） | 5/5 |
| `selftest.py` | 23/23 |
| 线上在线模式冒烟（OD+DenoDet→CrossModalInterpret，无 workbook） | `code=0, mode=shared_identity, degraded=False` |
| 线上 replay 冒烟（`ONEID_REPLAY_BINDING=iou_assisted`） | `code=0, bound=1, min_iou=0.929` |

### 可复现性：当前四臂结果**不可逐位复现**，原因不在本目录的代码

把 pilot 的 `base` 臂原样重跑（同 4 条样本、新输出目录、`--greedy`）：

```
identical final_answer: 0 / 4
pilot : rounds 4/3/7/6 ，多数正常 Terminate
rerun : rounds 16/16/5/16，3 条 max_rounds + no_action，最终答案为空
```

两个独立来源：

1. **检测器跨运行抖动**：同一模型、同一图，分数有 ~1e-3 差异（实测近似并列的框会
   互换位置，见 `weights.py`/`workbook_replay_adapter` 一节的记录）。观察文本因此不同，
   智能体后续决策随之分叉。这属于工具侧行为，与消融脚本无关。
2. **LLM 解码**：checkpoint 的 `generation_config` 是 `do_sample=True, temperature=0.7`。
   `--greedy` 确实把解码变成确定性，但实测**质量明显变差**（thought 变长 →
   到 `max_new_tokens` 被截断 → `no_action` → 打满轮次）。

因此：

* 现有 16 条 pilot 只能当"装置可用"的冒烟，不能当方法结论；
* 正式跑要么固定 seed + 统一温度并报告多次运行的均值±方差，要么缓存工具输出
  （`--cache_dir` 思路）后只让 LLM 侧随机；
* `--greedy` 默认关闭，保留为可复现性诊断开关，不建议用于正式结果。

### 与其它评测流程的耦合面

* `scripts/ci` 三套 e2e 不经过 `agent_runner`，不受影响（实测绿）；
* `tests/contract` 全部走进程内引擎，不受影响（实测绿）；
* `scripts/eval/eval_e2e*.sh` 走 `tf_eval`（`from .models import get_model` 仍缺包，且
  `tools_output` 路径由 `save_dir` 决定），本次改动只在
  `AgentRunner` 里加了一行 `enrich_detection_response`，对它既无帮助也无损害；
* UI 路径（`app/app.py`）直接调 `filter_detection_response`，因此把补齐动作放进
  `filter_detection_response` 内部，两条路径共享同一份逻辑，不需要改 app.py。
