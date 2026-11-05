# DS_VI：共享身份证据推理与跨模态装配

本包实现论文 **OneID-Agent** 的推理侧：

1. `models/ds_fusion/` —— D-S 证据融合与检测配对（物理先验、RDC/PCR6、BetP 判决）；
2. `models/shared_identity/` —— 论文 §3 的联合身份关联与真值发现（观测/claim/fact 表示、
   物理候选图、A/Q 交替变分推断、上下文可靠性、provenance 预算、证据状态）；
3. `models/cascade/` —— 把 D-S 配对的观测交给 VI，并按输出策略装配对象；
4. `models/cross_modal/` —— 任务化输出（选最大/最高分/距离/联合场景）。

依赖方向：本包不读真值或标注工作簿，推理入口只读工具检测结果。

## 最小入口

```bash
python -B -m DS_VI.experiments.run_detection --family match --pair-limit 2 --output-dir DS_VI/results/match_smoke
python -B -m DS_VI.experiments.run_detection --family time  --pair-limit 2 --output-dir DS_VI/results/time_smoke
```

`--pair-id task4_000001` 精确选样本，可重复指定。默认读取唯一出厂配置
`configs/shared_identity/runtime_best.yaml`；可用 `--vi-config` 覆盖。配置内的相对路径按项目根解析。
输出目录已存在时拒绝覆盖。

## 核心量

| 符号 | 含义 | 实现 |
|---|---|---|
| `A[m, k]` | 观测 `m` 与候选对象 `k` 的关联信念（同一观测的所有 claim 共用一行，含 NULL） | `inference.update_association` |
| `Q[j](t)` | 事实 `j = (对象, 属性, 时刻)` 的真值分布 | `inference.update_truth` |
| `r_i` | 上下文条件化的 claim 正确率 `σ(b_{s,a} + β^T x_i)` | `reliability.py`、`contextual_reliability.py` |
| `ϖ_i` | provenance 组内预算，`1/|g(i)|`，每组至多 1 | `provenance.compute_group_weights` |
| `p_mk` | 物理先验（IoU/中心/面积/置信度 + 门控） | `physical_prior.py`、`candidate_graph.py` |

物理先验取论文 Eq.(2)：`p[m,k] ∝ exp(dᵀf[m,k])`，NULL 以 `exp(d_null)` 参与同一归一化，`d = prior_scale × weights`、`d_null = null_logit`（两常数冻结；论文在独立标注上拟合，本仓库不含拟合脚本）。

更新式：`A ∝ p·exp(S/η)` 经 KL/Bregman 投影回可行集（行归一、候选图、容量约束），
`Q ∝ π·exp(Σ ϖ A log L)`；两者交替直到收敛（`inference.run_joint_inference`）。

## 出厂配置要点（`configs/shared_identity/runtime_best.yaml`）

| 项 | 值 | 说明 |
|---|---|---|
| `physical.weights` | 0.45 / 0.30 / 0.15 / 0.10 | IoU / 中心 / 面积 / 置信度 |
| `physical.center_scale_factor` | 0.75 | 中心得分的对角线尺度 |
| `physical.null_prior` | 0.15 | 完美单候选留在地 NULL 的质量（同时是 `null_logit` 的推导锚点） |
| `physical.prior_scale` | 6.0 | 论文 Eq.(2) 中 `d` 的幅度：`d = prior_scale × weights` |
| `physical.null_logit` | `null`（推导） | 论文 Eq.(2) 的 `d_null`；留空则由 `null_prior` 与 `prior_scale` 推出 |
| `inference.eta` | **1.5** | 物理先验锚定强度（证据项温度） |
| `inference.capacity_projection` | `true` | 启用容量投影 |
| `inference.capacity_mode` | `block_object_column` | 论文 Eq.(29) 的按块容量语义，显式声明 |
| `inference.default_capacity` | `null` | 论文无列容量；NULL = 不设每对象上限 |
| `inference.block_capacity` | 1.0 | 每个（采集/通道）块内，一个对象至多吸附 1 个观测 |
| `inference.time_invariant_attributes` | `[]` | 论文 C.5：不同时刻是不同事实，不做跨时相合并 |
| `inference.truth_prior_smoothing` | 1.0 | π 均匀先验的平滑 |
| `inference.max_iter` / `tol` | 50 / 1e-5 | 交替迭代上限与收敛阈 |
| `reliability.contextual.theta_path` | `configs/shared_identity/artifacts/contextual_theta_v1.json` | 校准后的 `θ0`（12,028 条独立标注 claim、最小支撑 30、7 个 modality–attribute 通道） |
| `decision.truth_threshold` / `qualified_truth_threshold` | 0.6 / 0.4 | Confirmed / Weak Support 的分界 |
| `cascade.policy` / `blend_weight` | `blend` / 0.5 | VI 输出策略与混合权重 |

容量约束只有一条，与论文 Eq.(29) 一致：同一（采集/通道）块内，一个对象至多吸附 1 个观测；NULL 列不受限。配置里不再有每对象列容量，`capacity_mode` 也显式写成 `block_object_column`，不再依赖运行时的自动切换。

## 证据状态（论文 §3.4 / 附录 C.4）

`decision.py` 只产出四个状态：

| 状态 | 触发条件 |
|---|---|
| `Confirmed` | 真值强度 ≥ `truth_threshold`，且身份/覆盖/矛盾检查通过 |
| `Weak Support` | 真值强度落在 `[qualified_truth_threshold, truth_threshold)`，或无候选可判 |
| `Conflict` | 出现被验证的活跃矛盾（`contradiction_margin` 内存在冲突 claim） |
| `Uncertain` | 证据不足、身份歧义、或对象不在证据状态中 |

`acquisition.py` 依据关联熵/真值熵给出下一步采集建议；`capability` 记录可用模态，
缺失模态直接导致 `Uncertain`。

## 主要开关（`run_detection`）

| 开关 | 作用 | 默认 |
|---|---|---|
| `--policy` | VI 输出策略：`status` / `posterior` / `rank` / `blend` / `all` | `cascade.policy` = `blend` |
| `--policy-threshold` | `posterior`/`rank` 的阈值、`blend` 的混合权重 | `cascade.blend_weight` = `0.5` |
| `--extra-links N` | 独立数据通道：VI 也考虑 D-S 未使用的检测（0 关闭） | `cascade.extra_links` = `2` |
| `--extra-mode` | `mutual`（互为最近同类）/ `pruned`（每观测最近 k 个） | `cascade.extra_mode` = `pruned` |
| `--extra-policy` | 独立候选的报告方式：`confirmed` / `all` / `posterior` | `cascade.extra_policy` = `confirmed` |
| `--associator` | `hungarian`（默认）/ `ot`（需要 torch） | `hungarian` |
| `--vi-config` | 覆盖共享身份配置路径 | `runtime_best.yaml` |

## 确定性

评测臂固定 CPU `float64`（同输入逐位相同）；GPU `float32` 只用于扫描，
`tests/opt/test_ot_equivalence.py` 给出 tolerance 断言。torch 缺失时 OT 路径抛
`TorchUnavailableError`（`backend="auto"` 才退回 numpy，并把实际 backend 写进 `stats.json`）。

Windows 上 torch 会再加载一套 OpenMP/MKL，测试进程需要 `KMP_DUPLICATE_LIB_OK=TRUE`
（已放在 `tests/conftest.py`，只影响测试进程）。

## 测试

```bash
pytest -q DS_VI/tests -k "not opt"     # opt/ 需要 torch
```
