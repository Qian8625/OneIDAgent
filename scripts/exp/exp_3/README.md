# exp_3 — 五折交叉拟合（contextual reliability θ 的样本外校准）

回应审稿意见：*"94 样本 + 单一工作点"*。做法是**让每个样本都当一次测试集**：
5 折、按 pair 分组，每折只用训练折重拟合 θ，再在留出折上跑完整 DS→VI→cascade，
最后用仓库自带的 `evaluate_family` 打论文原版的 G4/G5/G6 指标。

> 只输出折间均值与置信区间，**不重算 468 的构成**（按用户说明，剔除样本另在回复中解释）。

---

## 1. 目录

| 文件 | 作用 |
|---|---|
| `data.py` | 读工作簿 → claim 级训练表；定义数据根、family/view/modality 映射 |
| `theta.py` | θ 的**拟合端**（仓库只自带推理端），复刻 artifact 契约 + 样本外 log-loss/AUC |
| `folds.py` | pair 分组 k 折；预留 `group_key` 接口供将来按 site 分组 |
| `run_cv.py` | 主驱动：每折拟合 → 写 fold YAML → 跑引擎 → 物化预测工作簿 → 评估 |
| `aggregate.py` | 折间均值 / SD / t-CI / bootstrap CI / pooled 准确率 → `summary.md` |
| `engine.py` | 带属性 claim 注入的 D-S→VI→cascade 入口（`inject_attributes=off` 与 `run_family` 逐字节一致） |
| `compare_arms.py` | 三臂合并 → `comparison.md` |

## 2. 运行

```bash
export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI
cd /home/ubuntu/01_Code/03_OneIDAgent

# 冒烟（每折只留 2 个 pair）
python -m scripts.exp.exp_3.run_cv --max-pairs-per-fold 2 --out /tmp/exp3_smoke

# 正式：三臂（属性 claim 注入 + 以 VI 判定值评分，缺一不可，见第 3 节）
for arm in refit shipped noctx; do
  python -m scripts.exp.exp_3.run_cv --arm $arm --inject-attributes on --label-source vi \
      --out outputs/exp3_crossfit/$arm
  python -m scripts.exp.exp_3.aggregate --run-dir outputs/exp3_crossfit/$arm
done
python -m scripts.exp.exp_3.compare_arms --root outputs/exp3_crossfit
```

已跑结果见 [`RESULTS.md`](RESULTS.md)；七项指标表见 [`RESULTS_METRICS.md`](RESULTS_METRICS.md)。

三臂含义：

| arm | θ 来源 | 回答什么 |
|---|---|---|
| `refit` | 每折在训练折上重拟合 | **样本外**校准收益（主结果） |
| `shipped` | 固定用 `contextual_theta_v1.json` | 原论文口径；在测试折上属样本内，CI 会偏乐观 |
| `noctx` | 关闭 contextual（`f_online`） | 表格可靠率基线，即"θ 到底带来多少" |

## 3. 每折算什么

```
train pairs ──► claim 表 ──► fit_theta ──► fold_k/theta.json
                                              │
test pairs  ──► run_family(vi_config=fold_k/vi.yaml)  ──► predictions.jsonl
                                                          │
                       copy tool workbook + replace identities sheet
                                                          │
                     evaluate_family(predicted.xlsx, truth.xlsx, family)
```

- **分组单位是 pair**：同一场景的两个视角与其标注永远在同一侧，否则留出估计会被场景泄漏污染。
- **折 YAML** 只改 `reliability.contextual.theta_path`，其余与 `scripts/exp/arms/f_ctx.yaml` 逐字一致。
- **预测工作簿** 的 `instances` 表原样复制（`identity_metrics` 用它当 bridge），仅替换 `identities` 表为 VI 输出。

### 两个关键开关（默认已打开）

1. `--inject-attributes on`：`run_detection.run_family` 的 `extra_claims` 恒为空，
   属性只进 consistency gate。不注入属性 claim，θ 只能影响 category，
   而 category 检测准确率 ~99%，三臂会几乎恒等。
2. `--label-source vi`：identities 的 `category` 若照抄 D-S 标签，
   输出与 θ 无关；只有写 **VI 判定值**（`attribute_beliefs`）才能测到校准效果。
   设 `detector` 可复现 trunk 行为作对照。

## 4. 指标

`METRIC_PATHS` 里主要是：

- `G4_end_to_end.<attr>.accuracy` —— **Truth Acc**（缺预测按错算），论文 Table 2 的口径；
- `.fill_rate` / `.accuracy_on_filled` —— 覆盖率与"填了就对"；
- `G5_identity.composite.{detection_accuracy, association_accuracy, confirmed, gt_double_view}`。

同时输出 claim 级样本外质量（`auc` / `log_loss` / `brier`），用来证明折内 θ 确实泛化。

## 5. 必须写进论文限制的两点

1. **θ 的拟合数据是重建的，不是原始 artifact 的那份。**
   原文件 `data/shared_identity_real_eval_ctx_clean.json` 未随仓库发布（仅有 sha256）。
   本实验从 `tool_/truth_*_property.xlsx` 重建 claim 表：tool 行与 truth 行按 IoU≥0.5 配对，
   两侧都有值的属性才算一条 claim。因此行列数与 12,028 不完全一致（当前重建：match 5,122 行 / time 25,233 行）。
   **结论应表述为"交叉拟合后的相对变化"，而不是对论文绝对数字的复现。**
2. **"holdout-sites" 目前退化为 pair 分组。**
   工作簿 18 列里没有 site/region 字段，`test.jsonl` 也没有。`folds.make_folds(group_key=...)`
   已预留接口：拿到 RSTIBench 场景元数据后传 `{pair_id: site}` 即可整组迁移，无需改其他代码。

## 6. 已知实现差异

- 仓库只自带 θ 的推理端；`theta.py` 用「按属性 × 模态截距 + 共享系数 + 特征 L2=1.0」复刻，
  与原 `oneid_agent.tf_eval` 的拟合实现不保证逐位一致。
- 重建的 claim 表含 `color` / `orientation`（2026-11 标注版），而 `contextual_theta_v1.json`
  拟合的是 `state` / `subtype` / `size_bin` / `category`。所以 `shipped` 臂对 color 会退回表格可靠率，
  这是 artifact 版本错配，不是实验错误 —— 写论文时需说明。
