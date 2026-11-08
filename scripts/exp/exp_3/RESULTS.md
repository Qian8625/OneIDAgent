# exp_3 结果 — 五折交叉拟合

生成时间 2026-11-07，命令：

```bash
export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI
for arm in refit shipped noctx; do
  python -m scripts.exp.exp_3.run_cv --arm $arm --inject-attributes on --label-source vi \
      --out outputs/exp3_crossfit/$arm
  python -m scripts.exp.exp_3.aggregate --run-dir outputs/exp3_crossfit/$arm
done
python -m scripts.exp.exp_3.compare_arms --root outputs/exp3_crossfit
```

## 设置

| 项 | 值 |
|---|---|
| 折数 / 随机种子 | 5 / `20261107` |
| 分组单位 | pair（两个视角与其标注同折） |
| 评估规模 | 455 个 pair（match 95 + time 360），每折 ~91 |
| 每折 θ | 仅在训练折上重拟合（`reliability.contextual.theta_path` 指向 `fold_k/theta.json`） |
| 属性证据 | `subtype` / `color` / `size_bin` / `orientation` 作为 claim 注入 VI |
| 评分对象 | VI **判定**的属性（`attribute_beliefs`），缺判定时回退检测器值 |
| 指标代码 | `property_metrics.evaluate_family`（论文原版 G1/G2/G4/G5/G6） |

## 主结果（折间均值，[95% t-CI]）

### match（光学/ SAR 配对）

| 指标 | `refit`（样本外 θ） | `shipped`（固定 θ） | `noctx`（表格可靠率） |
|---|---|---|---|
| category acc (Truth Acc) | 0.5739 [0.5321, 0.6157] | 0.5739 [0.5321, 0.6157] | 0.5739 [0.5321, 0.6157] |
| category fill rate | 0.5809 [0.5421, 0.6197] | 0.5809 [0.5421, 0.6197] | 0.5809 [0.5421, 0.6197] |
| subtype acc | **0.3881** [0.3302, 0.4461] | 0.3781 [0.3215, 0.4348] | 0.3781 [0.3215, 0.4348] |
| color acc | 0.5110 [0.4451, 0.5769] | 0.5110 [0.4451, 0.5769] | 0.5110 [0.4451, 0.5769] |
| size_bin acc | 0.5674 [0.5287, 0.6061] | 0.5681 [0.5293, 0.6069] | 0.5681 [0.5293, 0.6069] |
| association acc (G5) | 0.3217 [0.1932, 0.4502] | 0.3217 [0.1932, 0.4502] | 0.3217 [0.1932, 0.4502] |

Pooled（sum correct / sum support）：category 751/1308；subtype **509/1308 vs 496/1308**；
color 470/919；size_bin 742/1308 vs 743/1308。

### time（时相对）

| 指标 | `refit` | `shipped` | `noctx` |
|---|---|---|---|
| category acc (Truth Acc) | 0.7603 [0.7242, 0.7964] | 0.7603 [0.7242, 0.7964] | 0.7607 [0.7245, 0.7969] |
| category fill rate | 0.8175 [0.7735, 0.8616] | 0.8175 [0.7735, 0.8616] | 0.8179 [0.7741, 0.8617] |
| subtype acc | 0.6149 [0.5531, 0.6768] | 0.6157 [0.5542, 0.6771] | 0.6137 [0.5502, 0.6772] |
| color acc | 0.5835 [0.5341, 0.6328] | 0.5838 [0.5347, 0.6329] | 0.5906 [0.5401, 0.6410] |
| size_bin acc | 0.7881 [0.7439, 0.8323] | 0.7889 [0.7440, 0.8339] | 0.7940 [0.7485, 0.8395] |
| association acc (G5) | 0.6969 [0.5829, 0.8109] | 0.6969 [0.5829, 0.8109] | 0.6980 [0.5848, 0.8111] |

Pooled：category 3479/4572；subtype **2827 vs 2831 vs 2822**；
color **2656 vs 2658 vs 2690**；size_bin **3598 vs 3603 vs 3626**。

## claim 级样本外质量（θ 本身是否可信）

| arm | AUC | log_loss | brier |
|---|---|---|---|
| `refit` | **0.8811** | 0.3131 | 0.1030 |
| `shipped` | 0.5658 | 0.3802 | 0.0715 |
| `noctx` | — | — | — |

## 怎么读

1. **θ 本身确实可学且可泛化**：逐折重拟合后，留出 claim 上 AUC 0.88、
   log-loss 0.31，说明上下文（visibility）对"这条 claim 对不对"有真实预测力，
   不是过拟合产物。
2. **但端到端收益远小于 claim 级收益**：最强的单点提升是 match 的 subtype
   （+1.0 pt，509 vs 496），其余指标在 ±0.7 pt 内波动且方向不一致
   （time 的 color / size_bin 反而略降）。
3. **折间 CI 很宽**（±3～5 pt），说明 95 个 match pair / 360 个 time pair 的规模
   不足以在单折上分辨 1 pt 级差异；要下"模块有效"的结论，应报 pooled 差值与
   配对检验，而不是单看均值。
4. `shipped` 的 AUC 只有 0.57：`contextual_theta_v1.json` 拟合的是
   `state` 通道且基于另一份数据构造，与本仓库 2026-11 标注版（`color`）不匹配。
   **论文若继续引用该 artifact，需要重新标注/重拟合，否则 `F` 臂等于用了错位的先验。**

## 结论表述建议

> 在按 pair 分组的五折交叉拟合下，上下文可靠率 θ 在 claim 级可泛化
> （AUC 0.88），但对最终对象属性判定的影响有限：match 族 subtype 提升
> +1.0 pt（pooled 509/1308 vs 496/1308），其余轴变化在 ±0.7 pt 内且不显著。
> 因此该模块的贡献应定位为"证据加权更校准"，而非"端到端显著提升"。

## 限制（必须与结果同页出现）

1. θ 的拟合数据是**从工作簿重建**的（原 `shared_identity_real_eval_ctx_clean.json`
   未随仓库发布），行列数与论文的 12,028 不完全一致；
2. "holdout-sites" 目前退化为 pair 分组（工作簿无 site 字段）；
3. 评分对象是 **VI 判定值**——这是唯一能反映 θ 影响的量；若改用检测器标签，
   三臂会恒等（见 `outputs/exp3_crossfit_categoryonly/`，该对照已保留）。
