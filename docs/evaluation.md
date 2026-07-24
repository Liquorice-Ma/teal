# Evaluation（第 6 章）草稿（含实验设计与表格骨架）

> 用途：实验章节底稿。所有数字为 [XX] 占位，GPU 实验完成后回填；每个实验给出
> 研究问题（RQ）、表格骨架和精确复现命令。英文段落为可用初稿，中文批注写作时删除。

---

## 章节骨架与研究问题

```
§6 Evaluation
  6.1 Setup                  — 拓扑/流量/切片/指标/种子协议/硬件
  6.2 Overall Performance    — RQ1: 稀疏观测下比 baseline 好多少？
  6.3 Ablation Study         — RQ2: 三个模块各贡献多少？
  6.4 Zero-Retraining        — RQ3: 需求集合漂移下权重能否直接复用？
  6.5 Runtime & Scale        — RQ4: 推理延迟满足卫星控制周期吗？剪枝省多少？
  6.6 Sensitivity            — RQ5: 观测粒度(flow/node)、hist-len、容量负载的影响
```

---

## 6.1 Setup

**English draft:**

**Topology and traffic.** Our primary testbed is a Starlink-like +Grid
constellation with 22 orbits × 72 satellites (1,584 nodes, 6,336 ISLs), with
101 consecutive traffic-matrix snapshots capturing real geographic demand
distribution and temporal drift (demand-pair turnover of ~10% across the
trace). ISL capacity is calibrated so that the fully-observed optimum leaves
moderate congestion (satisfied ratio ≈ 0.87), ensuring TE decisions are
non-trivial. We use snapshots 0–79 for training, 80–89 for validation, and
90–100 for testing — a strict temporal split with no leakage.

**Metrics.** (i) *Satisfied demand ratio*: allocated flow (after ADMM and
rounding, evaluated against ground-truth demands) divided by total demand;
(ii) *inference latency* per snapshot. All results average 3 random seeds;
we report mean ± std.

**Baselines.** Following the taxonomy of §3.4: (a) *Zero-fill*: unobserved
entries set to zero, no mask-aware modules (TEST-style treatment);
(b) *Mean-interpolation (MI)*: two-stage complete-then-optimize;
(c) *Teal-full*: original Teal with the complete TM — an upper reference,
not a competitor; (d) ours with all modules enabled.

**中文批注：**
- 容量定标 0.87 的合理性写一句就够（避免"为什么选 1000"的追问）。
- 严格时序切片（无泄漏）主动声明，demand-split 实验会再强调一次。
- baseline 分组严格对应 3.4 三分法 + 上界；若后续实现 NTC/压缩感知类补全，
  归入 (b) 组。
- 硬件一句话：RTX 4090 (24GB) / CPU 型号，写作时补。

## 6.2 Overall Performance（RQ1）

**Table 1 骨架：Satisfied demand ratio vs. observability ρ（Starlink2272）**

| Method | ρ=1.0 | ρ=0.7 | ρ=0.5 | ρ=0.3 | ρ=0.1 |
|---|---|---|---|---|---|
| Teal-full（上界，ρ=1.0 一格） | [XX] | — | — | — | — |
| Zero-fill | — | [XX] | [XX] | [XX] | [XX] |
| MI (two-stage) | — | [XX] | [XX] | [XX] | [XX] |
| **Ours** | — | [XX] | [XX] | [XX] | [XX] |

**期望叙事**：Ours 在 ρ=0.5 时达到上界的 [XX]%，比 zero-fill 高 [XX]%，比 MI
高 [XX]%；ρ 越低差距越大（缺失越多，掩码嵌入+时序补全的价值越大）。

**命令模板**：
```bash
# ours:      --mask-mode embed --hist-len 3
# zero-fill: --mask-mode zero --hist-len 1 --no-gate
# MI:        --mask-mode mean --hist-len 1 --no-gate
# 上界:      --obs-ratio 1.0 --hist-len 1
python teal.py --obj total_flow --topo Starlink2272.json --tm-model starlink \
  --epochs 100 --admm-steps 2 --prune-demands --obs-ratio {ρ} --seed {0,1,2} \
  --slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
  --slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101 {模式开关}
```

## 6.3 Ablation Study（RQ2）

**Table 2 骨架：模块消融（ρ=0.3，Starlink2272）**

| 配置 | mask embed | gate | temporal | obj | Δ vs full |
|---|:---:|:---:|:---:|---|---|
| Full (ours) | ✓ | ✓ | ✓ | [XX] | — |
| − temporal (`--hist-len 1`) | ✓ | ✓ | ✗ | [XX] | [XX] |
| − gate (`--no-gate`) | ✓ | ✗ | ✓ | [XX] | [XX] |
| − embed (`--mask-mode zero`) | ✗ | ✓ | ✓ | [XX] | [XX] |
| none (zero-fill) | ✗ | ✗ | ✗ | [XX] | [XX] |

**中文批注**：消融选 ρ=0.3（缺失多，模块差异最可见）；如果 B4/CPU 阶段那种
"分不出差异"在大拓扑上重现，如实报告并分析（可能结论：嵌入与门控在高时序
相关流量下部分冗余——这也是有价值的发现，别硬凹）。

## 6.4 Zero-Retraining Generalization（RQ3）

**English draft:**

We train with the demand-pair union of the training slice (6,320 pairs) and,
at test time, rebuild the graph from the test slice's own union (6,044 pairs,
including pairs never seen in training) while keeping the weights frozen
(`--demand-split`). We compare against the oracle setting where the demand
set is built from all snapshots.

**Table 3 骨架：**

| 设定 | 训练集合 | 测试集合 | obj | Δ |
|---|---|---|---|---|
| Oracle（全集合） | 6,334 | 同左 | [XX] | — |
| **Zero-retraining（split）** | 6,320 | 6,044（重建） | [XX] | [XX] |

初步数据点（CPU，10 epochs）：split 0.9078 vs oracle 0.9087，几乎零损失——
正式实验预期同量级。若确认，这是"train once, reuse under drift"贡献的直接证据。

## 6.5 Runtime & Scale（RQ4）

- 推理延迟表：ours vs Teal-full vs（若有）LP 求解器，Starlink2272 单快照
  [XX] ms（CPU 实测 15ms，GPU 预期更低）；对照 SaTE 报告的 17ms 量级。
- 剪枝效果：2.5M pairs → 6.3K（~400×），显存 [XX] GB → [XX] GB；
  不剪枝的资源需求用分析性估算（edge_index ≈8GB + 激活）说明不可行，不实跑。
- 时序/融合模块的额外延迟：hist-len 1 vs 3 的 runtime 差（CPU 实测
  0.7ms→1.1ms 量级，GPU 重测）。

## 6.6 Sensitivity（RQ5）

- **观测粒度**：flow vs node 采样在同 ρ 下的 obj 对比（node 级缺失有结构性，
  预期更难，我们的空间消息传递应更占优——验证或推翻都值得写）；
- **hist-len 扫描**：1/3/5/8，收益饱和点；
- **容量负载**：capacity 500/1000/2000 下的相对增益（高负载下 TE 空间更大）。

**中文批注**：6.6 是缓冲区——审稿人常见问题先自问自答；篇幅不够可砍到只留
观测粒度一个。

---

## 与前文占位符的回填清单

| 占位符位置 | 来源实验 |
|---|---|
| intro P3 "zero-fill degrades by [XX]%" | Table 2 最后一行 vs Full |
| intro P6 贡献 4 全部数字 | Table 1 + 6.5 |
| design 4.4 "[XX] ms" | 6.5 时序模块延迟 |
| design 4.6 "[XX] ADMM iterations" | admm-steps 消融（定稿用 2） |
