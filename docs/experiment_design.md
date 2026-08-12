# 实验设计备忘（六篇导向，不含 ELATE）

> 简要存档，供实验阶段参考。论文写作暂不改动——等实验效果定型再改。

## 参考论文与借鉴点

实验设计只参考以下六篇，**不参考 ELATE 的数据和呈现方式**：

| 论文 | 借鉴点 |
|---|---|
| TEAL (SIGCOMM'23) | 代码基础；satisfied demand ratio、runtime、ADMM |
| DOTE (NSDI'23) | 归一化到 omniscient oracle；报 quality + runtime |
| TEST | PR = U/U_opt；k-shortest paths（非 edge-disjoint）；稀疏观测 |
| HARP (SIGCOMM'24) | **百分位呈现**："MLU ≤ optimal+11% over 98% of time"；拓扑变化鲁棒 |
| SaTE (SIGCOMM'25) | 卫星星座；异构图+剪枝；satisfied demand%、runtime(ms)、speedup(×) |
| LMTE (SIGCOMM'26) | 变化下 degradation < 5%；MLU；speedup 10–100× |

## 关键决策

1. **主指标 = PR = U / U_opt**（六篇里五篇都是"归一化到最优"）。
2. **呈现用百分位/CDF 而非 mean±std**（学 HARP）：报"X% 的情况下 PR ≤ Y"。
   好处：单次训练发散的离群 seed（如 seed 2 卡在 3.20）天然被隔离，不污染主结论。
3. **路径用 k-shortest（`--shared-paths`），非 edge-disjoint**。
   实测：edge-disjoint 下每个 demand 的 4 条路径 0% 共享链路 → 均匀分流已近最优、学习无空间；
   k-shortest 下 100% 共享 → 路径选择才有意义。六篇全用最短路系。
4. **确定性 GPU（`--deterministic`）**：方法间信号仅 2–3%（TEST 同量级），必须压掉 ~5% 的
   GPU 原子操作噪声才能显现。
5. **故事定位 = 鲁棒性 + 稀疏观测，不是"学习绝对增益"**。
   HARP/LMTE 的卖点都是"变化下退化小"，不是"打败基线很多"。
   我们的独特轴：**稀疏观测 + 卫星 + 零重训**（六篇中无人覆盖此组合）。

## 实验矩阵

| 实验 | 对标 | 指标 |
|---|---|---|
| 主表：PR vs ρ（稀疏度扫描 0.5/0.3/0.1 + 全观测） | TEST | PR，百分位 |
| 拓扑漂移零重训（Drop5/Drop10） | HARP | X% 内 ≤ 最优 Y% |
| 需求漂移（demand-split）+ 链路故障（failures） | LMTE | degradation % |
| 模块消融（去掉 mask embed / gate / temporal） | HARP | PR gap |
| 运行时 + 加速比（vs LP solver） | SaTE/DOTE | ms、× |

baseline 集合：LP oracle、zero-fill、mean-interp、Teal-full（完整观测上界）。

## 已知问题

- **训练稳定性**：5 seed 里可能有 1 个坏初始化（seed 2 实测卡在 val 3.20 vs 正常 1.5）。
  对策：百分位呈现 + 增加 seed 数（5→7），用中位数/分位数而非均值。
- capacity 定标：MLU 目标下 k-shortest 的 LP oracle ≈ 1.38（capacity 2000）。
