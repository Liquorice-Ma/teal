# Conclusion（第 7 章）草稿（含局限与未来工作备忘）

> 用途：结论章底稿。英文段落可直接改写进论文；中文批注写作时删除。
> 结论章通常半页内：总结（1 段）+（可选）局限与未来工作（1 段或并入 Discussion）。

---

## 7.1 Conclusion

**English draft:**

This paper presented SiTE, the first learning-based traffic
engineering system for LEO satellite constellations that operates directly on
sparse traffic observations. SiTE resolves the ambiguity of
unmeasured demands with learnable mask embeddings, protects network-state
estimates with mask-aware gated message passing, and recovers missing
information from observation history through temporal-spatial cross-attention
— all trained end-to-end against the TE objective without an explicit
completion stage. Demand pruning and size-invariant weights let a model
trained once serve drifting topologies and demand sets with zero retraining.
On a 22×72 Starlink-like constellation with real traffic dynamics,
SiTE achieves [XX]% of fully-observed performance with only [XX]%
observability, outperforming zero-filling and two-stage baselines by
[XX–XX]%, at [XX] ms inference per snapshot.

**中文批注：**
- 结论段 = intro 贡献列表的"过去时"复述 + 最硬的数字，无新信息，写作最省力，
  等实验数字回填后 10 分钟完成。
- "first" 声明与调研报告 Q1 结论一致（已验证空白），保留。

## 7.2 Limitations and Future Work（可并入 Discussion）

**English draft:**

Three directions remain open. First, our observation masks are static within
a trace; real measurement availability fluctuates with ground-contact windows
and onboard load, motivating time-varying masks and measurement scheduling —
deciding *which* demands to measure next — as a joint control problem.
Second, our input modules are orthogonal to the training paradigm: combining
them with imitation learning [FNC] where post-hoc complete traces are
available, or with hybrid IL-RL schemes, may improve sample efficiency.
Third, the path templates of +Grid constellations suggest an O(1)
displacement-indexed path lookup that we currently approximate with per-set
path caching; a full implementation would further cut graph-rebuild latency.
We also plan to validate on measured (rather than synthesized) satellite
traffic as such datasets become available.

**中文批注：**
- 局限 1（时变掩码/测量调度）其实是下一篇论文的 idea，点到为止；
- 局限 2 呼应 FNC 防守点的"正交性"承诺（related_work/design 三处一致）；
- 局限 3 措辞已避免过度承诺（"currently approximate with per-set caching"）；
- 主动写局限是顶会加分项，但每条都要带"可解"的方向，不留死穴。

---

## 全文文档索引（docs/ 完成状态）

| 章节 | 文件 | 状态 |
|---|---|---|
| §1 Introduction | introduction.md | 草稿 ✓（数字占位） |
| §2 Related Work | related_work.md | 草稿 ✓（已核实文献） |
| §3 Formulation | formulation.md | 草稿 ✓ |
| §4 Design | design.md | 草稿 ✓（4.1 待瘦身） |
| §6 Evaluation | evaluation.md | 框架 ✓（等 GPU 数据） |
| §7 Conclusion | conclusion.md | 草稿 ✓（数字占位） |
| 操作指南 | operation_guide.md | ✓ |

**定稿顺序建议**：GPU 实验 → 回填 evaluation → 回填 intro P6/P3 和 design
[XX] → conclusion 数字 → 全文统一系统名和符号（K→κ）→ 中文批注删除。
