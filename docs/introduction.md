# Introduction 草稿（含写作脉络备忘）

> 用途：与 related_work.md 配套的第一章底稿。英文段落可直接改写进论文；
> 中文批注写作时删除。实验数字均为占位符 [XX]，等 GPU 正式实验后回填。

---

## Intro 的段落骨架（SIGCOMM/NSDI 式，每段一个职责）

| 段落 | 职责 | 一句话内容 |
|---|---|---|
| P1 | 背景+重要性 | LEO 星座规模化，TE 是吞吐/拥塞的关键控制手段 |
| P2 | 现状+隐含假设 | 学习式 TE 已把求解降到毫秒级，但全部假设完整 TM 可得 |
| P3 | 打破假设（核心矛盾） | 卫星平台测量成本高 → 现实是稀疏观测；零填充/两阶段补全都不行 |
| P4 | 挑战拆解 | C1 缺失表征 C2 动态集合 C3 规模 |
| P5 | 我们的方案 | 逐条回应挑战的模块设计 |
| P6 | 贡献列表 | 3-4 条 bullet |
| （P7） | 结果亮点 | 关键数字，可并入 P6 |

---

## P1 背景与重要性

**English draft:**

Low-Earth-orbit (LEO) satellite constellations are rapidly scaling toward tens of
thousands of satellites, carrying user traffic with pronounced spatial and temporal
fluctuations over inter-satellite links (ISLs) of limited capacity. Without global
coordination, shortest-path or purely distributed routing concentrates traffic on a
few congested ISLs, leaving network capacity underutilized. Traffic engineering
(TE) — centrally optimizing how each demand is split across candidate paths —
has therefore become a key control primitive for satellite networks, just as it has
been for cloud wide-area networks (WANs) over the past decade.

**中文批注：**
- 开篇即卫星（不是从 WAN 讲起再切换），因为我们的贡献锚定卫星场景；WAN 只作类比。
- "limited ISL capacity + 时空波动流量" 两个名词埋下后文拥塞/稀疏的伏笔。
- 素材来源：docx 第一段（拓扑变化快、传统路由缺乏全局视角）。

## P2 现状与隐含假设

**English draft:**

A recent line of learning-based TE systems has made centralized control practical
at scale: by replacing online optimization solvers with neural models, they cut
allocation time from minutes to milliseconds on WANs [Teal, DOTE, HARP] and, most
recently, on satellite constellations [SaTE]. These systems, however, share a
silent premise: the *complete* traffic matrix — the demand between every pair of
nodes — is assumed to be measured and delivered to the controller at every
decision epoch.

**中文批注：**
- 一段话把 related work 第 1、2 节压缩成两句，火力集中在"silent premise"上。
- 引用只点最必要的四篇（Teal/DOTE/HARP/SaTE），细节留给 Related Work。

## P3 打破假设（核心矛盾）

**English draft:**

This premise rarely holds in LEO constellations. Network-wide measurement requires
every satellite to meter, aggregate, and downlink fine-grained flow statistics
through bandwidth-constrained telemetry channels, competing with user traffic and
consuming scarce onboard resources; measurement gaps due to intermittent ground
contacts and equipment degradation are the norm rather than the exception. In
practice, the controller observes only a *sparse* traffic matrix — a subset of
demands measured at each epoch. Naive workarounds fall short on both ends: filling
unobserved entries with zeros systematically misleads the allocator into starving
unmeasured demands, while two-stage "complete-then-optimize" pipelines propagate
reconstruction errors into allocation decisions. The state-of-the-art sparse-TE
approach [TEST] fills unobserved entries with zeros and compensates with synthetic
training augmentation, but it targets terrestrial networks with static topologies —
leaving sparse-observation TE for dynamic satellite networks unaddressed.

**中文批注：**
- 逻辑链：测量为什么贵（星上资源/遥测带宽/接触间歇）→ 所以现实是稀疏 →
  两个 naive 方案为什么不行（零填充误导、两阶段误差传播）→ TEST 只解决了一半
  （地面+静态+零填充）→ gap。
- "零填充误导分配"如果后面消融实验能量化（zero-fill vs mask embedding 的 obj
  差距），在这里加一句数据支撑会非常有力，留桩：
  "e.g., degrading satisfied demand by [XX]% in our experiments"。
- TEST 零填充的表述已被官方代码证实（related_work.md 有证据链），可放心写。

## P4 挑战拆解

**English draft:**

Designing a TE system that operates on sparse observations of satellite traffic
raises three challenges. **(C1) Representing the unobserved.** Unobserved demands
are not zero; the model must distinguish "no traffic" from "not measured", infer
missing volumes from history, and prevent placeholder values from contaminating
the shared network representation during message passing. **(C2) Perpetual
dynamics.** Satellites move: topologies, candidate paths, and even the set of
active demand pairs drift over time. Retraining per snapshot is infeasible; the
model must be trained once and reused across snapshots unseen in training.
**(C3) Constellation scale.** A 22×72 constellation induces 2.5 million demand
pairs and tens of millions of path variables if modeled exhaustively — beyond the
memory and latency budget of any online system.

**中文批注：**
- C1↔掩码嵌入+门控+时序补全；C2↔尺寸无关权重+剪枝集合漂移处理；C3↔demand
  剪枝+路径模板。挑战和方案模块一一对应，审稿人核对起来舒服。
- "distinguish no-traffic from not-measured" 是 C1 的灵魂句，也是零填充的死穴。
- C2 呼应我们和用户讨论过的"卫星持续运动、pair 集合漂移"（数据实测 t=100 时
  交集降到 89.6%，可作脚注或实验章节证据）。

## P5 我们的方案

**English draft:**

We present SiTE, a learning-based TE system for LEO constellations that
allocates traffic directly from sparse observations. SiTE builds on the
flow-centric GNN + multi-agent RL backbone of Teal and augments it with three
sparse-aware components. First, unobserved demands are represented by *learnable
mask embeddings* rather than zeros, letting the model learn a prior for missing
traffic (C1). Second, a *mask-aware gated GNN* blocks message passing from
unobserved path nodes to link nodes — preventing placeholder pollution of link
utilization estimates — while still letting unobserved demands sense link states
in the reverse direction (C1). Third, a *Transformer temporal encoder* summarizes
the recent history of sparse observations and fuses it with spatial embeddings
via per-demand cross-attention, recovering missing information from temporal
correlation (C1). For deployment at scale, SiTE prunes the demand set to
pairs with observed traffic — 250× fewer variables on a 22×72 Starlink-like
constellation — and keeps all model weights independent of the demand-set size,
so a model trained once transfers across topology snapshots and drifting demand
sets without retraining (C2, C3). Allocations are finally refined by a
parallelizable ADMM step to repair residual capacity violations.

**中文批注：**
- SiTE 待起名。建议起一个卫星/稀疏相关的短名（如 SparTE、MaskTE、
  StarSparse 之类），定名后全局替换。
- 每个模块句尾标注回应的挑战编号，与 P4 严格对齐。
- "250× fewer"来自 250万→6334 对 ≈ 396×，保守写 250×，正式数字等实验定稿。
- RL 而非 IL 的理由不放 Intro（放 Design/Related Work 的 FNC 防守点）。

## P6 贡献列表

**English draft:**

In summary, this paper makes the following contributions:

- We identify sparse traffic observability as a fundamental yet unaddressed
  constraint for satellite TE, and formulate TE from sparse observations over
  dynamic constellations (§2, §3).
- We design SiTE, which combines learnable mask embeddings, mask-aware
  gated message passing, and temporal-spatial cross-attention to allocate traffic
  directly from sparse inputs, without an explicit completion stage (§4).
- We show how demand pruning and size-invariant weights enable train-once,
  zero-retraining deployment across topology snapshots and drifting demand sets
  (§4, §5).
- On a 22×72 Starlink-like constellation with real topology dynamics,
  SiTE achieves [XX]% of the fully-observed performance with only
  [XX]% observability, outperforming zero-filling and two-stage baselines by
  [XX–XX]%, with [XX] ms inference per snapshot (§6).

**中文批注：**
- 第 4 条全是占位符，等 GPU 正式实验回填；如果 pair 集合分离实验（零重训泛化）
  数字好看，单独加一条贡献。
- 章节号 §2-§6 按最终论文结构再对。

---

## 写作顺序建议（实操）

1. **先定系统名**（全文锚点，影响所有章节）；
2. P4/P5 先写（挑战↔方案对应关系是全文骨架，Related Work 和 Design 都挂在上面）；
3. P3 的说服力依赖"卫星测量为什么贵"的文献支撑——目前这段是合理推断，
   建议补 1-2 篇卫星遥测/测量开销的实测文献（可再让调研模型找：
   "LEO satellite telemetry bandwidth constraint flow measurement overhead"）；
4. P6 数字留到实验全部跑完再填，Intro 最后定稿；
5. 风格提醒：SIGCOMM Intro 通常 1.5 页内、无小标题、每段 5-8 行；上面每段
  草稿长度已按此控制。

## 与其他章节的素材对照表

| Intro 引用的事实 | 证据所在 |
|---|---|
| TEST 零填充 | related_work.md §3（官方代码 L32 证据链） |
| pair 集合漂移 89.6% | 本对话数据分析（starlink_22_72.npz，t=0 vs t=100） |
| 250万→6334 剪枝 | teal_env._get_demand_pairs 实测 |
| 权重尺寸无关 | FlowGNN/actor 代码结构核实（共享权重逐 demand 应用） |
| FNC 不放 Intro | related_work.md §2 防守点批注 |
