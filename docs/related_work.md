# Related Work 草稿（含写作脉络备忘）

> 用途：正式写作时的底稿与记忆恢复文档。英文段落可直接改写进论文；中文批注是脉络、
> 防守点和引用注意事项，写作时删除。
>
> 文献核实状态：本文只收录**已核实**的文献（原文或官方页面确认过）。调研报告中
> 未核实的条目（Tubo/PreTE/DeepTE/Chaos 2026/GraphRoute-Transfer/GUIDED/RIFT-STGNN
> 等）一律不写入正文，除非后续拿到原文。

---

## 结构总览（三条科研脉络 + 定位段）

1. **ML-driven TE for WANs**：从传统优化到学习加速——我们的方法论根基
2. **TE and routing for satellite networks**：卫星场景的动态性处理——我们的场景先例
3. **TE with incomplete traffic observations**：稀疏观测——我们的核心创新所在的空白带
4. **定位段**：与三个最近邻（SaTE / TEST / FNC）的差异表

---

## 1. ML-driven Traffic Engineering for WANs

**English draft:**

Traffic engineering has long been formulated as a constrained optimization problem
solved by linear programming (LP), whose runtime scales poorly with network size.
SMORE [Kumar et al., NSDI'18] decouples robust oblivious path selection from online
rate adaptation, while NCFlow [Abuzaid et al., NSDI'21] and POP [Narayanan et al.,
SOSP'21] accelerate LP by problem decomposition, trading allocation quality for speed.
A recent line of work replaces online solvers with learned models: DOTE [Perry et al.,
NSDI'23] directly maps historical traffic matrices to split ratios with stochastic
optimization; Teal [Xu et al., SIGCOMM'23] combines a flow-centric GNN, multi-agent
RL, and ADMM fine-tuning to achieve near-optimal allocations with orders-of-magnitude
speedups; FIGRET [SIGCOMM'24] enhances robustness against traffic uncertainty at
per-flow granularity. To survive topology changes without retraining, HARP
[AlQiam et al., SIGCOMM'24] enforces invariances to node permutation and tunnel
reordering, transferring across topologies unseen in training. More recently, large
language models have been explored as general-purpose traffic planners [NetLLM,
SIGCOMM'24; LMTE, 2026]. However, all of these systems assume the *complete*
traffic matrix is available at decision time — an assumption that rarely holds in
large-scale satellite networks, where network-wide measurement is prohibitively
expensive.

**中文批注：**
- 脉络：LP 求解慢 → 分解加速（NCFlow/POP，牺牲质量）→ 学习加速（DOTE/Teal）→
  跨拓扑迁移（HARP）→ LLM 化（NetLLM/LMTE）。收尾句把"全观测假设"作为转折点，
  引出我们的问题。
- Teal 是我们的骨架，要多给一句技术细节（FlowGNN + COMA + ADMM），为方法章节铺垫。
- ⚠️ LMTE venue 待确认：我的 docx 写 SIGCOMM 2026，调研报告写 INFOCOM 2026，
  以下载的原文为准。
- ⚠️ FIGRET 作者名未核实，写作时查 DOI。
- NCFlow/POP 的引用信息在 Teal README 里有现成的。

---

## 2. TE and Routing for Satellite Networks

**English draft:**

LEO mega-constellations pose unique challenges for TE: topologies change every few
tens of milliseconds as satellites move, and control decisions must be produced
within stringent latency budgets. Classical approaches absorb mobility through
topology abstractions — the virtual node model binds logical nodes to earth-fixed
footprints so that upper-layer routing observes a quasi-static grid [Lu et al.,
IEEE Comm. Letters 2013]. SaTE [Wu et al., SIGCOMM'25] is the state-of-the-art
learning-based TE for satellite constellations: it formulates a heterogeneous graph
over satellites, demands, paths, and links, prunes zero-traffic relations to make
training tractable on a single GPU, and selects representative topologies via
Graph2Vec+DPP sampling so that one trained model generalizes across topology
snapshots with millisecond inference. In parallel, FNC [Yang et al., IEEE/ACM ToN
2026; earlier version in INFOCOM'25] targets dynamic traffic allocation for earth-
observation constellations, replacing RL with end-to-end imitation learning and
substituting iterative constraint solvers with normalization-based feasibility
layers. Notably, FNC's imitation learning constructs its expert supervision by
solving an offline optimum over the *fully realized* demands of each episode —
a paradigm that fundamentally requires complete post-hoc observability, which is
unavailable under sparse measurement. Both SaTE and FNC, like their WAN
counterparts, assume the traffic matrix is fully observable at decision time.

**中文批注：**
- 脉络：VN 抽象（经典，2013，被引 119）→ SaTE（学习式卫星 TE 的 SOTA，全观测）→
  FNC（同期，观测星座数据下行场景，IL 范式）。
- **FNC 防守点（重要）**：它批评 RL 训练低效（比 Teal 好 8% 满足率、10× 延迟）。
  我们的回应写在这段末尾+定位段：IL 的专家解需要"事后完整 demand"求离线最优
  （FNC 原文 Sec. V：every episode 结束后 construct offline DTA and solve it），
  稀疏观测下事后也无完整 TM，专家信号不可得或有偏；reward 驱动的 RL 只需网络
  反馈，天然适配部分可观测。且我们的输入端模块（掩码嵌入+时序补全）与训练范式
  正交，未来可嫁接 IL——可放 Discussion 作展望。
- FNC 场景差异也要点明：earth-observation 数据回传 ≠ 通信星座用户流量 TE。
- FNC 团队（上海交大+上海卫星工程研究所）在该方向持续发表（INFOCOM'25→ToN'26），
  可能是审稿人，行文保持 respectful、区分要技术性。
- SaTE 的剪枝给我们的 demand 剪枝提供直接背书；引用时强调它"prunes zero-traffic
  relations"与我们"prune zero-demand pairs"的一致性与差异（它剪的是图关系，
  我们剪的是建模对象集合，并显式处理集合随时间漂移的部署问题）。
- 其他卫星 ML 路由（DRL/DQN 类）技术层次较低，可用一句话带过，不逐篇引。
- 可顺藤引用（背景）：Falcon [INFOCOM'23]、"Transmitting, Fast and Slow"
  [MobiCom]——对地观测调度方向，非竞争。

---

## 3. TE with Incomplete Traffic Observations

**English draft:**

Network-wide traffic measurement is expensive: collecting a complete traffic matrix
requires instrumenting all ingress nodes at fine time granularity, which is
particularly onerous on resource-constrained satellite platforms. A classical
workaround is a two-stage pipeline — first estimate or complete the traffic matrix
(network tomography, matrix completion), then optimize routing on the estimate —
but reconstruction errors propagate to and degrade the downstream allocation.
TEST [Guo et al., IEEE TMC 2026] is, to our knowledge, the first end-to-end
approach that learns routing policies directly from sparsely measured traffic
matrices, integrating a Transformer over historical sparse TM sequences with a GCN
over the topology; it targets terrestrial networks with static topologies. In the
graph learning literature, message passing under missing node features has been
addressed via teacher-student distillation [T2-GNN, AAAI'23], but such techniques
have not been applied to network TE. No prior work, to our knowledge, performs
TE from sparse traffic observations in satellite networks, where the difficulty
is compounded by time-varying topologies and drifting demand sets.

**中文批注：**
- 脉络：测量开销大 → 两阶段（补全+优化，误差传播）→ TEST 端到端稀疏 TE（地面、
  静态拓扑）→ 空白：卫星场景。最后一句话就是我们的 claim。
- TEST 是最近邻，差异点：a) 地面 vs 卫星；b) **缺失处理方式已经过官方代码确认
  （github.com/paper-TEST/TEST，run_TEST.py L32：`obs_tms = tms * mask`）：
  未观测位置直接置零，靠合成 TM 数据增强（gen_node/flow_mgm_tms）提升鲁棒性；
  无可学习掩码嵌入、无掩码感知门控**——这是我们方法层面最硬的差异，可写
  "TEST fills unobserved entries with zeros and relies on synthesized-TM
  augmentation, whereas we replace them with learnable mask embeddings and
  gate their message passing"；c) 无需求集合漂移问题；d) 我们有时序-空间
  交叉注意力融合。
- **TEST 的采样模式和 baseline 集合可直接复用到我们实验设计**：
  1. 它支持 node 级（测量部分节点）和 flow 级（测量部分流）两种稀疏采样；
     我们目前只有 demand(pair) 级，建议补 node 级（卫星场景语义=部分卫星可测，
     更贴近部署且与 TEST 可比）；
  2. 它的 baseline 集：DOTE、NTC（神经张量补全，两阶段代表）、MTSR-CS
     （压缩感知）、PPO、Oblivious(Räcke)、Mean Interpolation（均值插值，
     最简两阶段）——这就是稀疏 TE 的标准 baseline 集，我们卫星实验至少覆盖
     MI（易实现）+ 一个补全类方法作两阶段对照。
  仓库已 clone 到本地 /tmp/TEST 可随时参考实现。
- T2-GNN 只作方法论借鉴引用，措辞用 "have not been applied to network TE"
  这种可辩护的说法，避免绝对化否定。
- **T2-GNN 原文已读（arXiv:2212.12738，AAAI 2023，天津大学 Cuiying Huo 等），
  两处可直接用作我们的设计依据：**
  1. **Eq.5 参数化补全**：缺失特征用可学习参数 Θ_ij 填充而非填零——与我们的
     可学习掩码嵌入同源，方法章节可引作 learnable imputation 的先例；
  2. **特征-结构互相干扰结论**：不完整特征经消息传递会污染结构信息（它因此
     分开双教师训练）——直接支撑我们 Sparse-FlowGNN 门控（未观测 path 不向
     edge 传消息）的设计动机，方法章节引用。
  差异保持准确：它是静态图节点分类+蒸馏，我们是 TE 优化+在线推理+时序维度。
- 补全/tomography 的具体引文写作时再补（经典如 Zhang et al. 的 tomography 工作），
  这里先留桩。

---

## 4. 定位段 / 差异表

**English draft (positioning paragraph):**

Our work sits at the intersection of these three lines. Unlike SaTE and FNC, which
assume fully observable demands, we perform TE from sparse observations (10%–70%
observability). Unlike TEST, which targets static terrestrial networks, we handle
LEO dynamics — time-varying topologies and drifting demand-pair sets — via
demand pruning with size-invariant model weights, enabling train-once,
zero-retraining deployment. Architecturally, we extend Teal's flow-centric GNN
with (i) learnable mask embeddings in place of zero-filling, (ii) mask-aware
gated message passing that prevents unobserved demands from polluting link
embeddings, and (iii) a Transformer over historical sparse observations fused
with spatial embeddings via cross-attention.

**差异表（论文可做成 Table 1）：**

| 维度 | Teal (SIGCOMM'23) | SaTE (SIGCOMM'25) | TEST (TMC 2026) | FNC (ToN 2026) | **Ours** |
|---|---|---|---|---|---|
| 场景 | 地面 WAN | 通信星座 | 地面 WAN | 观测星座下行 | 通信星座 |
| 拓扑动态 | ✗ | ✓ | ✗ | ✓ | ✓ |
| 稀疏观测输入 | ✗ | ✗ | ✓ | ✗ | ✓ |
| 时序建模 | ✗ | ✗ | ✓ Transformer | ✗ | ✓ Transformer+交叉注意力 |
| 缺失处理 | — | — | 合成 TM 增强 | — | 可学习掩码嵌入+门控 GNN |
| 训练范式 | MARL | 监督/GNN | 监督 | IL（需事后完整观测） | MARL（适配部分可观测） |
| 约束处理 | ADMM | 约束修正 | — | 归一化可行层 | ADMM |
| 需求集合漂移 | — | 拓扑采样间接处理 | — | — | 需求剪枝+尺寸无关权重 |

**一句话定位（来自调研报告，已验证无撞车）：**

> 首个在 LEO 卫星网络中实现稀疏流量观测下端到端 TE 的学习框架：掩码感知 GNN +
> 历史时序补全，10%–70% 观测率下逼近全观测性能，训练一次、拓扑变化零重训。

---

## 已核实引用信息速查

| 简称 | 完整引用 |
|---|---|
| Teal | Xu et al., "Teal: Learning-Accelerated Optimization of WAN Traffic Engineering," ACM SIGCOMM 2023, pp. 378–393. DOI: 10.1145/3603269.3604857 |
| SMORE | Kumar et al., "Semi-Oblivious Traffic Engineering: The Road Not Taken," USENIX NSDI 2018, pp. 157–170 |
| DOTE | Perry et al., "DOTE: Rethinking (Predictive) WAN Traffic Engineering," USENIX NSDI 2023, pp. 1557–1581 |
| HARP | AlQiam et al., "Transferable Neural WAN TE for Changing Topologies," ACM SIGCOMM 2024, pp. 86–102. DOI: 10.1145/3651890.3672237 |
| SaTE | Wu et al., "SaTE: Low-Latency Traffic Engineering for Satellite Networks," ACM SIGCOMM 2025, pp. 896–916. DOI: 10.1145/3718958.3750524 |
| TEST | Guo et al., "An End-to-End Learning Approach for Traffic Engineering With Sparse Traffic Measurements," IEEE TMC, vol. 25, no. 4, pp. 5448–5463, 2026. DOI: 10.1109/TMC.2025.3628341 |
| FNC | Yang et al., "Scalable Traffic Allocation in Dynamic Networks via End-to-End Imitation Learning," IEEE/ACM ToN, vol. 34, 2026. DOI: 10.1109/TON.2026.3712057 |
| FNC 会议版 | Yang et al., "Learning to Accelerate Traffic Allocation over Large-Scale Networks," IEEE INFOCOM 2025. DOI: 10.1109/INFOCOM55648.2025.11044775 |
| VN 模型 | Lu et al., "Virtual Topology for LEO Satellite Networks Based on Earth-Fixed Footprint Mode," IEEE Communications Letters, vol. 17, no. 2, pp. 357–360, 2013 |
| T2-GNN | Huo et al., "T2-GNN: Graph Neural Networks for Graphs with Incomplete Features and Structure via Teacher-Student Distillation," AAAI 2023 |
| NCFlow | Abuzaid et al., "Contracting Wide-area Network Topologies to Solve Flow Problems Quickly," USENIX NSDI 2021 |
| POP | Narayanan et al., "Solving Large-Scale Granular Resource Allocation Problems Efficiently with POP," ACM SOSP 2021 |

**待核实（写作前必须确认，暂不引用）：**
- LMTE：venue（SIGCOMM vs INFOCOM 2026）与作者
- FIGRET（SIGCOMM 2024）：作者与页码
- NetLLM（SIGCOMM 2024）：作者与页码
- ENERO：venue（我记忆是 Computer Networks 2022，调研报告写 JSAC，二选一）
- RouteNet-Fermi：venue（ToN 2023 vs JSAC 2022）
- DisCoRoute：全名与 venue（印象中 Fraire 等人，约 2022）
- 补全/tomography 经典引文（留桩未填）
