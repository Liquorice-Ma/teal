# Design（核心创新模块）草稿（含写作脉络备忘）

> 用途：论文第 4 章（System Design）底稿。英文段落可直接改写进论文；中文批注
> 写作时删除。所有设计描述与代码严格对应（文末有代码映射表），不含未实现内容。

---

## 章节骨架

```
§4 [SYSTEM-NAME] Design
  4.1 Overview & Problem Formulation     — 稀疏观测 TE 问题定义 + 流水线总览
  4.2 Mask-Aware Demand Representation   — 创新 1：可学习掩码嵌入（C1）
  4.3 Gated Message Passing              — 创新 2：掩码感知门控 GNN（C1）
  4.4 Temporal-Spatial Fusion            — 创新 3：时序补全 + 交叉注意力（C1）
  4.5 Scaling to Dynamic Constellations  — 创新 4：需求剪枝 + 零重训部署（C2/C3）
  4.6 Training and Constraint Repair     — 训练（MARL）与 ADMM（复用，非创新）
```

---

## 4.1 Overview & Problem Formulation

**English draft:**

We consider TE over a constellation graph G = (V, E) with per-ISL capacity c_e.
At each epoch, traffic demands are described by a matrix D, of which the
controller observes only a subset: a binary mask M marks each demand pair as
measured (M_i = 1) or unmeasured (M_i = 0), yielding the sparse observation
D ⊙ M. Following path-based TE, each demand d_i is served by K pre-selected
candidate paths, and the system outputs split ratios over these paths to
maximize total satisfied demand under capacity constraints. Crucially, the
allocator must decide for *all* active demands — including unmeasured ones —
using only the sparse observation and a short history of past sparse
observations {D^(t-L+1) ⊙ M, ..., D^(t) ⊙ M}.

[SYSTEM-NAME] processes a sparse observation in four stages: (i) unmeasured
demands are represented by a learnable mask embedding (§4.2); (ii) a mask-aware
gated GNN propagates information between candidate paths and links without
letting placeholders contaminate link states (§4.3); (iii) a Transformer
encodes the observation history and fuses it with spatial embeddings via
per-demand cross-attention (§4.4); (iv) a shared policy head outputs split
ratios per demand, refined by ADMM (§4.6). All learnable weights are shared
across demands and independent of the demand-set size, which §4.5 exploits for
zero-retraining deployment under constellation dynamics.

**中文批注：**
- 问题定义的关键点：决策对象是**全部** active demand，但输入只有稀疏观测——
  一句话把"难在哪"钉死。
- 最后一句提前埋"权重尺寸无关"，为 4.5 的零重训做铺垫。
- 这里应配 Figure（流水线图）：稀疏 TM+mask → 掩码嵌入 → Sparse-FlowGNN ↔
  TemporalEncoder → cross-attention 融合 → per-demand policy → ADMM。

## 4.2 创新 1：Mask-Aware Demand Representation（可学习掩码嵌入）

**English draft:**

A sparse observation is fundamentally ambiguous: a zero entry may mean "no
traffic" or "not measured". Zero-filling — the de-facto treatment in prior
sparse TE [TEST] — collapses the two cases and systematically biases the
allocator toward starving unmeasured demands. [SYSTEM-NAME] instead assigns
each unmeasured demand a *learnable mask embedding* θ ∈ R^K (one scalar per
candidate path):

    x_i = M_i · d_i + (1 − M_i) · θ,

so the model learns a data-driven prior for missing traffic during end-to-end
training, and — critically — the input itself distinguishes "not measured"
from "zero". Learnable imputation has proven effective for incomplete graph
features [T2-GNN]; we bring it to TE, where the imputed value directly steers
resource allocation. The embedding is trained jointly with the policy under
the TE objective rather than a reconstruction loss, so θ converges to values
that maximize allocation quality, not reconstruction accuracy.

**中文批注：**
- 卖点句："trained under the TE objective, not a reconstruction loss"——这是
  与两阶段补全方法的本质区别，务必保留。
- θ 是 num_path 维（每候选路径一个标量），对应 `teal_actor.mask_embedding`。
- 消融锚点：`--mask-mode zero/mean/embed` 三档对照，实验章节引用。
- T2-GNN 引用作先例（related_work.md 已核实 Eq.5 参数化补全）。

## 4.3 创新 2：Mask-Aware Gated Message Passing（掩码感知门控）

**English draft:**

[SYSTEM-NAME] inherits Teal's flow-centric graph: candidate paths and physical
links form a bipartite structure where a PathNode connects to the EdgeNodes it
traverses, and message passing alternates between them. Under sparse
observation, however, symmetric message passing is harmful: placeholder values
on unmeasured PathNodes would propagate into EdgeNode embeddings and corrupt
the network-wide estimate of link utilization — a phenomenon consistent with
the feature-structure interference observed in incomplete graphs [T2-GNN].
We therefore gate the propagation *asymmetrically*. Let A be the normalized
bipartite adjacency; we zero out entries that carry path-to-edge messages from
unmeasured paths while keeping all edge-to-path entries:

    Ã[e, p] = M_p · A[e, p]   (path → edge, gated)
    Ã[p, e] = A[p, e]         (edge → path, kept)

Unmeasured demands thus remain *listeners*: they sense link states — which are
shaped by measured traffic — but inject no fabricated volume into them. The
gate is a fixed binary function of the mask with no extra parameters, adding
zero inference overhead.

**中文批注：**
- "listeners"（只听不说）是这个模块最形象的表述，建议保留。
- 非对称性是精髓：双向都砍会让未观测 demand 变成孤岛（无法感知拥塞，分配
  会瞎）；只砍 path→edge 才对。写作时可加一句 rationale。
- 对应 `FlowGNN._gate_index_values()`：edge_index 后半段（row=edge, col=path）
  按 demand 级 mask 置零。
- 消融锚点：`--no-gate`。
- 零参数、零开销——审稿人喜欢的性质，点明。

## 4.4 创新 3：Temporal-Spatial Fusion（时序补全 + 交叉注意力）

**English draft:**

Satellite traffic exhibits strong short-term temporal correlation; the recent
history of sparse observations therefore carries recoverable information about
currently missing entries. [SYSTEM-NAME] encodes, for each candidate path, its
observation sequence over the last L epochs with a lightweight Transformer:
scalar volumes are log-compressed and projected to d_model dimensions, summed
with learnable positional embeddings, and passed through a Transformer encoder;
the representation at the latest step forms the temporal embedding H_T.
Rather than concatenating temporal and spatial features naively, we fuse them
with *per-demand cross-attention*: within each demand, temporal embeddings of
its K candidate paths act as queries attending to the K spatial (GNN)
embeddings as keys and values,

    H_F = softmax(Q_T K_S^T / √d) V_S,   with Q_T = W_q H_T,
    K_S = W_k H_S,  V_S = W_v H_S,

and the fused features [H_S ‖ H_F] feed the policy head. Intuitively, the
temporal stream decides *which* spatial evidence to trust for each path: when
the current entry is missing, attention shifts toward paths and links whose
states corroborate the historical pattern. The whole module adds [XX] ms to
inference on our largest constellation (§6).

**中文批注:**
- log1p 压缩要在正文或脚注提一句（训练稳定性），我们有实测教训（不加则部分
  种子发散）——可写 "log compression is essential for training stability"。
- 交叉注意力是 per-demand 的（K×K 注意力矩阵，K=4），计算量极小——和"哪条
  路径的空间证据可信"这个直觉绑定，别写成泛泛的 feature fusion。
- 对应 `TemporalEncoder.py` + `teal_actor.forward` 的 fuse_q/k/v。
- 消融锚点：`--hist-len 1`（关时序）vs `3/5`。
- [XX] ms 占位，GPU 实验后回填。

## 4.5 创新 4：Scaling to Dynamic Constellations（剪枝 + 零重训部署）

**English draft:**

**Demand pruning.** Modeling all N(N−1) pairs of a 1,584-satellite
constellation yields 2.5M demands and tens of millions of path variables —
infeasible for online control. Satellite traffic, however, is geographically
concentrated: only 0.25% of pairs ever carry traffic in our dataset.
[SYSTEM-NAME] instantiates PathNodes only for the demand-pair union observed
in historical TMs, shrinking the graph by ~400× with no loss of served
traffic, in the same spirit as SaTE's traffic pruning [SaTE].

**Zero-retraining under drift.** Pruned demand sets are not static: as
satellites move, the active pair set drifts (we measure ~10% turnover across
our trace, §6). Rather than retraining, [SYSTEM-NAME] exploits a structural
property of its architecture: every learnable component — GNN layers, mask
embedding, temporal encoder, fusion and policy heads — is shared across
demands and independent of the demand-set size. The demand set, candidate
paths, and the bipartite graph are runtime *inputs*, not parameters. At each
control epoch the controller rebuilds the graph from the current active set
(a millisecond-scale index operation on +Grid constellations, where candidate
paths depend only on the orbital displacement between endpoints and reduce to
O(N) path templates) and reuses the trained weights as-is. Demands outside
the modeled set — necessarily sporadic mice flows — fall back to shortest-path
routing until the next set refresh.

**中文批注：**
- 剪枝合法性证据链：数据实测 0.25% 非零 + SaTE 先例；漂移证据：89.6% 交集
  （t=0 vs t=100），写实验章节，这里引用。
- "runtime inputs, not parameters" 是全段灵魂句。
- 路径模板（位移不变性）目前是设计论证 + 文献背书（时不变拓扑与航点路由），
  代码里是按集合现算+缓存——写作时措辞别过度承诺 O(1) 查表已实现。
- 兜底最短路对应 docx 里"老鼠流走最短路"的设计，闭环了当初的讨论。
- 实验锚点：`--demand-split`（训练 6320 对 / 测试 6044 对，首测零损失）。

## 4.6 Training and Constraint Repair（复用模块，简短）

**English draft:**

[SYSTEM-NAME] trains with the multi-agent RL scheme of Teal: each demand is an
agent sharing the policy network, optimized by a COMA-style reward that
estimates each agent's marginal contribution to the global objective. Two
points deserve emphasis. First, the reward is computed against *ground-truth*
demands during offline training — sparse observation constrains the policy
input, never the training signal — which is exactly what a simulator or
historical trace provides. Second, we deliberately retain reward-driven RL
rather than imitation learning [FNC]: IL requires expert allocations solved
from *fully realized* demands of each episode, a supervision signal that is
unavailable when even post-hoc observation is sparse; RL needs only the
reward, which the network itself provides. At inference, allocations are
refined by [XX] parallelizable ADMM iterations to repair residual capacity
violations, adding <1 ms.

**中文批注：**
- "sparse observation constrains the policy input, never the training signal"
  ——训练用真值、输入用稀疏的双轨设计，一句话讲清，防审稿人误解为作弊。
- FNC 防守点在此落地（与 related_work.md §2 批注呼应）。
- ADMM 步数消融（0/2/5，B4 上 ±0.5%）可在实验章节一笔带过。

---

## 创新点与挑战/消融/代码映射总表

| 创新模块 | 回应挑战 | 消融开关 | 代码位置 |
|---|---|---|---|
| 4.2 可学习掩码嵌入 | C1 | `--mask-mode embed/zero/mean` | `teal_actor.mask_embedding` |
| 4.3 掩码感知门控 | C1 | `--no-gate` | `FlowGNN._gate_index_values` |
| 4.4 时序-空间融合 | C1 | `--hist-len 1/3/5` | `TemporalEncoder.py` + `teal_actor.forward` |
| 4.5 剪枝+零重训 | C2/C3 | `--prune-demands` `--demand-split` | `teal_env._build_graph` `FlowGNN.refresh_graph` |
| 4.6 MARL+ADMM | — | `--admm-steps` | 复用 Teal（`teal_model` `ADMM.py`） |

## 写作提醒

1. 4.2/4.3/4.4 是"三件套回应 C1"，行文可用统一句式（ambiguity → design →
   property → ablation hook），排比感强；
2. 4.6 务必谦逊标注"复用 Teal"，创新边界清晰是顶会审稿的加分项；
3. 公式符号表（D, M, θ, A, Ã, H_S/H_T/H_F, K, L）在 4.1 统一定义，后文不重复；
4. 每个模块末尾一句"ablation in §6.x"前向引用，把设计和实验缝起来。
