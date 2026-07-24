# Model & Problem Formulation（第 3 章）草稿（含写作脉络备忘）

> 用途：论文第 3 章底稿，位于 Related Work 之后、Design 之前。英文段落可直接
> 改写进论文；中文批注写作时删除。公式与代码实现严格对应（文末映射表）。
> 本章就位后，design.md 的 4.1 应瘦身为一段 pipeline 总览 + 前向引用本章。

---

## 章节骨架

```
§3 Model and Problem Formulation
  3.1 Network and Traffic Model        — 图、容量、时隙、需求、候选路径
  3.2 TE with Complete Observation     — 经典 path-based TE 优化问题（LP）
  3.3 TE with Sparse Observation       — 观测模型 + 我们的问题定义（核心）
  3.4 Solution Paradigm                — 为什么端到端学习；优化目标的 RL 形式
  （符号表放 3.1 末尾或单独 Table）
```

---

## 3.1 Network and Traffic Model

**English draft:**

We model the constellation at control epoch t as a directed graph
G^t = (V, E^t), where V is the set of N satellites and E^t the set of
inter-satellite links (ISLs), each with capacity c_e. Time is slotted into
control epochs; within an epoch the topology snapshot is fixed, while across
epochs both E^t and the traffic change as satellites move.

Traffic is described by a demand matrix D^t ∈ R^{N×N}, where d_i denotes the
volume of the i-th source–destination pair. Since satellite traffic is
geographically concentrated, we maintain an *active demand set* K^t — the
pairs with nonzero demand in recent history (§4.5) — and only these are
modeled. Each demand i ∈ K^t is served by K pre-selected candidate paths
P_i = {p_{i,1}, ..., p_{i,K}} (K = 4 edge-disjoint shortest paths in our
implementation). The TE decision for demand i is a split-ratio vector
r_i ∈ R^K with r_{i,k} ≥ 0 and Σ_k r_{i,k} ≤ 1, allocating flow
f_{i,k} = r_{i,k} · d_i to path p_{i,k}; the residual 1 − Σ_k r_{i,k}
represents intentionally unserved traffic.

**中文批注：**
- "残差 = 主动不服务的流量"这个设定来自 Teal 的 softmax 形式（softmax 分母
  +1 那一项），对应 `transform_raw_action` 里 `raw_action/(1+sum)`——写清楚，
  否则审稿人会问 Σr<1 去哪了。
- K=4 边不相交最短路对应 PATH_FORM_HYPERPARAMS (4, True, 'min-hop')。
- 符号表建议：G, V, E, c_e, N, D, d_i, K^t(集合)/K(路径数——**符号冲突，
  写作时改：路径数用 P 或 κ**), P_i, r_i, f_ik, M, O, L, θ, π。

## 3.2 TE with Complete Observation（经典问题）

**English draft:**

With the complete demand matrix available, TE is the classic path-based
allocation problem. Using the total satisfied demand as the objective
(total-flow TE [Teal, NCFlow]):

    maximize    Σ_{i∈K^t} Σ_{k=1}^{K} f_{i,k}                    (1)
    subject to  Σ_{k} f_{i,k} ≤ d_i                ∀ i ∈ K^t     (2)
                Σ_{(i,k): e ∈ p_{i,k}} f_{i,k} ≤ c_e   ∀ e ∈ E^t (3)
                f_{i,k} ≥ 0                                       (4)

Constraint (2) caps allocation at the actual demand, and (3) enforces link
capacities. Problem (1)–(4) is a linear program solvable to optimality, but
at constellation scale its solve time far exceeds the control epoch [SaTE],
and — more fundamentally for this paper — both the objective through f = r·d
and constraint (2) require knowing d_i for *every* active demand.

**中文批注：**
- 最后一句是本章的"转折钩子"：LP 不仅慢，而且**结构上依赖完整 D**——这句
  把稀疏观测问题从"工程困难"提升为"问题本身变了"，是 3.3 的引子。
- (1)-(4) 与代码对应：obj=total_flow（`get_obj` 里 action.sum），(2) 由
  softmax 形式天然满足，(3) 由 ADMM+rounding 修复。
- min-max link utilization 变体可加一句 "an alternative objective"（代码支持
  `--obj min_max_link_util`），不展开。

## 3.3 TE with Sparse Observation（本文问题，核心小节）

**English draft:**

**Observation model.** At each epoch only a subset of demands is measured.
A binary mask M ∈ {0,1}^{|K^t|} marks demand i as measured (M_i = 1) or not
(M_i = 0), and the controller receives the sparse observation

    O^t = D^t ⊙ M,   together with M itself,                      (5)

where ⊙ is element-wise product. We consider two measurement granularities:
*flow-level*, where a fraction ρ of demand pairs is sampled, and *node-level*,
where a fraction ρ of satellites meter all their outgoing demands — the
latter matching deployments where measurement capability resides on a subset
of satellites. The controller additionally retains the last L sparse
observations H^t = (O^{t−L+1}, ..., O^t).

**Problem statement.** Sparse-observation TE seeks a policy π that maps the
observable state to split ratios for *all* active demands,

    {r_i}_{i∈K^t} = π(H^t, M, G^t),                               (6)

maximizing the *ground-truth* objective: allocated flow is min(r_i·d_i, d_i)
counted against the true demand d_i and the capacity constraints (3), even
though d_i is unknown to π for unmeasured i. Formally,

    maximize_π  E_t [ Σ_i Σ_k f_{i,k}(π) ]                        (7)
    subject to  (2)–(4) with the true D^t,

where the expectation is over traffic and topology dynamics. The problem is a
partially observable sequential decision problem: unlike (1)–(4), the decision
variables and the evaluation live in different information sets — the policy
sees O and M, while feasibility and performance are dictated by the hidden D.

**中文批注：**
- 核心句："decision variables and evaluation live in different information
  sets"——这是本文问题与经典 TE 的数学分界线，必须醒目。
- (5)(6)(7) 与实现对应：O/M/H = `env.get_obs()` 的 tm_seq+mask；真值评估 =
  `env.step()` 用完整 TM 算 reward/ADMM/round（双轨设计）。
- flow/node 两种观测粒度对应 `--obs-type`；ρ 对应 `--obs-ratio`。
- 这里可以放第二张图：同一 TM 的完整/稀疏对照热力图（黑块=未观测），直观。

## 3.4 Solution Paradigm（求解方式）

**English draft:**

Three solution families exist for (7). *Solve-with-zeros* plugs O directly
into the LP (1)–(4), implicitly assuming unmeasured demands are zero; it is
fast to formulate but structurally starves unmeasured traffic. *Two-stage*
approaches first estimate D̂ from (H, M) — via interpolation, tensor
completion, or compressive sensing — then solve the LP on D̂; estimation is
optimized for reconstruction error, which is misaligned with allocation
quality, and errors propagate un-damped into the allocation. We instead adopt
*end-to-end learning*: parameterize π_θ as a neural network and optimize θ
directly against the TE objective,

    max_θ  E [ R(π_θ(H, M, G), D) ],                              (8)

where R is the total satisfied demand computed on the true D. Since R is
non-differentiable through the capacity-constrained network and the true D is
available only as a training-time signal, we optimize (8) with multi-agent
reinforcement learning — each demand acts as an agent sharing θ, receiving a
COMA-style marginal-contribution reward (§4.6) — followed by a lightweight
ADMM refinement at inference to repair residual violations of (3). This
paradigm makes the observation model (5) a property of the *input*, and the
optimization target (7) a property of the *training signal*, cleanly
separating what the policy can see from what it is optimized for.

**中文批注：**
- 三分法（solve-with-zeros / two-stage / end-to-end）与实验 baseline 一一
  对应：zero-fill、MI/补全类、ours——第 6 章对比表直接按这个分类组织。
- 收尾句再次强调双轨分离（输入 vs 训练信号），与 4.6 呼应，三处一致。
- (8) 就是 RL 目标，reward R 的具体 COMA 展开留给 4.6，本章不写 RL 细节。
- ADMM 在这里只提"轻量修复(3)"，公式留给附录或引用 Teal。

---

## 公式与代码映射表

| 公式 | 代码位置 |
|---|---|
| r 的 softmax 参数化（Σr≤1） | `teal_env.transform_raw_action` |
| (1) 目标 total flow | `teal_env.get_obj` (obj='total_flow') |
| (2) demand 约束 | softmax 形式天然满足 + `round_action(round_demand)` |
| (3) 容量约束修复 | `ADMM.tune_action` + `round_action(round_capacity)` |
| (5) 稀疏观测 O, M | `teal_env._read_hist_tms` / `_init_obs_mask` |
| (6) 策略 π | `teal_actor.evaluate`（输入 obs dict） |
| (7) 真值评估 | `teal_env.step`（reward 用完整 TM） |
| (8) RL 目标 | `teal_model.train` + `teal_env.take_action`（COMA） |
| flow/node 观测粒度 | `--obs-type`，`_init_obs_mask` 两分支 |

## 写作提醒

1. **符号冲突要修**：候选路径数 K 与 active demand 集合 K^t 撞了，建议路径数
   改用 κ 或直接写 "4 candidate paths"；
2. 章节间分工：本章只定义问题和范式，**不出现任何神经网络结构**（掩码嵌入/
   门控/Transformer 全部留给 §4），审稿人读完本章应该能自己判断"这问题为什么
   非平凡"；
3. design.md 的 4.1 相应瘦身：删去 formulation 内容，改为一段 pipeline 总览
   + "following the formulation in §3"；
4. (7) 的 POMDP 性质如果想加理论味，可以一句话点到 "a POMDP whose observation
   kernel is the masking operator" 即止，不必展开。
