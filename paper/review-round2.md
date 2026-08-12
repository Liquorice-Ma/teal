# 模拟同行评审报告 · 第一轮（Full Review）

> **归档说明**：本轮为首次完整评审，审查对象是当时的稿件状态（10 页，编译无警告，0 未定义引用，0 `\todo`）。第二轮验证审查见本文件后半部分。

**稿件**：When Repair Absorbs Learning: A Mechanism Study of Learned Traffic Engineering under Sparse Traffic Observations
**目标期刊**：IEEE TNSE
**稿件类型**：实证机制研究（empirical mechanism study / negative result），非新系统 SOTA 提案
**审查模式**：`full`

---

## Phase 0 · 领域识别与审稿团配置

| # | 审稿人 | 身份设定 | 关注焦点 |
|---|--------|---------|---------|
| EIC | Journal-Fit Reviewer | TNSE 编委，network science + ML-for-systems | 期刊匹配度、negative result 的价值定位 |
| R1 | Methodology | 实验设计与统计推断专家 | 统计功效、因果识别、可复现性 |
| R2 | Domain | 卫星网络 / TE 领域专家 | 文献覆盖、baseline 完整性、场景泛化 |
| R3 | Perspective | 可微优化 / differentiable projection 方向 | 与投影层/feasibility layer 文献的关系、novelty 威胁 |
| DA | Devil's Advocate | 专攻核心论点证伪 | absorption 结论的替代解释 |

---

## Phase 1 · 五份独立审稿意见

### 🟦 R1 — Methodology（方法论）· 推荐：**Major Revision**

这是全文最该被严格审视的维度——一篇以"无差异"为核心发现的论文，统计论证必须无可挑剔。

**M1-1【MAJOR】用非显著性来论证"无贡献"，缺功效分析。**
RQ1 的结论 "training contributes no measurable improvement" 建立在 9/25、$p=0.115$ 上。这是典型的 **absence-of-evidence ≠ evidence-of-absence** 混淆。25 个配对样本的 sign test 功效很低——审稿人会立刻问：*你要多少样本才能检测出 5% 的 MLU 差异？* 当前结果无法区分"训练真的无效"与"样本太少检测不出"。**必须补 equivalence test（TOST）**，明确给出"trained 与 untrained 的差距落在 ±δ 内"的正面证据，否则 "absorption" 只是"未检出"。

**M1-2【MAJOR】跨 ρ 混合做 sign test 的独立性存疑。**
25 个配对 = 5 个 ρ × 5 个 seed，把 5 个不同实验条件的格 pooled 成一个 sign test。ρ 是强混杂变量，严格做法是**按 ρ 分层**报告再综合。当前 pooled p 值在统计上不够干净。

**M1-3【MAJOR】repair saturation 的机制解释是断言，非证据。**
"once repair operates near its fixed point, the input allocation matters little" 是全文核心机制，但 Tab 2 只证明了**步数不敏感**（1/2/5 步差异小），没有直接证明**输入不重要**。建议补一个直接测量：trained vs untrained 的 raw allocation 在 ADMM 迭代中的轨迹收敛距离——若两者确实被投影到近乎同一可行点，absorption 才有直接证据。

**M1-4【MINOR】saturation 表里 trained 一致更差，解释框架值得推敲。**
Tab 2 中 trained 的 median 在 1/2/5 步**全部高于** untrained（1.655/1.614/1.649 vs 1.597/1.568/1.599），MLU 越低越好。这是一致性信号，比 "9/25 无差异" 更接近"training 轻微有害"。作者用中性框架（absorption/neutrality）表述是合理的，但审稿人会追问为什么不讨论这个一致劣化的方向。

---

### 🟦 R2 — Domain（领域）· 推荐：**Minor Revision**

**M2-1【MAJOR】缺最接近工作 TEST 的直接复现对比。**
TEST 是唯一的端到端稀疏 TE 先前工作，related work 里对比了，但实验中 `nbr`（neighbor fill）只是弱替身，**没有真正复现 TEST 的 zero-fill + augmentation 方法作 baseline**。审稿人大概率会要求加一个 TEST-style baseline，否则"我们的 mask embedding 优于既有稀疏 TE 做法"缺乏直接支撑。

**M2-2【MAJOR】拓扑单一，"跨规模"≠"跨拓扑"。**
只有 Starlink Shell-1 家族（1584 + 528 两个规模）。RQ3 声称"机制不依赖规模"，但这是**同一架构的缩放**，不是独立拓扑泛化。建议至少补一个异构对照（OneWeb/Kuiper 参数，或一个 terrestrial WAN 如 B4）——哪怕只跑 absorption 的关键格。

**M2-3【MAJOR，必须修】cross-scale 表格 median 与文字结论表面矛盾。**
Tab 6 中 ρ=0.3 repair-off：trained **3.69 < untrained 3.83**（median 口径下 trained 更好，约 +3.7%），但正文写 "ρ=0.3 turns negative (−3.2%, 1/3 wins)"。读者从表格算出正增益，文字却说负——这是**表格展示 median-of-values、文字用 median-of-paired-differences** 两套口径造成的。**必须**在表注或正文明确说明两种口径，否则会被质疑数据处理不当。ρ=0.05 无此问题。

**M2-4【MINOR】只测了 MLU 一个目标。**
formulation 声称 objective-agnostic、throughput 作为 secondary instantiation，但实验零覆盖 throughput。要么补，要么把 "objective-agnostic" 的措辞降级。

---

### 🟦 R3 — Perspective（可微优化视角）· 推荐：**Major Revision**

**M3-1【MAJOR，novelty 威胁】遗漏 differentiable optimization / projection layer 文献。**
"凸投影层吸收上游学习信号"这个观察，与 **differentiable convex optimization（OptNet、cvxpylayers）和 projection-based feasibility layer** 文献强相关。这条线作者完全没有引用。风险有二：(a) 审稿人认为文献调研不完整；(b) 更糟——**可能已有工作在可微投影层上观察到类似 absorption**，威胁原创性。必须补这条线并明确区分。

**M3-2【MINOR】与 missing-data imputation 文献的连接可加强。**
mask embedding 本质是任务导向 imputation（end-to-end 对下游目标优化，而非重建误差），这点 §4.2 提到了，但可以更明确地定位到 imputation 文献，强化"为什么对 TE 用任务导向 imputation"。

---

### 🟦 EIC — Journal-Fit（期刊匹配）· 推荐：**Minor Revision**

**E1【MINOR】negative result 的价值定位需要前置。**
TNSE 读者需要被明确告知"为什么一篇不提供新 SOTA 的机制论文值得读"。当前 intro 做得不错，但建议在 abstract 首句就点明**对评估方法论的影响**（"end-to-end 数字可能认证一个权重毫无贡献的模型"），这是最能打动编委的钩子。

**E2【MINOR，投稿就绪】多处占位未清。**
- refs.bib 有 **11 处 `TODO: verify author list`**（test/vn2013/ncflow/pop/figret/netllm/jain2013/hong2018/wei2024/hu2022/flexsate2024），投稿前必须核实清除
- 架构图 `architecture-draft.png` 是 ImageGen 草稿占位，注释里写着"定稿后用 TikZ 矢量重绘"——**必须重绘**
- `figures/` 目录有一批 B 路线遗留孤儿图（exp-overall/ablation/runtime 等 10+ 张）未被引用，建议清理

---

### 🔴 DA — Devil's Advocate（魔鬼代言人）

**DA-1【CRITICAL】"absorption" 可能只是"没有可吸收的信号"——训练充分性未排除。**
核心论点是"repair 吸收了学习信号"。但一个更平凡的替代解释：**trained 模型本身就没学好**（欠训练 / 超参未调优），所以 repair 前后无差异。作者需要正面证明 trained 模型**确实学到了东西**——RQ2 部分做到了（无修复时 trained 有效），但应补：(a) 训练收敛曲线；(b) trained 模型在**有修复 + 高观测**下相对某个明确参照的优势。若不能排除欠训练，"absorption" 就退化为 "nothing to absorb"。
> **裁决**：部分成立。RQ2 的 13/15 已构成"模型确实能学"的强证据，故不 fatal，但**必须在正文显式回应**这个替代解释并附训练曲线。→ Major Revision 而非 Reject。

**DA-2【CRITICAL】机制证据是相关性，缺干预实验的连续谱。**
"repair 导致 absorption"目前只有相关性（trained≈untrained when repair on）+ 步数消融（1/2/5 都饱和）。缺一个**关键干预**：连续减弱 repair 强度（如松弛投影精度、减少约束严格度），观察 trained 优势是否**逐步恢复**。没有这条 "repair 强度↓ → learning 优势↑" 的单调证据，因果方向就不牢。
> **裁决**：成立，是全文论证链最薄弱环节。需补 repair 强度连续谱实验。→ Major。

**DA-3【MAJOR】Full-observation paradox 的解释反噬全文可信度。**
作者解释 paradox 时说 "sparse models win some seeds by luck"——承认 repair 把性能景观抹平到 seed noise 主导。但若如此，**RQ1、RQ3 的所有结论是否也受 seed luck 污染？** 这是内部一致性问题：不能用"noise 主导"解释对自己不利的格子，却用 "signal" 解释对自己有利的格子。

---

## Phase 2 · 编辑决策

### 决定：**Major Revision**

核心发现有真实价值——"修复层吸收学习信号"是对 learned TE 评估范式的有力警示，实验设计（配对 sign test、多 seed、跨规模）比多数同类工作严谨，写作清晰。但**统计论证（功效/equivalence）与因果论证（干预连续谱）两处是承重墙，必须补强后才能成立**。DA 的两个 CRITICAL 均裁决为"可修复"而非 "fatal"，故不 Reject。

### 修改路线图（按优先级）

| 优先级 | 任务 | 对应意见 |
|--------|------|---------|
| 🔴 P0 | 补 equivalence test / 功效分析，把"无贡献"从 non-significance 升级为正面等价证据 | M1-1 |
| 🔴 P0 | 修 cross-scale 表格口径矛盾（表注说明 median-of-values vs median-of-paired-gain） | M2-3 |
| 🔴 P0 | 显式回应"欠训练"替代解释 + 附训练收敛曲线 | DA-1 |
| 🟠 P1 | 补 repair 强度连续谱实验，建立因果单调关系 | DA-2 / M1-3 |
| 🟠 P1 | 补 differentiable optimization / projection layer 文献并区分 novelty | M3-1 |
| 🟠 P1 | 加 TEST-style baseline 直接对比 | M2-1 |
| 🟡 P2 | 按 ρ 分层报告 sign test | M1-2 |
| 🟡 P2 | 补一个异构拓扑对照（哪怕只跑关键格） | M2-2 |
| 🟢 P3 | 清 bib TODO、重绘架构图、清理孤儿图、定位 negative-result 价值 | E1/E2 |

### 投稿就绪度小结（第一轮时点）

- ✅ 引用完整性：36 个 cite 全部有 bib 条目，0 缺失 0 冗余
- ✅ 编译：0 warning，无 undefined reference
- ⚠️ 阻塞项：bib 11 处 TODO、架构 draft 图、cross-scale 口径矛盾

---
---

# 模拟同行评审报告 · 第二轮（Re-Review / 验证审查）

**稿件**：When Repair Absorbs Learning: A Mechanism Study of Learned Traffic Engineering under Sparse Traffic Observations
**目标期刊**：IEEE TNSE
**稿件类型**：实证机制研究（empirical mechanism study，含负结果）
**审查模式**：`re-review`（验证第一轮意见是否落实 + 新发现）
**审查日期**：2026-08-12
**审查对象**：当前工作树状态（12 页，编译 exit=0，0 未定义引用，**2 处 `\todo` 未填**）

---

## 0. 本轮结论速览

第一轮 15 项意见中，**7 项已落实且质量高于预期**，2 项进行中，**6 项未动**。同时本轮发现 **1 个新的阻断级问题**：作者在 §4.6 更正了修复层的技术本质，但摘要、引言、结论、关键词等 **12 处**仍保留旧的错误表述，造成全文自相矛盾。

**决定：Major Revision**（较上轮无升级，但阻断原因已从"统计论证不足"转移到"表述一致性与待填数据"）

值得强调：本轮新增的 §5.8 *Repair Without the Oracle* 是**实质性的科学升级**，把一个可能致命的方法论漏洞转化为论文最有价值的贡献。若能完成落实，稿件质量将显著超过第一轮水平。

---

## 1. R&R 追溯矩阵（第一轮意见 → 落实核验）

| # | 第一轮意见 | 严重度 | 作者改动 | 核验 | 说明 |
|---|---|---|---|---|---|
| M1-1 | 用非显著性论证"无贡献"，缺功效分析 | MAJOR | §5.2 新增配对差 95% CI `[−5.54%, +0.56%]`，措辞改为"可排除 >0.6% 的增益" | ✅ **已解决** | 从"未检出"升级为正面等价证据，处理得当。建议补一句样本量/可检测最小效应量以更严谨 |
| M1-2 | pooled sign test 混杂 ρ | MAJOR | 新增 `tab:rq1-strat` 按 ρ 分层（含胜数与配对增益） | ✅ **已解决** | — |
| M1-3 | 饱和机制是断言而非证据 | MAJOR | 重构为 0/1/2/5 连续谱表；删除"凸投影→不动点"解释，改为贪心排序→局部均衡 | ✅ **已解决（且超出要求）** | 原建议的 ADMM 轨迹测量已被更准确的机制解释取代，无需再做 |
| M1-4 | trained 一致更差未讨论 | MINOR | §5.2 新增段落，明确承认 ρ≥0.1 时 untrained 稳定领先（3/5、4/5、4/5） | ✅ **已解决** | 诚实且不过度辩解，处理得体 |
| M2-1 | 缺 TEST 直接 baseline | MAJOR | 已实现 `test-style` 配置（zero-fill + Transformer + 无门控 GCN），实验运行中 **7/24** | 🔄 **进行中** | 配置选择正确：作者查明现有 `zero-fill`(hist-len 1) 与 `no-embed`(保留门控) 均不等于 TEST，补的正是缺失格 |
| M2-2 | 拓扑单一（同架构缩放≠跨拓扑） | MAJOR | 无改动 | ❌ **未落实** | §5.6 仍只有 1584/528 两个同族实例 |
| M2-3 | cross-scale 表格口径矛盾 | MAJOR | §5.1 定义全局 gain 口径；`tab:cross-scale` 增配对增益/胜数列；正文显式说明 ρ=0.3 两口径反号 | ✅ **已解决（模范处理）** | 不仅修了该处，还把根因（双口径）提升为全文约定，并主动披露反号 |
| M2-4 | 只测 MLU，throughput 零覆盖 | MINOR | 无改动 | ❌ **未落实** | §3.2 仍声称 "objective-agnostic"。**注**：`final_matrix.csv` 中已存在 `obj=flow` 结果，属"有数据未报告"，成本极低 |
| M3-1 | 遗漏可微优化/投影层文献 | MAJOR | 无改动 | ❌ **未落实，且已恶化** | §6 结论 L31–32 主动提到 "projection mechanisms, feasibility layers in differentiable optimization" 却不引任何文献 —— 承认该领域存在却不引用，比原先只是遗漏更易招致质疑 |
| M3-2 | imputation 文献连接可加强 | MINOR | 无改动 | ❌ **未落实** | — |
| E1 | negative result 价值定位需前置 | MINOR | 无改动 | ❌ **未落实** | 摘要首句仍是背景铺陈 |
| E2 | bib 11 处 TODO、架构图为 draft、孤儿图 | MINOR | 无改动 | ❌ **未落实** | 投稿阻断项 |
| DA-1 | 未排除"欠训练"替代解释 | CRITICAL | 无改动 | ❌ **未落实** | 仍缺训练收敛曲线与显式反驳段落。**注**：§5.8 的 oracle 发现间接削弱了该质疑（吸收有了独立的信息学解释），但不能替代直接证据 |
| DA-2 | 因果链缺干预连续谱 | CRITICAL | 0/1/2/5 步合并呈现，叙事改为"repair 是开关不是旋钮" | ✅ **已解决（且超出要求）** | 更进一步：§5.8 通过限制 repair 的信息输入，提供了第二条独立干预维度 |
| DA-3 | 噪声/信号解释双标 | MAJOR | §5.5 新增一致性辩护段（RQ1 是零假设、RQ3 定义在 seed 集合之上） | ✅ **已解决** | 论证锐利，可直接用于 rebuttal |

**统计**：已解决 7 / 进行中 1 / 未落实 7

---

## 2. 本轮新发现

### 🔴 N-1【阻断级】修复层术语在全文自相矛盾（12 处）

作者在 §4.6 正确更正了核心事实：

> "The repair step is therefore **not a projection** but a *conservation-preserving rebalancing*"

但以下 12 处仍保留被推翻的旧表述，与 §4.6 直接冲突：

| 位置 | 现有表述 |
|---|---|
| `main.tex:58` | "a **convex** repair layer that **projects** allocations back onto feasible flows" |
| `main.tex:68` | "a single **ADMM step** collapses the training gain" |
| `main.tex:89`（关键词） | "**convex repair**" |
| `01-intro:19-20` | "a **convex** procedure, e.g., **ADMM**, that **projects** ... onto the feasible flow **polytope**" |
| `01-intro:50` | "neural allocator with **convex repair**" |
| `01-intro:71` | "an **ADMM repair layer**" |
| `01-intro:78` | "a single **ADMM step**" |
| `01-intro:120` | "with a **convex repair layer**" |
| `04-design:18` | "refined by **ADMM**" |
| `04-design:54`（图注） | "split ratios refined by **ADMM**" |
| `05-eval:409` | "the **two-step ADMM** repair" |
| `06-conclusion:6-7` | "when a **convex** repair layer is present, it saturates within a single **ADMM iteration**" |

**审稿人视角**：任何通读者都会发现 §4.6 与摘要互相打脸。这不是笔误，而是"论文最核心机制的名称在自身内部不统一"，会直接引发对全文可靠性的怀疑。**必须在投稿前统一**，且此项修改不依赖任何新实验。

### 🔴 N-2【阻断级】§5.9 Takeaways 与 §6 结论未反映 oracle 发现，给出无法执行的建议

§5.9 Deployment 条与 §6 均写：

> "The real deployment decision is whether to pay for repair at all"

但 §5.8 已确立：oracle repair **在稀疏观测下不可部署**（它需要恰好缺失的那份数据）。因此"要不要为 repair 付费"是个伪选择——它不是价格问题而是可得性问题。同理，Takeaways 里 "$44\%$ lower MLU" 的归因也需按 §5.8 拆分。

### 🟠 N-3【MAJOR】§5.2 的 "44% 绝对改善" 归因与 §5.8 冲突

§5.2 称 repair 带来 44% 绝对改善（2.83→1.60）并据此写入 Takeaways。但 §5.8 的阶梯表明：

| repair 可见矩阵 | 中位 MLU | 相对无修复改善 |
|---|---|---|
| 无 | 2.833 | — |
| `obs-nbr` | 1.740 | 38.6% |
| `oracle` | 1.599 | 43.6% |

即 44% 中约 5 个百分点直接来自真值矩阵。建议在 §5.2 首次给出 44% 时即加前向指针（"部分归因于其信息优势，见 §5.8"），否则读者会觉得 §5.8 在推翻前文而非补全前文。

### 🟠 N-4【MAJOR】§5.7 推理延迟数字缺可复现依据，且与实测差约 7 倍

§5.7 称 "completes in under $0.4$~s per snapshot"。但在本轮冒烟测试中，同拓扑单快照实测 `runtime≈0.056 s`。两者差 7 倍。可能解释：0.4s 含图重建等一次性开销，或来自不同配置。问题在于**该数字没有留下任何实验记录**，本地与服务器 CSV 均无延迟数据，无法复算。

同段还称 "The repair layer itself accounts for a negligible share of this budget"——**repair 单独开销从未测量**，属无证据断言。

### 🟡 N-5【MINOR】§5.1 "varies four factors" 与实际列举数不符

新增 repair-input 因子后，列举变为 ρ / repair budget / repair 可见矩阵 / training status / seeds 共 5 项，但仍写 "four factors"。

### 🟡 N-6【MINOR】用语未随术语更正同步

§5.2 L109 "the default **two-step** repair"、§5.7 "two-step" 应与新定义的 "sweeps"（1 单位 = 10 轮）一致，否则读者会把"步"理解为 ADMM 迭代。

---

## 3. 五位审稿人卡片（本轮增量意见）

### R1 · Methodology — 推荐：Minor Revision（上轮 Major）

统计论证已达到可发表水平：等价区间、分层报告、双口径约定、噪声一致性辩护四项到位，处理质量高于我第一轮的要求。剩余关切：
- **DA-1 仍未回应**：需要训练收敛曲线证明模型确实学到了东西。§5.8 的信息学解释虽然强，但"repair 有信息优势"与"模型欠训练"并不互斥，仍需独立排除。
- §5.8 的 `tab:ladder` 目前只有单一 ρ、单一训练状态（untrained/ρ=0.3）。作为"信息优势"这一核心新论点的唯一支撑，**至少需要两个 ρ 和多 seed 的一致性**，否则又回到"单格外推"的老问题。
- 建议为 CI 补充可检测最小效应量（MDE），使"排除 >0.6%"更具说服力。

### R2 · Domain — 推荐：Major Revision

- M2-1 方向完全正确。作者查明现有配置均不等价于 TEST 并补齐缺失格，这比我原本的要求更严谨。等待 24/24 完成。
- **M2-2 仍是硬伤**：§5.6 标题为 "Cross-Scale Replication"，但两个实例同属 Shell-1 家族。建议或补一个异构拓扑，或把标题与声明降级为 "Cross-Scale (same family)"，明确范围。
- M2-4 属"数据已有、报告缺失"，成本最低的一项，建议本轮清掉。
- §5.8 显著提升了本文对卫星 TE 社区的价值：它把"稀疏观测下修复层需要什么信息"这一从未被隔离的问题摆上台面，这比原先的吸收结论更贴近实际部署关切。

### R3 · Perspective（可微优化视角）— 推荐：Major Revision

- **M3-1 未落实且风险上升**。§6 主动提及 feasibility layer / differentiable optimization 却零引用。既然 §4.6 现已明确本文修复层**不是**凸投影，作者更有必要把这条线引全并说明区别：本文的发现是关于"启发式修复 + 信息不对称"，而非关于凸投影层——这个界定既保护了 novelty，也避免过度声明。
- 建议同时讨论：若换成真正的凸投影层（且只喂观测矩阵），吸收是否仍成立？这是一个有力的 future work，也能预先回应"你测的不是 ADMM"的质疑。

### EIC · Journal-Fit — 推荐：Minor Revision

- 12 页落在 TNSE 常见区间（11–14），页数健康。
- E1/E2 仍未动。E2 为硬性投稿阻断项：bib 11 处 `TODO: verify`、架构图为 ImageGen 草稿、`figures/` 存在 10+ 张未引用孤儿图。
- **新增建议**：§5.8 的发现足以支撑摘要与标题的重新定位。当前标题 *When Repair Absorbs Learning* 仍然成立且贴切，但摘要应当把"修复层隐含依赖真值流量矩阵"提到前部——这是本文最具警示价值、也最可能被引用的一句。

### DA · Devil's Advocate

- **DA-2 已解决，且作者给了我一个更强的版本**：不只沿修复强度维度做了干预，还沿"修复可见信息"维度做了第二条独立干预。因果论证现已站得住。
- **DA-3 已解决**。
- **DA-1 仍开放**（见 R1）。
- **新质疑 DA-4**：§5.8 若最终显示"可部署修复下训练重新有效"，则 §5.2–5.4 的全部结论都变成"关于一个不可部署配置的结论"。作者必须正面回答：**RQ1/RQ3 还有什么独立价值？** 我的建议是明确定位为"对既有文献评估协议的诊断"（既有系统确实是这么跑的，所以这个诊断有价值），而不是"对可部署系统的结论"。若不显式划清，读者会认为半篇论文在测一个不存在的系统。
  > **裁决**：非致命，但必须在正文显式回应，否则构成 Accept 阻断。

---

## 4. 编辑决定与修订路线图

### 决定：**Major Revision**

统计与因果两处承重墙已修好，且 §5.8 把一个潜在致命缺陷转成了核心贡献——这是本轮最值得肯定之处。阻断项现在集中在**一致性**与**待填数据**，两者都无科学不确定性，属可执行工作。

### 路线图（按优先级）

| 优先级 | 任务 | 依赖 | 对应 |
|---|---|---|---|
| 🔴 P0 | 统一 12 处修复层术语（摘要/引言/§4.1/§5.7/§5.9/结论/关键词） | 无 | N-1 |
| 🔴 P0 | 重写 Takeaways + 结论，反映 oracle 不可部署 | 无 | N-2 |
| 🔴 P0 | 填 §5.8 两处 `\todo` | deployable 批 | — |
| 🔴 P0 | §5.8 阶梯表扩到 ≥2 个 ρ、多 seed | deployable 批 | R1 |
| 🟠 P1 | 显式回应 DA-1（收敛曲线 + 反驳段） | 训练日志 | DA-1 |
| 🟠 P1 | 显式回应 DA-4（RQ1/RQ3 的独立定位） | 无 | DA-4 |
| 🟠 P1 | 补可微优化/投影层文献并划清边界 | 查文献 | M3-1 |
| 🟠 P1 | 写入 TEST baseline 结果 | B1 批 | M2-1 |
| 🟠 P1 | §5.2 加 44% 归因前向指针 | 无 | N-3 |
| 🟡 P2 | 重测并记录推理延迟（含 repair 单独开销） | 需实测 | N-4 |
| 🟡 P2 | 报告已有 throughput 结果，或降级 objective-agnostic 声明 | 无 | M2-4 |
| 🟡 P2 | §5.6 标题/声明范围降级，或补异构拓扑 | 可选实验 | M2-2 |
| 🟢 P3 | 清 bib 11 处 TODO、重绘架构图、清理孤儿图、摘要前置钩子 | 无 | E1/E2 |
| 🟢 P3 | "four factors" 计数、"two-step"→"two-sweep" | 无 | N-5/N-6 |

### 投稿就绪度

| 项 | 状态 |
|---|---|
| 编译 | ✅ exit=0，12 页 |
| 未定义引用 | ✅ 0 |
| 引用完整性 | ✅ 36 个 cite 全部有 bib 条目，0 缺失 0 冗余 |
| 残留 `\todo` | ⚠️ 2 处（等数据） |
| 术语一致性 | ❌ 12 处冲突 |
| bib 条目核实 | ❌ 11 处 `TODO: verify` |
| 图件 | ❌ 架构图为草稿 PNG |

---

## 附录 C：第二轮意见的落实记录（作者回应）

> 本节记录第二轮报告发出后的即时修订，供第三轮核验。

| # | 意见 | 落实 | 核验方式 |
|---|---|---|---|
| N-1 | 修复层术语 12 处自相矛盾 | ✅ **已解决** | 全文 grep 复核：`ADMM` 剩 3 处，均为**合法用法**（§1 描述领域普遍做法「convex procedure such as ADMM, or a lightweight rebalancing heuristic」；§3.3 与 §4.6 指吞吐目标下真实使用的 ADMM+rounding）。`convex repair`/`polytope`/`projects onto` 已清零；关键词改为 constraint repair |
| N-2 | Takeaways/结论给出不可执行建议 | ✅ **已解决** | §5.9 与 §6 均改为「操作性选择不是要不要为 repair 付费，而是把哪个矩阵交给它」；摘要新增 oracle 依赖披露；引言 deployment 段同步 |
| N-3 | 44% 归因与 §5.8 冲突 | ✅ **已解决** | §5.2 首次给出 44% 处加前向指针，指明部分来自 repair 读到的信息而非再平衡本身 |
| N-4 | 延迟数字与开销断言无证据 | 🔶 **部分** | 已删除未测量的「negligible share of this budget」，改为代码结构可支撑的表述（两轮 = 20 次稀疏 scatter）。**`under 0.4 s` 仍待实测**：注意 `info['runtime']` 仅覆盖 transform+repair+extract，不含 GNN/Transformer 前向，故 0.056 s 与 0.4 s 量的不是同一段，不可直接替换 |
| N-5 | 「four factors」计数错误 | ✅ **已解决** | 改为 five factors |
| N-6 | two-step / sweeps 用语不一致 | ✅ **已解决** | 全文 `two-step` 清零 |
| DA-4 | RQ1/RQ3 的独立定位未交代 | ✅ **已解决** | §5.9 新增 **Scope** 条、§6 新增独立段落，明确定位为「对既有评估协议的诊断」而非「对可部署系统的预测」 |
| M3-1 | 可微优化文献缺失 | 🔶 **部分** | §6 future work 已把该方向重构为可检验问题（换成真正的可微投影后吸收是否仍成立），措辞不再是「承认存在却不引用」；**bib 条目仍需补** |
| E1 | negative result 价值前置 | 🔶 **部分** | 摘要末段新增最具警示性的发现（repair 隐含读真值），但首句仍为背景铺陈 |

**编译核验**：12 页，exit=0，0 未定义引用，0 严重 Overfull，残留 `\todo` 2 处（均等 `deployable_repair.csv` 数据）。

---

## 附录 A：实验批次状态（审查时点）

| 批次 | 用途 | 进度 | 失败 |
|---|---|---|---|
| `test_baseline` | TEST-style baseline（M2-1）+ 跨架构吸收验证 | 7/24 | 0 |
| `deployable_repair` | repair 只读观测矩阵（§5.8 核心） | 3/36 | 0 |

已确立的可复算事实：
- 回归校验：`--repair-input oracle` 精确复现 1.4766，与 `verify.csv` 逐位一致
- 信息阶梯（untrained, ρ=0.3, 3 seeds 中位）：无修复 2.833 → `obs-nbr` 1.740 → `oracle` 1.599
- RQ2 全 15 格本地复核通过：13/15，五个 ρ 的配对中位与正文逐一吻合

## 附录 B：本轮未变更但仍成立的第一轮结论

- 核心科学价值成立：修复层与学习组件的功能重叠是对 learned TE 评估范式的有力警示
- 实验设计严谨度（配对检验、多 seed、跨规模）高于同类工作平均水平
- 写作清晰，RQ 组织得当

---

*本报告由 `academic-paper-reviewer` v1.10.0 框架生成（5 审稿人 + 编辑综合，re-review 模式）。所有事实性论断均基于对当前工作树的直接核查（编译日志、引用比对、CSV 复算、代码溯源），未采用推断替代验证。*
