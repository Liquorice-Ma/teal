# Defence Technology 投稿前问题与修改建议

审稿对象：`interactcadsample.tex`  
目标期刊：Defence Technology  
审稿方式：使用 `academic-paper-reviewer` 科研论文审稿 skill，按投稿前模拟审稿标准检查。

## 一、总体结论

当前稿件的研究方向与 Defence Technology 基本匹配。论文聚焦红外与可见光图像融合，应用场景涉及卫星、无人机、夜间低照度、车辆和船只等目标感知任务，具备防务技术应用潜力。

但是，当前版本不建议直接投稿。主要原因不是选刊方向错误，而是投稿材料完整性、双盲格式、实验可复现性、指标定义、创新性论证和英文表达仍存在明显问题。若直接提交，存在技术检查退回、编辑初筛质疑或外审 Major Revision 的风险。

建议结论：投稿前中等强度修改。优先修复格式和声明类硬伤，再增强实验说明和 Defence Technology 适配叙述。

## 二、必须修改的问题

### 1. 匿名稿设置必须调整

Defence Technology 采用 double anonymized review。当前 LaTeX 文件中设置为：

```latex
\anonymousfalse
```

这会在正文中显示作者姓名、单位和通信邮箱。投稿匿名正文时应改为：

```latex
\anonymoustrue
```

同时应单独准备 title page 文件，包含作者、单位、通信作者邮箱、致谢、利益冲突声明等信息。

建议修改：

- 匿名正文中不得出现作者姓名、单位、邮箱。
- 匿名正文中不得出现 acknowledgements。
- title page 作为单独文件提交。

### 2. Acknowledgments 部分未完成

当前正文中存在未完成句子：

```latex
This work was supported by
```

这是投稿硬伤。若直接提交，容易被技术检查退回。

建议：

- 若有基金，完整写明基金名称和项目编号。
- 若无基金，按期刊要求写：

```text
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.
```

但注意 Defence Technology 双盲要求下，Acknowledgments 应放在 title page，而不是匿名正文。

### 3. Funding 和 Declaration of competing interests 需要补齐

当前 competing interest 和 data availability 相关内容被注释，没有形成正式声明。Defence Technology 投稿系统通常需要这些信息。

建议准备以下内容：

```text
Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
```

若有潜在利益冲突，则必须如实披露。

Funding statement 也应单独准备。若无基金，使用无基金声明；若有基金，列明基金来源和编号。

### 4. PSNR、MSE、VIF 的参考对象必须说明

红外-可见光图像融合通常没有真实 fused ground truth。当前稿件使用 PSNR、MSE 和 VIF，但没有明确这些 reference-based metrics 是相对于哪个参考图像计算的。

当前源码中还保留了注释：

```latex
%Because no ground-truth fused image is available, the reference-based metrics...
```

这说明指标定义尚未完全确认。审稿人很可能质疑：

- PSNR 是相对 visible image、infrared image、二者平均图，还是某种伪真值？
- MSE 的 reference 是什么？
- VIF 对 multimodal fusion 是否合理？
- 60--70 dB 的 PSNR 数值是否可信？

建议补充一段清晰说明：

```text
Because no ground-truth fused image is available for infrared-visible image fusion, reference-based metrics were computed with respect to [明确参考对象]. The same evaluation protocol was applied to all compared methods.
```

如果实际评估代码中 reference 不明确，必须先核查代码和实验记录，不能凭文本猜测。

### 5. Defence Technology 应用关联需要强化

当前论文更像通用 image fusion 论文，虽然提到 satellite、UAV、vehicles、ships，但 Defence Technology 的目标读者会更关心防务任务价值。

建议在摘要、Introduction、实验分析和 Conclusion 中增加以下表述方向：

- all-weather and day-night reconnaissance；
- UAV surveillance；
- battlefield situational awareness；
- maritime target observation；
- vehicle and ship target visibility；
- low-light and thermal target preservation；
- defence-oriented remote sensing interpretation。

注意不要空泛堆词，应结合实验图中的车辆、港口、河流、船只、夜间低照度场景具体说明。

### 6. 方法创新表述需要收敛或补证据

稿件多处声称 MARM 可以显式解耦 infrared thermal saliency 和 visible spatial texture。但方法公式显示：先将 IR 和 VIS concatenate，再对共享特征做 SAB 和 CAB。

这会带来审稿疑问：

- SAB 是否真的只增强 visible texture？
- CAB 是否真的只增强 infrared thermal saliency？
- 这是否只是普通 spatial/channel attention，而不是 modality-specific decoupling？

建议二选一：

方案 A：收敛表述。将 “explicitly decouples modality-specific contributions” 改为更稳妥的 “separates spatial and channel attribute enhancement after multimodal feature embedding”。

方案 B：补充证据。增加 attention visualization、feature response maps 或 modality perturbation analysis，证明 SAB/CAB 对不同模态属性确实有不同响应。

### 7. Related Work 中存在明显拼写和语义错误

当前句子：

```text
MTFusion under semantic segmenby adaption totation guidance.
```

明显是拼写残留，应立即修正。建议改为：

```text
Zhou et al. proposed MTFusion, which combines Mamba and Transformer blocks under semantic-segmentation-guided supervision.
```

同时建议全文检查类似语法问题。

## 三、建议增强的问题

### 1. 增加模型复杂度和效率分析

Defence Technology 读者会关心实际部署，尤其是 airborne、spaceborne 或 on-board 场景。当前稿件只声称 Swin Transformer 具有线性复杂度，但没有给出实际成本。

建议补充：

- Parameters；
- FLOPs；
- inference time；
- GPU/CPU 测试环境；
- image size；
- 与 AUIF、FusionGAN、SwinFusion 的运行时间对比。

如果结果显示模型较重，也可以诚实说明未来会做 compression，但最好至少证明当前版本在合理硬件上可运行。

### 2. 增加更多强 baseline

当前对比方法包括 LatLRR、AUIF、DenseFuse、FusionGAN、SwinFusion，但近年强方法在实验中缺席较多。Related Work 中引用了很多新方法，但实验没有比较，会削弱说服力。

建议至少增加 2--4 个方法，例如：

- U2Fusion；
- RFN-Nest；
- CDDFuse；
- DIVFusion；
- DATFuse；
- S4Fusion；
- CrossFuse。

如果某些方法无法公平适配，需要给出更具体理由，而不是简单说 released implementations could not be fairly adapted。

### 3. 明确训练/测试划分，避免 patch leakage 质疑

当前每个数据集生成 4000 对图像，按 9:1 分训练和测试。若 patch 是从同一原始大图随机裁剪，再随机划分，可能导致训练集和测试集来自同一场景甚至空间相邻区域。

建议补充说明：

- 是否按原始 image/scene 级别划分；
- 是否保证测试场景不出现在训练集中；
- patch 之间是否存在重叠；
- 随机种子是否固定；
- 是否有独立测试区域。

推荐表述：

```text
To avoid spatial leakage, the train/test split was performed at the scene level before patch extraction. Patches cropped from the same original scene were assigned exclusively to either the training or test set.
```

仅当实际实验确实如此时才能这样写。

### 4. 补充 MARM 内部消融

当前模块消融只有：

- w/o STM；
- w/o MARM；
- Proposed。

这不足以证明 MARM 中各个子模块的必要性。建议增加：

- w/o CA；
- w/o SAB；
- w/o CAB；
- w/o RB；
- serial spatial-channel attention；
- parallel spatial-channel attention；
- direct STM without decoupled attention。

这样可以更有力支撑 “decoupled attention” 的核心贡献。

### 5. Loss ablation 建议增加量化表

当前 loss ablation 主要是视觉对比图和文字描述，没有量化表。建议补一张表，列出以下配置的 SF、AG、PSNR、MSE、VIF：

- pixel intensity only；
- gradient only；
- structural only；
- pixel + gradient；
- pixel + structural；
- gradient + structural；
- full loss。

这会显著提高说服力。

### 6. 增加统计稳定性说明

当前表格只给均值，没有标准差、置信区间或显著性检验。建议至少增加：

- mean ± std；
- 或说明结果是 400 test pairs 的平均值；
- 对关键指标做 paired t-test / Wilcoxon test。

如果篇幅不够，至少在实验设置中说明所有指标是对全部测试样本平均得到。

## 四、语言和排版问题

### 1. 标题双空格

当前标题：

```text
Decoupled  Attention with Global Coupling...
```

`Decoupled` 和 `Attention` 之间有两个空格。应改为：

```text
Decoupled Attention with Global Coupling for Infrared and Visible Image Fusion
```

### 2. “Related Works” 建议改为 “Related Work”

学术论文中章节标题通常使用单数：

```latex
\section{Related Work}
```

### 3. Figure caption 标点统一

例如 Figure 8 caption 缺句号：

```latex
\caption{Qualitative  results of four methods on infrared–visible aerial dataset}
```

建议改为：

```latex
\caption{Qualitative results of four methods on the infrared--visible aerial dataset.}
```

### 4. 英文语法问题示例

原句：

```text
Figure presents qualitative the fusion results...
```

建议：

```text
Figure presents the qualitative fusion results...
```

原句：

```text
The shallow features ... is formulated as
```

建议：

```text
The shallow feature representation ... is formulated as
```

或：

```text
The shallow features ... are formulated as
```

### 5. Landsat 8 发射时间需要核查

文中写 Landsat 8 launched in November 2013。通常 Landsat 8 发射时间为 2013 年 2 月，正式投入运行在 2013 年。建议核查并修正，避免事实错误。

## 五、投稿前建议准备的文件

Defence Technology 投稿前建议至少准备：

1. Anonymized manuscript：匿名正文 PDF/LaTeX。
2. Title page：标题、作者、单位、通信作者、邮箱、致谢、基金、声明。
3. Declaration of competing interest 文件。
4. Funding statement。
5. Highlights：通常 3--5 条。
6. Graphical abstract：若期刊系统要求或推荐。
7. Cover letter。
8. Source files：LaTeX、图片、参考文献。
9. Supplementary material：如更多视觉结果、复杂度分析、额外消融实验。

## 六、推荐的最小投稿前修改路线

如果时间很紧，建议按以下顺序处理：

1. 改匿名设置，生成匿名稿。
2. 移除匿名正文中的 Acknowledgments 和作者相关信息。
3. 单独准备 title page。
4. 补 Funding 和 competing interest 声明。
5. 修正 PSNR/MSE/VIF 的 reference 定义。
6. 修复明显 typo 和语法错误。
7. 在摘要、Introduction 和 Conclusion 中强化 defence application framing。
8. 补充模型复杂度表。
9. 如果还有时间，增加强 baseline 和 MARM 内部消融。

## 七、最终建议

若只看选刊，Defence Technology 比 TNSE 更适合这篇稿件。  
若看当前稿件成熟度，还不建议直接提交。建议至少完成“匿名稿、声明、指标定义、致谢、语言错误、防务应用定位”这六项修改后再投稿。

预估修改后投稿结果：有较大概率进入外审；外审大概率是 Major Revision，而不是直接 Reject。若能补足强 baseline、复杂度分析和内部消融，录用机会会明显提高。
