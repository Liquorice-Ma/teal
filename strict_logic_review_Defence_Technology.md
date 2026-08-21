# 严格科研论文逻辑审核与修改建议

审稿对象：`interactcadsample.tex`  
目标期刊：Defence Technology  
审核重点：全文科学逻辑、方法论闭环、实验支撑关系、创新点真实性  
使用 skill：`academic-paper-reviewer`，按 `methodology-focus + Devil's Advocate` 方式执行。

## 一、总评：当前真正的核心风险

这篇论文目前最大的问题不是格式、语言或匿名声明，而是“核心论证链没有完全闭合”。文章试图建立如下逻辑：

1. 红外图像主要提供 thermal saliency；可见光图像主要提供 texture/detail。
2. 现有方法没有充分区分两种模态的贡献，因此存在 cross-modal suppression 或 modality competition。
3. 所以需要 decoupled attention：用 channel attention 强化红外热显著信息，用 spatial attention 强化可见光纹理信息。
4. 再用 Swin Transformer 模块进行 global coupling，以防止小目标丢失并建模长程依赖。
5. 实验指标提升证明该设计有效。

这个故事线本身是有吸引力的，但当前论文的问题在于：第 2、3、4、5 步之间的证据链不够严密。审稿人如果严格看逻辑，会追问：

- 你是否真的证明了现有方法存在 cross-modal suppression？
- 你的 MARM 是否真的进行了“模态解耦”，还是只是普通 spatial/channel attention？
- channel attention 是否一定对应 infrared saliency？spatial attention 是否一定对应 visible texture？
- Swin shifted-window 是否足以支撑“global coupling”这个概念？
- 当前实验指标是否能证明“小目标不丢失”和“防务场景有效”？
- 消融实验是否证明了 proposed mechanism，而不仅是证明“加模块比删模块好”？

严格判断：当前稿件可以作为一篇工程型图像融合论文，但作为 Defence Technology 投稿稿，核心机制的科学解释还偏弱。建议把论文从“堆模块 + 指标提升”改成“假设—机制—验证”的完整科研逻辑。

## 二、核心逻辑链问题

### 问题 1：问题定义没有被实验证明，只是被叙述性提出

论文提出的关键问题是：现有方法将两种模态混合处理，导致 cross-modal interference、modality competition、小目标丢失和细节损失。

这个问题在 Introduction 中是通过文字提出的，但正文没有给出直接证据证明：

- 哪些 baseline 出现了 cross-modal suppression？
- cross-modal suppression 在图像上表现为什么？
- 是否有 attention map、feature response map、target contrast、edge preservation 或 saliency degradation 的定量证据？
- 小目标丢失是否真的由“缺少全局建模”导致，而不是由 loss、训练数据、分辨率差异或插值造成？

当前写法属于“提出一个合理但未验证的解释”。严格审稿人会认为这是 post-hoc explanation，即先看到模型效果好，再用 cross-modal suppression 解释。

#### 修改建议

需要在 Introduction 或 Experiments 中补一个“problem evidence”小节，用实验或可视化说明问题确实存在。

可选验证方式：

1. 对比 baseline 的 fused images，在车辆、船只、道路边缘等区域标框，展示：
   - 红外目标被可见光背景淹没；
   - 可见光纹理被红外热斑覆盖；
   - 小目标边缘不清晰。

2. 引入局部定量指标：
   - target-to-background contrast；
   - edge intensity around targets；
   - local AG/SF in target ROIs；
   - object saliency retention ratio。

3. 增加一张 attention/feature visualization，说明普通 shared attention 会把响应集中在背景纹理或大面积亮区，而 MARM 对目标和纹理有更平衡响应。

推荐新增逻辑句：

```text
To verify that the performance degradation of existing methods is indeed related to cross-modal competition, we further analyze target-level contrast and edge preservation in selected defense-oriented regions of interest.
```

### 问题 2：核心创新“modality-specific decoupling”与方法公式不完全一致

论文声称：

- channel attention branch enhances infrared thermal saliency；
- spatial attention branch enhances visible texture；
- MARM explicitly decouples modality-specific contributions。

但方法公式显示：

1. 先将 `I_IR` 和 `I_VIS` concatenate；
2. 经过卷积得到共享特征 `F_C1^Out`；
3. CA、RB、SAB、CAB 都作用在共享特征上；
4. 最后将 `F_CAB^Out * F_RB^Out` 和 `F_SAB^Out * F_RB^Out` 相加。

这意味着 SAB 和 CAB 并没有直接接收单独的 visible 或 infrared 分支。它们处理的是已经混合的特征。因此，当前方法严格来说只能证明“spatial/channel attribute decoupling”，不能直接证明“modality-specific decoupling”。

这是全文最核心的逻辑风险。

#### 审稿人可能的质疑

- If the infrared and visible inputs are concatenated before MARM, how can the model ensure that the channel attention branch corresponds to infrared thermal saliency and the spatial attention branch corresponds to visible texture?
- Is the proposed decoupling modality-specific or merely attention-type-specific?
- What prevents the channel attention branch from emphasizing visible channels or the spatial attention branch from emphasizing infrared structures?

#### 修改建议

有两个可选方向。

方案 A：不改模型，只改论证，降低声称强度。

将“modality-specific decoupling”改成更准确的：

```text
attribute-oriented decoupling of spatial and channel responses after multimodal embedding
```

或者：

```text
The MARM does not explicitly separate raw modalities, but it decomposes the fused representation into spatial and channel enhancement pathways, which are designed according to the dominant attributes of visible and infrared imagery.
```

这样逻辑更安全，但创新性会稍弱。

方案 B：补实验支撑原声称。

如果想保留“modality-specific decoupling”，需要增加证明：

1. 分别屏蔽 IR 或 VIS 输入，观察 SAB/CAB 响应变化；
2. 计算 SAB attention map 与 visible gradient map 的相关性；
3. 计算 CAB response 与 infrared saliency/intensity map 的相关性；
4. 展示 MARM 输出中特定通道或空间区域与两个模态的对应关系。

可新增实验表：

| Variant | VIS-gradient correlation | IR-saliency correlation | Target contrast | AG | VIF |
| --- | --- | --- | --- | --- | --- |
| Shared attention | ... | ... | ... | ... | ... |
| SAB only | ... | ... | ... | ... | ... |
| CAB only | ... | ... | ... | ... | ... |
| MARM | ... | ... | ... | ... | ... |

如果没有这些证据，不建议继续强说“explicitly resolves modality discrepancies”。

### 问题 3：Swin Transformer 被称为 global coupling，但实际是局部窗口和移位窗口交互

论文将 STM 称为 global coupling，并多次说可以 model long-range dependencies。Swin Transformer 确实比 CNN 有更大上下文建模能力，但 Stage 1 两个 blocks 的 W-MSA/SW-MSA 仍是窗口内和相邻窗口交互，不等于全图 global self-attention。

严格审稿人会认为“global coupling”表述过度，尤其如果网络只使用 two Swin-T Stage-1 blocks，其有效感受野和真正全局建模之间存在差距。

#### 逻辑缺口

当前论文没有说明：

- STM 的输入 patch/window size；
- 使用多少 Swin blocks；
- 两个 blocks 后信息能传播多远；
- 为什么这足以保留遥感图像中的小目标；
- global coupling 与普通 SwinFusion 的差异在哪里。

#### 修改建议

1. 如果不增加实验，建议将 “global coupling” 改成更稳妥的 “cross-window contextual coupling” 或 “hierarchical contextual coupling”。
2. 若保留 “global”，需要增加证据：
   - attention distance analysis；
   - effective receptive field visualization；
   - long-range dependency ablation，如 window size 4/7/8 对比；
   - W-MSA only vs W-MSA+SW-MSA；
   - STM depth 对性能影响。
3. 在方法部分明确说明：
   - patch size；
   - window size；
   - embedding dimension；
   - number of heads；
   - number of blocks；
   - whether patch merging is used。

### 问题 4：实验指标不能直接支撑“小目标保留”和“防务应用有效”

论文强调 small targets、vehicles、ships、defense-oriented reconnaissance，但评价指标主要是 SF、AG、PSNR、MSE、VIF。这些指标更多反映图像统计质量，不等价于防务场景中的目标可见性或任务性能。

#### 逻辑缺口

- AG 高可能代表边缘增强，也可能代表噪声增强。
- SF 高可能代表纹理丰富，也可能代表伪影增加。
- PSNR/MSE 在无 ground truth fusion 任务中解释困难。
- VIF 不一定能反映红外目标是否保留。
- 没有目标检测、分割或人工可视判别实验支撑 defence utility。

因此当前结论“有利于 downstream object detection and segmentation”属于未验证外推。

#### 修改建议

至少加入一种 defense-relevant evaluation：

1. 目标 ROI 指标：
   - target contrast；
   - target-background ratio；
   - local entropy；
   - local gradient around targets。

2. 下游任务验证：
   - 在 DroneVehicle 上跑 vehicle detector；
   - 比较 fused image 输入下的 mAP、Recall、F1；
   - 或用 segmentation model 比较 mIoU。

3. 主观评价：
   - 邀请若干评估者对目标可见性、纹理保真度、伪影程度评分；
   - 虽然不如检测实验强，但比只给 AG/SF 更有说服力。

如果暂时无法补下游实验，至少删除或弱化 “beneficial for downstream tasks” 这类强结论。

### 问题 5：PSNR/MSE/VIF 的使用存在方法论风险

这是比格式更严重的逻辑问题。图像融合任务没有真实融合图像，reference-based metrics 必须非常谨慎。

当前稿件没有解释 PSNR/MSE/VIF 的 reference。若使用 visible image 作为 reference，则方法更接近 visible 会获得更高 PSNR；若使用 infrared 作为 reference，则热目标更强的方法获益；若使用 max/mean 伪参考，则评价又依赖人为构造。

这会影响论文最重要的定量结论。

#### 修改建议

必须明确：

- reference 是什么；
- 为什么该 reference 合理；
- 是否对所有方法统一使用同一协议；
- 是否同时报告 no-reference fusion metrics，如 EN、MI、Qabf、SSIM-based fusion metric、Nabf、SCD 等。

建议不要把 PSNR/MSE/VIF 作为主要证明，而是将其作为辅助指标。核心结果应更多依赖 fusion community 常用无参考指标和任务相关指标。

推荐补充指标：

- EN；
- MI；
- SD；
- Qabf；
- SCD；
- MS-SSIM 或 FMI；
- target contrast / detection metrics。

### 问题 6：数据构造可能引入配准偏差或数据泄漏

文中说通过随机裁剪 visible patch，然后四角扰动并对 infrared 做 homography inverse，再裁取 patch。这个数据增强策略有合理性，但也带来两个逻辑风险。

#### 风险 A：训练和测试可能不独立

如果先从全部图像生成 4000 patches，再随机 9:1 划分，那么训练和测试可能来自同一原始场景，甚至相邻或重叠区域。这会导致测试结果偏乐观。

#### 风险 B：人工 homography 扰动可能改变任务定义

图像融合通常假设输入已经配准。你们人为制造 residual misregistration，但网络结构并没有显式 registration module。需要解释：

- 模型是否被设计为处理 misregistration？
- 如果是，为什么没有 registration loss 或 alignment module？
- 如果不是，为什么训练时要引入 homography perturbation？
- 测试集是否也有同样扰动？
- 扰动是否对所有 baseline 公平？

#### 修改建议

1. 明确 train/test split 是 scene-level 还是 patch-level。
2. 明确 homography perturbation 只用于 training augmentation，还是 train/test 都用。
3. 给出无扰动和有扰动两种测试结果，证明模型不是只适应人工扰动。
4. 如果目标是 robust fusion under misregistration，应在贡献和实验中明确提出并验证。

### 问题 7：对比实验无法排除“训练协议优势”

论文中不同 baseline 的适配方式不清楚。传统方法、预训练方法、重新训练方法混在一起比较，容易出现不公平。

#### 审稿人会问

- AUIF、DenseFuse、FusionGAN、SwinFusion 是否在同一训练集重新训练？
- 如果没有重新训练，是否使用作者原始权重？
- 输入尺寸、灰度化、归一化、训练 epoch 是否一致？
- LatLRR 是 MATLAB，其他是 PyTorch，运行环境不同是否影响 runtime 或可复现？
- 为什么有些数据集排除 LatLRR 和 DenseFuse？“could not be fairly adapted”过于笼统。

#### 修改建议

增加一个 baseline implementation table：

| Method | Code source | Training strategy | Input size | Re-trained? | Parameters fixed? |
| --- | --- | --- | --- | --- | --- |
| AUIF | official/unofficial | ... | ... | yes/no | ... |
| FusionGAN | ... | ... | ... | ... | ... |
| SwinFusion | ... | ... | ... | ... | ... |
| Proposed | own | ... | ... | yes | ... |

同时对排除方法给出技术性具体原因。

### 问题 8：消融实验只能证明“大模块有用”，不能证明“核心机制正确”

当前 module ablation：w/o STM、w/o MARM、Proposed。这个实验能说明两个模块对结果有贡献，但不能证明：

- decoupled attention 比普通 attention 好；
- parallel spatial/channel attention 比 sequential attention 好；
- channel/spatial 分支确实分别对应 IR/VIS 属性；
- CA/RB/SAB/CAB 各自必要；
- MARM 先于 STM 的顺序优于 STM 先于 MARM。

因此消融无法支撑核心创新叙述。

#### 修改建议

补充以下消融：

1. MARM 内部：
   - w/o CA；
   - w/o SAB；
   - w/o CAB；
   - w/o RB。

2. 注意力设计：
   - single shared attention；
   - sequential channel-spatial attention；
   - parallel channel-spatial attention；
   - proposed MARM。

3. 模块顺序：
   - MARM → STM；
   - STM → MARM；
   - direct concatenation → STM；
   - direct concatenation → MARM。

4. STM 配置：
   - W-MSA only；
   - W-MSA + SW-MSA；
   - different window sizes。

这样才能证明“decouple first, couple later”的主张。

### 问题 9：Related Work 与实验设计之间断裂

Related Work 中列举了很多近年强方法，但实验只比较少数较老或较基础方法。逻辑上会给审稿人一种感觉：作者知道近年 SOTA，但没有与它们比较。

这比“引用少”更严重，因为它会影响论文贡献可信度。

#### 修改建议

至少做以下之一：

1. 增加近期方法实验比较；
2. 如果无法比较，说明不可比较原因，例如：
   - no public code；
   - trained on incompatible data；
   - requires semantic labels unavailable in our datasets；
   - input modality mismatch；
   - excessive computational requirements。
3. 在 Related Work 中降低对未比较方法的铺陈，避免让审稿人自然期待实验中出现这些方法。

### 问题 10：结论中有若干过度外推

论文结论说方法适合 downstream target detection and semantic segmentation，未来用于 on-board deployment。但当前没有：

- detection/segmentation 实验；
- deployment runtime；
- model size；
- energy consumption；
- robustness analysis。

因此这些属于合理展望，但不能作为已经证明的结论。

#### 修改建议

将强结论改成弱结论：

```text
These results suggest the potential of the proposed method for downstream defence-oriented interpretation tasks, although direct task-level validation will be investigated in future work.
```

不要写成“demonstrating its benefit for downstream tasks”，除非补实验。

## 三、逐段逻辑审核意见

### Abstract

当前摘要逻辑完整，但存在两个强声称风险：

1. “model global context poorly leads to loss of small targets”没有直接实验支撑；
2. “decouples complementary contributions of the two modalities”与实际共享特征公式不完全一致。

建议摘要中降低这两处绝对表述。例如：

```text
MARM decomposes the embedded multimodal representation into spatial and channel enhancement pathways motivated by the complementary characteristics of visible and infrared imagery.
```

### Introduction

Introduction 的问题是动机讲得顺，但证据不足。建议加入一段更明确的 gap：

- existing attention methods mix modalities early；
- their attention weights are difficult to attribute to modality-specific cues；
- this may weaken target saliency or texture preservation；
- therefore this paper investigates attribute-oriented decoupling before contextual coupling。

注意用 “may” 或 “can” 比 “inevitably induces” 更安全。

### Related Work

Related Work 目前覆盖较广，但逻辑功能不够强。它现在像文献罗列，而不是为本文机制服务。

建议将最后一段改成明确的 gap synthesis：

1. CNN/GAN methods：local/detail focused but global dependency limited；
2. Transformer methods：global modeling improved but modality attributes often mixed；
3. task-guided methods：improve semantic utility but require extra labels or priors；
4. 本文定位：在不引入额外语义标签的情况下，研究 multimodal attribute decoupling before contextual coupling。

### Proposed Method

方法部分最大问题是“术语强于公式”。

建议将 MARM 命名中的“multi-attention residual”保留，但把机制解释改为：

- input-level：IR/VIS concatenate；
- representation-level：shared multimodal embedding；
- attribute-level：spatial/channel enhancement；
- context-level：Swin-based cross-window coupling。

这样逻辑更严谨。

### Experiments

实验部分目前证明了“指标上 proposed 更好”，但没有充分证明“为什么更好”。建议把实验结构调整为：

1. Overall comparison；
2. Defence-oriented ROI analysis；
3. Attribute decoupling validation；
4. Global coupling validation；
5. Ablation study；
6. Complexity analysis。

### Conclusion

Conclusion 应避免说得比实验多。尤其 downstream tasks 和 deployment 只能作为 future work，不能作为已验证贡献。

## 四、严格审稿下的严重性排序

### Critical Issues

1. “modality-specific decoupling”与实际共享特征公式不一致。
2. PSNR/MSE/VIF reference 未定义，可能影响全部定量结论可信度。
3. 数据划分可能存在 patch-level leakage，独立测试性不明确。
4. 实验指标不能直接支撑 small target preservation 和 defence utility。

### Major Issues

1. global coupling 表述强于 Swin window attention 的实际能力。
2. 消融实验不足以证明核心机制。
3. baseline 选择偏弱，近年强方法缺席。
4. homography perturbation 的任务意义和公平性解释不足。
5. 缺少复杂度、runtime、部署可行性分析。

### Minor but still important Issues

1. Related Work 与实验比较不一致。
2. 若干结论外推过强。
3. 个别事实和英文表达错误。
4. Defence Technology 应用场景还可更集中。

## 五、建议重构后的论文主线

当前主线是：

```text
IR/VIS have complementary information → propose MARM + STM → metrics improve.
```

建议改为更严格的科研主线：

```text
Defense-oriented IR/VIS fusion requires simultaneous target saliency and structural detail preservation.
Existing methods either mix modality attributes too early or model context before suppressing redundant responses.
We hypothesize that attribute-level decoupling before contextual coupling can better preserve thermal targets and visible structures.
MARM implements spatial/channel attribute enhancement on multimodal embeddings.
STM performs cross-window contextual coupling on refined features.
ROI-level, no-reference, reference-based, and ablation experiments jointly verify the hypothesis.
```

这条主线更容易说服审稿人，因为它把“假设—方法—验证”连接起来了。

## 六、最值得优先补的实验

如果时间有限，最优先补以下 4 类实验：

### 1. 指标定义和新增无参考指标

必须明确 PSNR/MSE/VIF reference，并新增 EN、MI、Qabf、SCD 等无参考融合指标。

### 2. MARM 机制验证

至少补：

- w/o SAB；
- w/o CAB；
- w/o CA；
- shared attention vs proposed MARM。

### 3. ROI-level target preservation

选车辆、船只、港口、夜间目标区域，计算 target contrast 或 local AG/SF，证明小目标确实更清晰。

### 4. 数据划分说明

如果实际是 scene-level split，明确写出。若不是，建议重新划分并重跑关键表格，否则 reviewer 会质疑泛化性。

## 七、最小文字改法：不补大量实验时如何降低风险

如果无法补实验，至少应通过文字降低逻辑风险：

1. 将 “explicitly decouples modality-specific contributions” 改为 “decouples spatial and channel enhancement pathways motivated by modality characteristics”。
2. 将 “global coupling” 改为 “cross-window contextual coupling”。
3. 将 “ensuring small targets are not lost” 改为 “mitigating the loss of small targets”。
4. 将 “demonstrates benefit for downstream tasks” 改为 “suggests potential for downstream tasks”。
5. 明确所有 reference-based metrics 的计算协议。
6. 不再声称实验完全证明防务部署，只说具备潜力。

## 八、最终严格结论

从严格科研逻辑角度看，这篇稿件目前的主要问题是：论证强度大于证据强度。

方法本身不是没有价值，实验结果也显示 proposed 方法在多个指标上有优势。但当前文本把这种优势解释为“模态解耦 + 全局耦合 + 小目标保留 + 防务任务有效”，而实验还没有充分排除其他解释，例如：

- 模型容量更大；
- loss 权重更适合当前数据；
- reference metrics 偏向 proposed 输出；
- patch-level 数据划分带来偏乐观结果；
- baseline 没有充分调参或重新训练；
- attention 分支只是普通空间/通道增强，而非真正模态解耦。

建议投稿前至少补充 MARM 内部消融、指标定义、数据划分说明和 ROI-level 目标保留分析。若能完成这些，论文逻辑会从“工程经验上有效”提升到“科学机制上较可信”。

当前严格推荐：Major Revision before submission。
