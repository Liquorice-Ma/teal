# 稀疏观测卫星 TE 实验操作指南

> 面向新协作者。基于 Teal (SIGCOMM'23) 改造，研究稀疏流量观测下的 LEO 卫星网络
> 流量工程。照本指南从零到复现全部实验。

## 1. 项目一句话

模型只能看到部分源宿对的流量（稀疏观测），通过可学习掩码嵌入 + 掩码感知 GNN 门控 +
Transformer 历史时序补全，在 Starlink 22×72 拓扑上做接近全观测性能的流量分配。

## 2. 环境安装（Linux + NVIDIA GPU）

```bash
# 1) 取代码和数据（code/ 目录是卫星拓扑与流量数据，不在 git 里，需单独拷贝）
git clone <本项目仓库地址> teal && cd teal
scp -r <来源机器>:~/teal/code ./code

# 2) conda 环境（不要用仓库里的 environment.yml，那是 linux-64 旧锁定版本）
conda create -n teal python=3.10 -y && conda activate teal

# 3) PyTorch（按 nvidia-smi 显示的 CUDA 版本选 index-url，示例为 CUDA 12.1）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4) PyG 扩展（Linux+CUDA 有预编译 wheel；若报 "No module named torch"
#    则加 --no-build-isolation）
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<torch版本>+cu121.html

# 5) 其余依赖
pip install -r requirements.txt && pip install pandas

# 6) 自检
python -c "import torch, torch_scatter, torch_sparse; print(torch.cuda.is_available())"
```

Mac（仅调试用，CPU 慢）：`brew install miniforge`，torch 装 CPU 版，
torch-scatter/sparse 必须 `pip install --no-build-isolation` 源码编译。

## 3. 数据准备（一次性）

```bash
cd run
# 把 code/ 里的 CSV 拓扑和 npz 流量转成 Teal 格式
# capacity=1000 是定过标的负载水平（全观测 obj≈0.87，有拥塞压力），不要随意改
python prepare_starlink.py --size-x 22 --size-y 72 --capacity 1000
```

产物：`topologies/Starlink2272.json` + `traffic-matrices/starlink/*.pkl`（101 个时刻，
时间连续，天然是时序数据）。

## 4. 跑实验

标准命令（Starlink 22×72，稀疏观测 50%，历史窗口 3 步）：

```bash
cd run
python teal.py --obj total_flow --topo Starlink2272.json --tm-model starlink \
  --epochs 100 --admm-steps 2 --prune-demands \
  --obs-ratio 0.5 --hist-len 3 --seed 0 \
  --slice-train-start 0 --slice-train-stop 80 \
  --slice-val-start 80 --slice-val-stop 90 \
  --slice-test-start 90 --slice-test-stop 101
```

关键参数：

| 参数 | 含义 | 说明 |
|---|---|---|
| `--obs-ratio` | 观测率 | 1.0=原版 Teal 行为；消融取 1.0/0.7/0.5/0.3/0.1 |
| `--hist-len` | 历史窗口长度 | 1=关闭 Transformer 时序模块（消融开关）；>1 开启 |
| `--prune-demands` | 需求剪枝 | 卫星拓扑必开（250 万对→6334 对），B4 等小拓扑不用 |
| `--seed` | 随机种子 | 已全局固定，同种子结果完全可复现；正式实验跑 0/1/2 取均值 |
| `--admm-steps` | ADMM 微调步数 | 默认 2 |
| `--slice-*` | 训练/验证/测试切片 | 按 TM 时间序号划分，勿打乱时序 |

消融矩阵（每晚可跑完）：`obs-ratio {1.0,0.5,0.3,0.1} × hist-len {1,3} × seed {0,1,2}`。

## 5. 看结果

- 进度条末尾 `obj=` ：测试集满足需求比例（越高越好），`runtime=` ：单次推理秒数；
- `run/teal-total_flow-all.csv`：逐条结果（追加写入）；
- `run/teal-models/`：模型权重，文件名含 obs/hist 后缀，不同配置不会互相覆盖；
- 路径缓存 `topologies/paths/path-form/*pruned-6334*.pkl`：首跑自动生成（约 1 分钟），
  之后复用。

## 6. 改了哪些代码（相对原版 Teal）

| 文件 | 改动 |
|---|---|
| `lib/teal_env.py` | 稀疏观测（掩码/历史 TM 序列）、需求剪枝、完整 TM 仍用于 reward/ADMM（评估不受稀疏影响） |
| `lib/teal_actor.py` | 可学习掩码嵌入、时序-空间交叉注意力融合 |
| `lib/FlowGNN.py` | 掩码感知门控：未观测 path→edge 消息置零，edge→path 保留 |
| `lib/TemporalEncoder.py` | 新增：Transformer 历史序列编码（输入 log1p 归一化，勿删，删了训练会发散） |
| `run/teal_helper.py` / `run/teal.py` | 新增 CLI 参数、全局种子 |
| `run/prepare_starlink.py` | 新增：卫星数据格式转换 |

设计要点（勿破坏）：`env.get_obs()` 给模型的是稀疏观测；`env.step()` 内部用**完整
真值 TM** 算 reward 和 ADMM——两者分离是实验正确性的根基。

## 7. 当前进度与待办

已完成：全部模块实现并在 B4/UsCarrier/Starlink2272 上跑通；种子可复现；
文献调研与 Intro/Related Work 草稿在 `docs/`。

待办（按序）：
1. GPU 上大 epoch（100+）× 3 种子重跑消融——验证 hist-len 3 的训练稳定性
   （CPU 上 20 epochs 时 seed 间方差大）；
2. 训练/测试 demand 集合分离实验（验证"训练一次、集合漂移零重训"，需改 env 构图时机）；
3. 补 node 级稀疏采样模式（对齐 TEST 的实验设定）；
4. 实现两阶段 baseline（均值插值 MI + 一个补全类方法，参考 github.com/paper-TEST/TEST）；
5. 论文数字回填 `docs/introduction.md` 的 [XX] 占位符。
