# GPU 服务器实验操作手册（4090 / Ubuntu 20.04）

> 从零开始到出结果，按顺序复制执行即可。所有命令假设用 bash。
> 遇到问题看最后的"常见问题"。

## 第 1 步：拉代码（含全部数据源，无需另外传文件）

```bash
git clone -b wlh https://github.com/Liquorice-Ma/teal.git
cd teal
```

## 第 2 步：装环境（约 10 分钟）

```bash
conda create -n teal python=3.10 -y
conda activate teal

# 先看驱动支持的 CUDA 版本（右上角 CUDA Version，>=12.1 即可用 cu121）
nvidia-smi

pip install torch --index-url https://download.pytorch.org/whl/cu121
# 装 PyG 扩展（预编译 wheel，把版本号自动带入）
pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-$(python -c 'import torch;print(torch.__version__)').html
pip install -r requirements.txt
pip install scipy pandas matplotlib networkx

# 自检：应打印 True
python -c "import torch; print(torch.cuda.is_available())"
```

## 第 3 步：重建数据（约 2 分钟）

```bash
cd run
python prepare_starlink.py --size-x 22 --size-y 72 --capacity 1000
python prepare_starlink.py --size-x 22 --size-y 72 --drop-isl-percent 5
python prepare_starlink.py --size-x 22 --size-y 72 --drop-isl-percent 10
```

## 第 4 步：冒烟测试（约 2 分钟，确认 GPU 能跑）

```bash
python teal.py --obj min_max_link_util --topo Starlink2272.json \
  --tm-model starlink --prune-demands --obs-ratio 0.5 --hist-len 3 \
  --epochs 5 --seed 0 \
  --slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
  --slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101
```

正常应看到训练进度条且速度明显快于 CPU。记下单个 epoch 耗时。

## 第 5 步：收敛探测 pilot（约 0.5–1 小时，决定正式 epochs）

```bash
python teal.py --obj min_max_link_util --topo Starlink2272.json \
  --tm-model starlink --prune-demands --obs-ratio 0.3 --mask-mode embed \
  --hist-len 3 --epochs 300 --early-stop True --seed 0 \
  --slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
  --slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101 \
  2>&1 | tee pilot.log
```

看 val obj 从第几个 epoch 开始不再明显下降（MLU 越低越好），
在该拐点基础上加 20% 余量作为正式 EPOCHS（拿不准就把 pilot.log 发回来）。

## 第 6 步：LP oracle（一次性，约 10 分钟，PR 归一化的分母）

```bash
python lp_oracle.py --topo Starlink2272.json --tm-model starlink --prune-demands
# 输出 lp-oracle-Starlink2272.json.csv
```

## 第 7 步：跑实验矩阵（过夜）

```bash
# 主对比 + 消融先跑（把 100 换成第 5 步定的 EPOCHS）
EPOCHS=100 nohup bash run_experiments.sh A B > nohup-AB.log 2>&1 &

# 查看进度
tail -f nohup-AB.log

# A、B 跑完后再跑 C、D（D 会切换数据文件，不能与其他批并行）
EPOCHS=100 nohup bash run_experiments.sh C D > nohup-CD.log 2>&1 &
```

说明：

- 中断/断线不怕：重跑同一命令会自动跳过已完成的 run（记录在 `.exp-done`）。
- 单个 run 失败会打 `[FAIL]` 并继续，不阻塞后面。
- 每个 run 的详细日志在 `logs/` 下。

## 第 8 步：随时看图检查（实验跑一半也能看）

```bash
python process_results.py
ls result-figs/
# A-pr-vs-rho.png     主图：PR vs 观测率（PR 越低越好，Ours 应最低）
# A-flow-vs-rho.png   副目标：满足率
# B-ablation-*.png    消融
# C-zero-retrain.txt  零重训表
# D-*.png             敏感性
```

预警信号：

- Ours 的 PR 反超 baseline → 该 run 可能训练发散，去 `logs/` 查对应日志；
- 误差棒跨方法大面积重叠 → 需要加 EPOCHS 或加 seed。

## 第 9 步：回传结果

```bash
# 在本地 Mac 上执行（把 user@server 换成实际地址）
scp user@server:~/teal/run/results-summary.csv ~/Downloads/
scp user@server:~/teal/run/lp-oracle-*.csv ~/Downloads/
scp -r user@server:~/teal/run/result-figs ~/Downloads/
```

把这三样发回来即可完成论文图表和数字回填。

## 常见问题

| 现象 | 处理 |
|---|---|
| `torch-scatter` 编译报错 | 确认用了 `-f https://data.pyg.org/whl/...` 预编译源；仍失败则加 `--no-build-isolation` |
| `CUDA out of memory` | 单卡 24GB 正常够；若爆显存说明跑了未剪枝配置，确认命令带 `--prune-demands` |
| pip 装到系统 Python | 用 `python -m pip install ...`，先 `which python` 确认在 conda env 里 |
| 图上某条线缺失 | 对应批次还没跑完，正常；`grep FAIL nohup-*.log` 检查有无失败 run |
| 想重跑某个 run | 从 `.exp-done` 里删掉对应行再执行脚本 |
