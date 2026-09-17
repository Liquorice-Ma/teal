"""历史预测与可追溯 checkpoint 的公共约束；纯数据部分不依赖 GPU。"""
import hashlib
import json
from pathlib import Path


def history_indices(target, length, input_lag=0, slice_start=0):
    if length < 1 or input_lag not in (0, 1):
        raise ValueError("历史长度必须为正，input_lag 仅支持0或1")
    if target < slice_start:
        raise ValueError("目标时刻早于当前切片")
    if input_lag == 0:
        return [max(slice_start, t) for t in range(target - length + 1, target + 1)]
    # 预测域跨切片读取已发生历史；序列开始处填零，不能复制目标 TM。
    return [t if t >= 0 else None for t in range(target - length, target)]


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        data = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(data.dtype).encode())
        digest.update(str(tuple(data.shape)).encode())
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def check_checkpoint(payload, expected):
    if payload.get("schema_version") != 1 or "state_dict" not in payload:
        raise ValueError("checkpoint 无可核验元数据；不能当成已训练模型")
    if payload.get("trained_epochs", 0) <= 0:
        raise ValueError("checkpoint 没有正数训练轮次，拒绝随机初始化")
    if payload.get("config") != expected:
        actual = payload.get("config", {})
        mismatch = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise ValueError(f"checkpoint 配置不匹配: {mismatch}")
    if state_digest(payload["state_dict"]) != payload.get("weights_sha256"):
        raise ValueError("checkpoint 权重哈希不匹配")


def save_checkpoint(path, state, config, trained_epochs):
    import torch
    if trained_epochs <= 0:
        raise ValueError("不能将 epochs=0 保存为已训练 checkpoint")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema_version=1, state_dict=state, config=config,
                   trained_epochs=trained_epochs, weights_sha256=state_digest(state))
    # 使用独占创建，防止不同 repeat 或测试误覆盖已有权重。
    with target.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def experiment_config(args, problems):
    """排除测试观测率、输出路径和 run-id，允许固定权重的跨观测率测试。"""
    fields = ("topo", "tm_model", "scale_factor", "obj", "seed", "obs_seed", "obs_type",
              "obs_sample", "train_obs_ratio", "input_lag", "hist_len", "num_path", "shared_paths",
              "layers", "mask_mode", "mask_init", "no_gate", "prune_demands", "demand_split",
              "slice_train_start", "slice_train_stop", "slice_val_start", "slice_val_stop")
    config = {key: getattr(args, key) for key in fields}
    config["topology_sha256"] = file_digest(problems[0][1])
    # 用输入数据内容建立身份，不把本机绝对路径写成复现实验的约束。
    signature = hashlib.sha256()
    for _, _, filename in problems:
        signature.update(file_digest(filename).encode())
    config["traffic_sequence_sha256"] = signature.hexdigest()
    return config
