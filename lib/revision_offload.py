"""独立推导的分块 Linear 预检；不是已经完成的 DOTE-sparse 基线。

核对依据：PredWanTE/DOTE@43e628bd457e31958eba4f696e3de869d3e7789e，
https://github.com/PredWanTE/DOTE/blob/43e628bd457e31958eba4f696e3de869d3e7789e/dote.py
仅按数学公式实现，不引入或分发该仓库源码。参数、梯度和 Adam 状态留在 CPU；
计算精度不变。分块只改变矩阵求和顺序，不截断 OD、隐藏单元或路径。
"""
import math

import torch
from torch import nn
from torch.autograd.function import once_differentiable


class _StagedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, bias, axis, *weights):
        ctx.axis = axis
        ctx.save_for_backward(inputs, *weights)
        output_width = sum(w.shape[0] for w in weights) if axis == 0 else weights[0].shape[0]
        output = inputs.new_zeros((inputs.shape[0], output_width))
        offset = 0
        for weight in weights:
            size = weight.shape[axis]
            staged = weight.to(inputs.device)
            if axis == 0:
                output[:, offset:offset + size] = inputs @ staged.t()
            else:
                output.add_(inputs[:, offset:offset + size] @ staged.t())
            offset += size
            del staged
        output.add_(bias.to(inputs.device))
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        inputs, *weights = ctx.saved_tensors
        grad_inputs = torch.zeros_like(inputs) if ctx.needs_input_grad[0] else None
        grad_bias = grad_output.sum(dim=0).cpu()
        gradients, offset = [], 0
        for weight in weights:
            size = weight.shape[ctx.axis]
            if ctx.axis == 0:
                part = grad_output[:, offset:offset + size]
                gradient = (part.t() @ inputs).cpu()
                if grad_inputs is not None:
                    staged = weight.to(inputs.device)
                    grad_inputs.add_(part @ staged)
                    del staged
            else:
                gradient = (grad_output.t() @ inputs[:, offset:offset + size]).cpu()
                if grad_inputs is not None:
                    staged = weight.to(inputs.device)
                    grad_inputs[:, offset:offset + size] = grad_output @ staged
                    del staged
            gradients.append(gradient)
            offset += size
        return grad_inputs, grad_bias, None, *gradients


class CPUStagedLinear(nn.Module):
    """第一层按输入列分块，其他层按输出行分块；不支持高阶导数。"""
    def __init__(self, in_features, out_features, chunk_size=32768, axis=0,
                 dtype=torch.float64):
        super().__init__()
        if min(in_features, out_features, chunk_size) < 1 or axis not in (0, 1):
            raise ValueError('Linear 维度、分块大小必须为正，axis 必须为0或1')
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('只预检 FP32/FP64，不自动降低精度')
        self.in_features, self.out_features = in_features, out_features
        self.axis = axis
        shape = [out_features, in_features]
        bound = 1 / math.sqrt(in_features)
        shards = []
        for start in range(0, shape[axis], chunk_size):
            size = shape.copy()
            size[axis] = min(chunk_size, shape[axis] - start)
            # 原 Linear 先按 FP32 初始化再转 double；采用完整 fan_in 而非分块 fan_in。
            shards.append(nn.Parameter(torch.empty(size, dtype=torch.float32).uniform_(-bound, bound).to(dtype)))
        self.weights = nn.ParameterList(shards)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32).uniform_(-bound, bound).to(dtype))

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError('输入必须为 batch×完整特征，不允许省略零 OD')
        if inputs.dtype != self.bias.dtype:
            raise ValueError('输入与权重精度必须一致')
        if any(p.device.type != 'cpu' for p in self.parameters()):
            raise ValueError('offload 参数必须留在 CPU，不要对模型调用 cuda()')
        return _StagedLinear.apply(inputs, self.bias, self.axis, *self.weights)

    @torch.no_grad()
    def copy_from_dense(self, layer):
        if layer.weight.shape != (self.out_features, self.in_features):
            raise ValueError('稠密参考维度不匹配')
        offset = 0
        for weight in self.weights:
            size = weight.shape[self.axis]
            weight.copy_(layer.weight.narrow(self.axis, offset, size))
            offset += size
        self.bias.copy_(layer.bias)


class OffloadedMLP(nn.Module):
    """MAXUTIL 网络形状：HQ→128→128→128→128→P；返回 Sigmoid 前的 logits。

    分块初始化分布与完整 Linear 相同，但不承诺相同 seed 跨布局逐位相同。
    一致性验证必须显式复制同一初始权重，不能比较两次独立随机初始化。
    """
    def __init__(self, pairs, paths=4, history=3, width=128, chunk_size=32768,
                 dtype=torch.float64):
        super().__init__()
        if min(pairs, paths, history, width) < 1:
            raise ValueError('网络维度必须为正')
        dimensions = [pairs * history, width, width, width, width, pairs * paths]
        self.layers = nn.ModuleList([
            CPUStagedLinear(a, b, chunk_size, 1 if i == 0 else 0, dtype)
            for i, (a, b) in enumerate(zip(dimensions, dimensions[1:]))])

    def forward(self, inputs):
        for layer in self.layers[:-1]:
            inputs = torch.relu(layer(inputs))
        return self.layers[-1](inputs)


def normalized_path_weights(logits, valid=None):
    """输入 batch×OD×候选槽；epsilon 只加到有效路径，并按完整 OD 归一化。"""
    if logits.ndim != 3:
        raise ValueError('路径 logits 必须为 batch×OD×候选槽')
    weights = torch.sigmoid(logits) + 1e-16
    if valid is not None:
        if valid.dtype != torch.bool or tuple(valid.shape) != tuple(logits.shape[1:]):
            raise ValueError('有效路径掩码必须为 OD×候选槽的布尔矩阵')
        if not bool(valid.any(dim=-1).all()):
            raise ValueError('存在没有有效路径的 OD，不能静默丢弃或改变原公式')
        weights = weights * valid.to(logits.device)
    return weights / weights.sum(dim=-1, keepdim=True)


def maxutil_objective(utilization):
    """每样本全边 max；并列最大值均分梯度，batch 再取平均。

    m>0 时为 m/stop_gradient(m)，m=0 时为1-m；绝不能化简为常数1。
    """
    if utilization.ndim != 2 or min(utilization.shape) < 1:
        raise ValueError('利用率必须为非空 batch×完整边集合')
    maximum = utilization.amax(dim=-1)
    denominator = torch.where(maximum.detach() == 0, torch.ones_like(maximum), maximum.detach())
    loss = torch.where(maximum.detach() == 0, 1 - maximum, maximum / denominator)
    return loss.mean(), maximum


def offload_inventory(nodes, paths=4, history=3, width=128, chunk_size=32768,
                      itemsize=8, batch=1):
    """解析预算仅覆盖 MLP 内核；完整路径图、归一化及评分另行实测。"""
    if min(nodes - 1, paths, history, width, chunk_size, batch) < 1 or itemsize not in (4, 8):
        raise ValueError('资源预检参数不合法')
    pairs = nodes * (nodes - 1)
    dimensions = [history * pairs, width, width, width, width, paths * pairs]
    parameters = sum(a * b + b for a, b in zip(dimensions, dimensions[1:]))
    largest_shard = max(width * min(chunk_size, history * pairs),
                        width * min(chunk_size, paths * pairs), width * min(chunk_size, width))
    activation_elements = batch * sum(dimensions)
    return dict(pairs=pairs, parameters=parameters, itemsize=itemsize,
                cpu_parameter_gradient_adam_bytes=parameters * itemsize * 4,
                largest_weight_shard_bytes=largest_shard * itemsize,
                saved_linear_input_elements=activation_elements,
                # 为 staging、输入/输出、反传及激活留出保守工作空间；不是实测峰值。
                kernel_device_working_estimate_bytes=itemsize * (8 * activation_elements + 4 * largest_shard),
                scope='MLP_only_excludes_path_graph_and_full_training')
