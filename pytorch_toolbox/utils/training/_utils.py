from enum import Enum
from typing import Optional

import torch
from torch import nn, Tensor

from pytorch_toolbox.defaults import Tensors, default_hardware, ModuleList, ParamList, List
from pytorch_toolbox.utils import if_none, is_listy


class Phase(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


def split_model_idx(model: nn.Module, idxs: List[int]) -> ModuleList:
    "Split `model` according to the indices in `idxs`."
    layers = flatten_model(model)
    if idxs[0] != 0:
        idxs = [0] + idxs
    if idxs[-1] != len(layers):
        idxs.append(len(layers))
    return [nn.Sequential(*layers[i:j]) for i, j in zip(idxs[:-1], idxs[1:])]


def flatten_model(m: nn.Module):
    return sum(map(flatten_model, m.children()), []) if num_children(m) else [m]


def num_children(m: nn.Module) -> int:
    return len(children(m))


def children(m: nn.Module) -> ModuleList:
    return list(m.children())


def split_layers_into_batch_norm_and_non_batch_norm(layer_groups: ModuleList) -> ModuleList:
    "Sort each layer in  `layer_groups` into batchnorm (`bn_types`) and non-batchnorm groups."
    split_groups = []
    for l in layer_groups:
        non_batch_norm_layers, batch_norm_layers = [], []
        for c in l.children():
            if isinstance(c, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                batch_norm_layers.append(c)
            else:
                non_batch_norm_layers.append(c)
        split_groups += [
            nn.Sequential(*non_batch_norm_layers),
            nn.Sequential(*batch_norm_layers),
        ]
    return split_groups


def trainable_params(m: nn.Module) -> ParamList:
    "Return list of trainable params in `m`."
    res = filter(lambda p: p.requires_grad, m.parameters())
    return res


def to_detach(b: Tensors):
    "Recursively detach lists of tensors in `b `"
    if is_listy(b):
        return [to_detach(o) for o in b]
    return b.detach() if isinstance(b, Tensor) else b


def requires_grad(m: nn.Module, b: Optional[bool] = None) -> Optional[bool]:
    "If `b` is not set `requires_grad` on all params in `m`, else return `requires_grad` of first param."
    ps = list(m.parameters())
    if not ps:
        return None
    if b is None:
        return ps[0].requires_grad
    for p in ps:
        p.requires_grad = b


def bn2float(module: nn.Module) -> nn.Module:
    "If `module` is batchnorm don't use half precision."
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        bn2float(child)
    return module


def model2half(model: nn.Module) -> nn.Module:
    "Convert `model` to half precision except the batchnorm layers."
    return bn2float(model.half())


def num_parameters(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
