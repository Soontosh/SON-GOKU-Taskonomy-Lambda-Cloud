from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import torch
from torch import nn, Tensor

def iter_named_parameters(module: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    for n, p in module.named_parameters(recurse=True):
        if p.requires_grad:
            yield n, p

def flatten_grads(params: Sequence[nn.Parameter]) -> Tensor:
    flats = []
    for p in params:
        if p.grad is None:
            flats.append(torch.zeros_like(p).view(-1))
        else:
            flats.append(p.grad.view(-1))
    if len(flats) == 0:
        return torch.tensor([])
    return torch.cat(flats, dim=0)

def zero_like_params(params: Sequence[nn.Parameter]) -> List[Tensor]:
    return [torch.zeros_like(p) for p in params]

def add_in_place(target: List[Tensor], source: Sequence[Tensor]) -> None:
    assert len(target) == len(source)
    for t, s in zip(target, source):
        t.add_(s)

def params_by_filter(module: nn.Module, pred) -> List[nn.Parameter]:
    return [p for _, p in module.named_parameters() if p.requires_grad and pred(p)]

def all_params(module: nn.Module) -> List[nn.Parameter]:
    return [p for _, p in module.named_parameters() if p.requires_grad]

def set_param_grads(params: Sequence[nn.Parameter], grads: Sequence[Tensor]) -> None:
    for p, g in zip(params, grads):
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)

def clear_grads(module: nn.Module) -> None:
    for p in module.parameters():
        p.grad = None

def vector_from_params(params: Sequence[nn.Parameter]) -> Tensor:
    """Concatenate data tensors (not grads) for size reference."""
    return torch.cat([p.detach().view(-1) for p in params], dim=0) if params else torch.tensor([])
