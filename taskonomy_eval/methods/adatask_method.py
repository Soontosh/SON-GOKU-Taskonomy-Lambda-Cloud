from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import math
import torch
from torch import nn

from son_goku import TaskSpec  # same TaskSpec used by SON-GOKU
from .base import MultiTaskMethod, register_method


@register_method("adatask")
class AdaTaskMethod(MultiTaskMethod):
    """
    AdaTask-style task-aware adaptive learning rate (Yang et al., AAAI 2023).

    High-level behavior:
      * Maintain per-task Adam-style statistics (exp_avg, exp_avg_sq) for
        each SHARED parameter.
      * Maintain a standard Adam-style state for TASK-SPECIFIC parameters.
      * For each step:
          1. Compute per-task losses L_i.
          2. Compute gradients of each L_i for shared parameters separately.
          3. Compute a single gradient (Σ w_i L_i) for task-specific params.
          4. Update shared params by applying each task’s Adam-like update
             in sequence (task-aware learning rates per parameter).
          5. Update task-specific params with standard Adam.

    Notes:
      * We use equal task weights by default (all ones). You can easily
        expose task-wise weights if you want to experiment.
      * The runner already moves `batch` to the correct device.
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: Sequence[TaskSpec],
        shared_param_filter,
        base_optimizer,  # not used directly; we only grab hyperparams
        lr: float | None = None,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tasks = list(tasks)
        self.n_tasks = len(self.tasks)

        # Device
        if device is not None:
            self.device = device
        else:
            self.device = next(self.model.parameters()).device

        # Hyperparams: default to the base_optimizer's lr if not provided
        if lr is None and base_optimizer is not None and len(base_optimizer.param_groups) > 0:
            lr = float(base_optimizer.param_groups[0]["lr"])
        self.lr = lr if lr is not None else 1e-3
        self.betas = betas
        self.eps = eps

        # Equal weights for tasks (you can change to something else if needed)
        self.task_weights = torch.ones(self.n_tasks, device=self.device, dtype=torch.float32)

        # Shared vs task-specific parameters
        shared_params = [p for p in self.model.parameters() if shared_param_filter(p)]
        if not shared_params:
            # Fallback: if filter selects nothing, treat all params as shared
            shared_params = [p for p in self.model.parameters() if p.requires_grad]
        self.shared_params: List[nn.Parameter] = shared_params

        # Task-specific params are all trainable params that are not in shared
        shared_set = set(self.shared_params)
        self.task_specific_params: List[nn.Parameter] = [
            p for p in self.model.parameters()
            if p.requires_grad and p not in shared_set
        ]

        # Per-parameter optimizer state
        #   For shared params:  step, per-task exp_avg list, per-task exp_avg_sq list
        #   For task-specific:  step, exp_avg, exp_avg_sq
        self.state: Dict[nn.Parameter, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_param_grads(params: List[nn.Parameter]) -> None:
        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def _ensure_shared_state(self, p: nn.Parameter) -> Dict[str, Any]:
        """
        Initialize state for a shared parameter if needed.
        """
        state = self.state.setdefault(p, {})
        if "step" not in state:
            state["step"] = 0
            state["exp_avg_list"] = [
                torch.zeros_like(p, memory_format=torch.preserve_format)
                for _ in range(self.n_tasks)
            ]
            state["exp_avg_sq_list"] = [
                torch.zeros_like(p, memory_format=torch.preserve_format)
                for _ in range(self.n_tasks)
            ]
        return state

    def _ensure_task_specific_state(self, p: nn.Parameter) -> Dict[str, Any]:
        """
        Initialize state for a task-specific parameter if needed.
        """
        state = self.state.setdefault(p, {})
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return state

    @torch.no_grad()
    def _adatask_update(
        self,
        shared_grads: List[List[torch.Tensor]],
        task_specific_grads: List[torch.Tensor],
    ) -> None:
        """
        Apply AdaTask-style updates to shared parameters, and Adam to task-specific.

        Arguments:
          shared_grads:      list over tasks, each is list over shared_params
          task_specific_grads: list over task_specific_params
        """
        beta1, beta2 = self.betas

        # --- Update shared parameters with task-wise Adam-like updates ---
        for pi, p in enumerate(self.shared_params):
            state = self._ensure_shared_state(p)
            state["step"] += 1
            step = state["step"]

            # bias correction (shared across tasks for this param)
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = self.lr / bias_correction1

            exp_avg_list: List[torch.Tensor] = state["exp_avg_list"]
            exp_avg_sq_list: List[torch.Tensor] = state["exp_avg_sq_list"]

            # Apply each task's update sequentially
            for t in range(self.n_tasks):
                grad_t = shared_grads[t][pi]
                m_t = exp_avg_list[t]
                v_t = exp_avg_sq_list[t]

                # Exponential moving averages
                m_t.mul_(beta1).add_(grad_t, alpha=1.0 - beta1)
                v_t.mul_(beta2).addcmul_(grad_t, grad_t, value=1.0 - beta2)

                # Bias-corrected
                m_hat = m_t / bias_correction1
                v_hat = v_t / bias_correction2

                denom = v_hat.sqrt().add_(self.eps)
                p.data.addcdiv_(m_hat, denom, value=-step_size)

        # --- Update task-specific parameters with standard Adam ---
        for pi, p in enumerate(self.task_specific_params):
            grad = task_specific_grads[pi]
            state = self._ensure_task_specific_state(p)
            state["step"] += 1
            step = state["step"]

            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            step_size = self.lr / bias_correction1

            m = state["exp_avg"]
            v = state["exp_avg_sq"]

            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            denom = v_hat.sqrt().add_(self.eps)
            p.data.addcdiv_(m_hat, denom, value=-step_size)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, batch: Mapping[str, Any], global_step: int) -> Dict[str, float]:
        """
        One training step of AdaTask.

        Assumes:
          * `batch` is already on the correct device (runner handles this).
          * `self.model` is on the same device.
        """
        self.model.train()

        raw_losses: List[torch.Tensor] = []

        # 1) Compute per-task losses
        for spec in self.tasks:
            Li = spec.loss_fn(self.model, batch)  # scalar tensor
            raw_losses.append(Li)

        losses_tensor = torch.stack(raw_losses)  # [T]
        # Ensure weights are on correct device / dtype
        task_weights = self.task_weights.to(losses_tensor.device, dtype=losses_tensor.dtype)

        # 2) Gradients w.r.t. shared parameters for each task separately
        shared_grads: List[List[torch.Tensor]] = []
        for i, Li in enumerate(raw_losses):
            # Scale by task weight
            Li_scaled = task_weights[i] * Li
            grads_i = torch.autograd.grad(
                Li_scaled,
                self.shared_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            grads_i_filled: List[torch.Tensor] = []
            for g, p in zip(grads_i, self.shared_params):
                if g is None:
                    grads_i_filled.append(torch.zeros_like(p))
                else:
                    grads_i_filled.append(g)
            shared_grads.append(grads_i_filled)

        # 3) Gradients for task-specific parameters: sum_i w_i * L_i
        task_specific_grads: List[torch.Tensor] = []
        if self.task_specific_params:
            scalar_loss_ts = torch.sum(task_weights * losses_tensor)
            grads_ts = torch.autograd.grad(
                scalar_loss_ts,
                self.task_specific_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            for g, p in zip(grads_ts, self.task_specific_params):
                if g is None:
                    task_specific_grads.append(torch.zeros_like(p))
                else:
                    task_specific_grads.append(g)

        # 4) Apply AdaTask updates (no additional backward)
        self._adatask_update(shared_grads, task_specific_grads)

        # 5) Logging
        logs: Dict[str, float] = {}
        total_loss = float(torch.sum(task_weights * losses_tensor).detach().item())
        logs["loss/total"] = total_loss
        for w, Li, spec in zip(task_weights.tolist(), raw_losses, self.tasks):
            logs[f"loss/{spec.name}"] = float(Li.detach().item())
            logs[f"weight/{spec.name}"] = float(w)

        return logs

    # Optional checkpointing hooks, if you want them
    def state_dict(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "task_weights": self.task_weights.detach().cpu().tolist(),
            "adam_state": {
                id(p): {
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in st.items()
                }
                for p, st in self.state.items()
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.lr = state.get("lr", self.lr)
        self.betas = tuple(state.get("betas", self.betas))
        self.eps = state.get("eps", self.eps)
        tw = state.get("task_weights", None)
        if tw is not None:
            self.task_weights = torch.tensor(tw, device=self.device, dtype=torch.float32)

        # NOTE: Rebinding state to parameters is non-trivial because we keyed
        #       on the param object itself (id). If you need full resume from
        #       checkpoint, you can add your own mapping logic here.