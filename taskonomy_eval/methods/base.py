from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, Type


METHOD_REGISTRY: Dict[str, "type[MultiTaskMethod]"] = {}


def register_method(name: str) -> Callable[[Type["MultiTaskMethod"]], Type["MultiTaskMethod"]]:
    """
    Decorator to register a multi-task training method under a string name.
    Example names: "son_goku", "gradnorm", "pcgrad", etc.
    """
    def deco(cls: Type["MultiTaskMethod"]) -> Type["MultiTaskMethod"]:
        METHOD_REGISTRY[name] = cls
        return cls
    return deco


class MultiTaskMethod(ABC):
    """
    Generic interface for a multi-task training algorithm.
    It owns *how* tasks are weighted / scheduled given a multi-task batch.
    """

    @abstractmethod
    def step(self, batch: Mapping[str, Any], global_step: int) -> Dict[str, float]:
        """
        Consume one multi-task batch, update model/optimizer, and return a dict of
        scalar logs (e.g., per-task losses, weights).
        """
        ...

    def state_dict(self) -> Dict[str, Any]:  # optional, for checkpointing algorithms
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass
