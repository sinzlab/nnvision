"""Contains callable classes used to stop the MEI optimization process once it has reached an acceptable result."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any

from mei.domain import State
from mei.stoppers import OptimizationStopper
from ..tables.from_mei import MEI


class AdaptiveActivationStopper(OptimizationStopper):
    """Callable that stops the optimization process after a specified number of steps."""

    def __init__(self, num_iterations, key, method_hash, percent_of_max):
        """Initializes NumIterations.

        Args:
            num_iterations: The number of optimization steps before the process is stopped.
        """
        self.num_iterations = num_iterations
        stop_key = dict()
        stop_key.update(key)
        stop_key["method_hash"] = method_hash
        stop_key.pop("mei_seed")
        print(len(MEI & stop_key))
        self.stop_at = (MEI & stop_key).fetch("score").max() * percent_of_max

    def __call__(self, current_state: State) -> Tuple[bool, Optional[Any]]:
        """Stops the optimization process after a set number of steps by returning True."""
        if current_state.i_iter == self.num_iterations:
            return True, None
        if current_state.evaluation >= self.stop_at:
            return True, None
        return False, None

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.num_iterations})"
