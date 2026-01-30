from typing import Literal


class EarlyStopping:
    """
    Monitors a metric to stop training when improvement plateaus or a threshold is met.
    """
    def __init__(self, direction: Literal[1, -1] = 1, patience: int = 5, min_steps: int = 0) -> None:
        self.direction = direction
        self.patience = patience
        self.min_steps = min_steps

        self.best = float("inf") * -1 if direction > 0 else float("inf")
        self.bad_steps: int = 0
        self.steps: int = 0

        self.stopped = False

    def _is_improvement(self, value: float) -> float:
        return (value - self.best) * self.direction > 0

    def step(self, value: float) -> bool:
        """
        Updates the monitor with a new metric value and checks if training should stop.

        1. Ignore checks if total steps are less than `min_steps`.
        2. Increment `bad_steps` if no improvement over `best` is found.

        returns True if the stopping criteria are met (bad_steps > patience), False otherwise.
        """
        if self.stopped:
            return True

        self.steps += 1
        if self.steps < self.min_steps:
            return False

        if self._is_improvement(value):
            self.best = value
            self.bad_steps = 0
        else:
            self.bad_steps += 1

        should_stop = self.bad_steps > self.patience
        if should_stop:
            self.stopped = True
        return should_stop
