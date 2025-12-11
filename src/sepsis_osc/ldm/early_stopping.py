from typing import Literal


class EarlyStopping:
    def __init__(
        self, threshold: float = float("inf"), direction: Literal[1, -1] = 1, patience: int = 5, min_steps: int = 0
    ) -> None:
        self.threshold = threshold
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
        if self.stopped:
            return False
        # returns True if should stop

        self.steps += 1
        if self.steps < self.min_steps:
            return False

        # check threshold crossed
        if self.threshold is not float("inf") and (value - self.threshold) * self.direction < 0:
            self.best = (
                min(self.best, value) if self.direction == -1 else max(self.best, value)
            )
            self.bad_steps = 0
            return False

        if self._is_improvement(value):
            self.best = value
            self.bad_steps = 0
        else:
            self.bad_steps += 1

        should_stop =  self.bad_steps > self.patience
        if should_stop:
            self.stopped = True
        return should_stop
