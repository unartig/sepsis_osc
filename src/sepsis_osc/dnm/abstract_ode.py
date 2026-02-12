from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import fields
from typing import TypeVar

import equinox as eqx
import jax.numpy as jnp
import msgpack
import msgpack_numpy as mnp
import numpy as np
from diffrax import (
    AbstractSolver,
    Event,
    NoProgressMeter,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    Solution,
    TqdmProgressMeter,
    diffeqsolve,
)
from jax import tree as jtree
from jaxtyping import Array, Bool, DTypeLike, Float, PRNGKeyArray
from typing_extensions import Self

from sepsis_osc.utils.utils import timing

mnp.patch()

ConfigT = TypeVar("ConfigT")
MetricT = TypeVar("MetricT")
StateT = TypeVar("StateT")


class ConfigArgBase(ABC, eqx.Module):
    """
    Base class for configuration arguments used in ODE systems.
    Subclasses should define the static parameters required by the
    differential equation's derivative function.
    """



class ConfigBase(ABC, eqx.Module):
    """
    Base class for experimental configurations.

    This class handles the mapping between human-readable experiment
    parameters and the JAX-compatible arguments used during integration.
    """

    @property
    @abstractmethod
    def as_args(self) -> ConfigArgBase:
        """
        Converts the high-level config into an argument PyTree for Diffrax.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def as_index(self) -> tuple[float, ...]:
        """
        Returns a unique numerical identifier for the config state.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def batch_as_index(*args, **kwargs) -> jnp.ndarray:
        """
        Returns indices for a batch of configurations.
        """
        raise NotImplementedError
        return jnp.empty(())


class TreeBase(eqx.Module):
    """
    A utility base class for Equinox PyTrees with common array operations.

    Provides convenient mapping methods to manipulate all leaves in a
    structured object (State, Metrics, etc.) simultaneously.
    """

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        """
        Returns the shape of every leaf in the PyTree.
        """
        return jtree.map(lambda x: x.shape if x is not None else None, self.__dict__)

    @property
    def last(self) -> StateT:
        """
        Slices the last element along the first dimension for all leaves.
        """
        return jtree.map(lambda x: x[-1], self)

    def to_jax(self) -> StateT:
        """
        Recursively converts all leaves to JAX arrays.
        """
        return jtree.map(lambda x: jnp.asarray(x) if x is not None else None, self)

    def reshape(self, shape: tuple[int, ...]) -> StateT:
        return jtree.map(lambda x: jnp.reshape(x, shape) if x is not None and ~jnp.isnan(x).any() else None, self)

    def astype(self, dtype: DTypeLike = jnp.float32) -> StateT:
        return jtree.map(lambda x: x.astype(dtype) if x is not None else None, self)

    def squeeze(self) -> StateT:
        return jtree.map(lambda x: x.squeeze() if x is not None else None, self)

    @classmethod
    def copy(cls) -> Self:
        return cls(**{f.name: getattr(cls, f.name) for f in fields(cls)})

    def remove_infs(self) -> "TreeBase":
        """
        Filters out samples containing 'inf' values across the entire tree.
        (Happens in incomplete diffrax solves)
        """

        def inf_mask(x: Float[Array, "*"]) -> Bool[Array, "*"] | None:
            if not hasattr(x, "shape"):
                return jnp.zeros((x.shape[0],), dtype=bool) if hasattr(x, "shape") else None
            axes = tuple(range(1, x.ndim))
            return jnp.isinf(x).any(axis=axes)

        masks = [inf_mask(leaf) for leaf in jtree.leaves(self) if hasattr(leaf, "shape") and leaf.ndim > 0]

        combined_mask = ~jnp.logical_or.reduce(jnp.asarray(masks))

        return jtree.map(lambda x: x[combined_mask] if hasattr(x, "shape") and x.ndim > 0 else x, self)


class StateBase(ABC, TreeBase):
    """
    Base class for representing the physical state of a system.
    """

    pass


class MetricBase(ABC, TreeBase):
    """
    Base class for recording and serializing simulation metrics.

    Includes built-in support for Msgpack serialization and partial updates
    via JAX-style 'at' syntax.
    """

    @abstractmethod
    def as_single(self) -> MetricT:
        raise NotImplementedError

    @classmethod
    def np_empty(cls, shape: tuple[int, ...], dtype: DTypeLike = np.float32) -> MetricT:
        initialized_fields = {f.name: np.zeros(shape, dtype=dtype) for f in fields(cls)}
        return cls(**initialized_fields)

    def serialise(self) -> bytes:
        return msgpack.packb({f.name: np.asarray(getattr(self, f.name)) for f in fields(self)}, default=mnp.encode)

    @classmethod
    def deserialise(cls, data_bytes: bytes) -> Self:
        unpacked = msgpack.unpackb(data_bytes, object_hook=mnp.decode)
        return cls(**unpacked)

    def insert_at(self, index: tuple[int, ...], other: MetricT) -> MetricT:
        """
        Updates the metrics at a specific index with another Metric object.
        """
        if not isinstance(other, type(self)):
            raise TypeError(f"Expected type {type(self)}, but got {type(other)}")

        def insert_leaf(self_leaf: Float[Array, "*"], other_leaf: Float[Array, "*"]) -> Float[Array, "*"] | None:
            if isinstance(self_leaf, np.ndarray):
                self_leaf[index] = other_leaf
                return None
            if isinstance(self_leaf, jnp.ndarray):
                return self_leaf.at[index].set(other_leaf)
            raise ValueError("Unexpected Leaf Type")
            return None

        return jtree.map(insert_leaf, self, other)


class ODEBase(ABC):
    """
    A generic framework for defining and solving differential equations.

    This class abstracts the Diffrax boilerplate, providing integrated support
    for PID step-size control, steady-state detection, and progress tracking.
    """

    def __init__(
        self,
        step_rtol: float = 1e-3,
        step_atol: float = 1e-6,
        *,
        full_save: bool = False,
        full_save_dtype: DTypeLike = jnp.float64,
        steady_state_check: bool = False,
        progress_bar: bool = True,
    ) -> None:
        deriv: Callable = self.system_deriv
        self.init_sampler = self.generate_init_sampler()
        self.term = ODETerm(deriv)
        self.save_method = (
            self.generate_full_save(deriv, dtype=full_save_dtype) if full_save else self.generate_metric_save(deriv)
        )
        self.steady_state_check = Event(cond_fn=self.generate_steady_state_check()) if steady_state_check else None
        self.pid_controller = PIDController(rtol=step_rtol, atol=step_atol)
        self.progress_bar = TqdmProgressMeter() if progress_bar else NoProgressMeter()

    @abstractmethod
    def generate_init_sampler(self) -> Callable:
        """
        Defines the initial condition sampling.
        """
        raise NotImplementedError

    @abstractmethod
    def system_deriv(self) -> Callable:
        """
        Defines the derivative function (dy/dt) of the ODE system.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_metric_save(self, deriv: Callable) -> Callable:
        """
        Defines the saving method using reduced metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_full_save(
        self, deriv: Callable, dtype: DTypeLike, *, save_y: bool = True, save_dy: bool = True
    ) -> Callable:
        """
        Defines the full saving method where the whole system state is saved.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_steady_state_check(self) -> Callable:
        """
        Defines the steady state check.
        """
        raise NotImplementedError

    @timing
    def integrate(
        self,
        config: ConfigBase,
        *,
        M: int,
        key: PRNGKeyArray,
        T_init: float,
        T_max: float,
        T_step: float,
        solver: AbstractSolver,
        ts: list | None = None,
    ) -> Solution:
        """
        Performs numerical integration using Diffrax.
        """
        saveat = (
            SaveAt(t0=True, ts=ts, fn=self.save_method)
            if ts is not None
            else SaveAt(t0=True, t1=True, fn=self.save_method)
        )
        return diffeqsolve(
            self.term,
            solver,
            y0=self.init_sampler(config, M, key),
            args=config.as_args,
            t0=T_init,
            t1=T_max,
            dt0=T_step,
            stepsize_controller=self.pid_controller,
            max_steps=int(1e6),
            saveat=saveat,
            progress_meter=self.progress_bar,
            event=self.steady_state_check,
            adjoint=RecursiveCheckpointAdjoint(checkpoints=1),
        )
