from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Callable, TypeVar
import msgpack
import msgpack_numpy as mnp

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from diffrax import (
    Event,
    NoProgressMeter,
    ODETerm,
    PIDController,
    SaveAt,
    TqdmProgressMeter,
    Tsit5,
    RecursiveCheckpointAdjoint,
    diffeqsolve,
)
from jax import tree as jtree
from numpy.typing import DTypeLike


mnp.patch()

ConfigT = TypeVar("ConfigT")
MetricT = TypeVar("MetricT")
StateT = TypeVar("StateT")


class ConfigArgBase(ABC, eqx.Module):
    pass


class ConfigBase(ABC, eqx.Module):
    @property
    @abstractmethod
    def as_args(self) -> ConfigArgBase:
        raise NotImplementedError

    @property
    @abstractmethod
    def as_index(self) -> tuple[float, ...]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def batch_as_index(*args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
        return jnp.empty(())


class TreeBase(eqx.Module):
    @property
    def shape(self):
        return jtree.map(lambda x: x.shape if x is not None else None, self.__dict__)

    @property
    def last(self) -> StateT:
        return jtree.map(lambda x: x[-1], self)

    def to_jax(self) -> StateT:
        return jtree.map(lambda x: jnp.asarray(x) if x is not None else None, self)

    def reshape(self, shape: tuple[int, ...]) -> StateT:
        return jtree.map(lambda x: jnp.reshape(x, shape) if x is not None and ~jnp.isnan(x).any() else None, self)

    def astype(self, dtype: jnp.dtype = jnp.float32) -> StateT:
        return jtree.map(lambda x: x.astype(dtype) if x is not None else None, self)

    def squeeze(self) -> StateT:
        return jtree.map(lambda x: x.squeeze() if x is not None else None, self)

    @classmethod
    def copy(cls: type[StateT]) -> StateT:
        return cls(**{f.name: getattr(cls, f.name) for f in fields(cls)})

    def remove_infs(self) -> "TreeBase":
        def inf_mask(x):
            if not hasattr(x, "shape"):
                return jnp.zeros((x.shape[0],), dtype=bool) if hasattr(x, "shape") else None
            axes = tuple(range(1, x.ndim))
            return jnp.isinf(x).any(axis=axes)

        masks = []
        for leaf in jtree.leaves(self):
            if hasattr(leaf, "shape") and leaf.ndim > 0:
                masks.append(inf_mask(leaf))

        combined_mask = ~jnp.logical_or.reduce(jnp.asarray(masks))

        return jtree.map(lambda x: x[combined_mask] if hasattr(x, "shape") and x.ndim > 0 else x, self)


class StateBase(ABC, TreeBase):
    pass


class MetricBase(ABC, TreeBase):
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
    def deserialise(cls: type[MetricT], data_bytes: bytes) -> MetricT:
        unpacked = msgpack.unpackb(data_bytes, object_hook=mnp.decode)
        return cls(**unpacked)

    def insert_at(self, index: tuple[int, ...], other: MetricT) -> MetricT:
        if not isinstance(other, type(self)):
            raise TypeError(f"Expected type {type(self)}, but got {type(other)}")

        def insert_leaf(self_leaf, other_leaf):
            if isinstance(self_leaf, np.ndarray):
                self_leaf[index] = other_leaf
            elif isinstance(self_leaf, jnp.ndarray):
                return self_leaf.at[index].set(other_leaf)
            else:
                raise ValueError("Unexpected Leaf Type")

        return jtree.map(insert_leaf, self, other)


class ODEBase(ABC):
    def __init__(
        self,
        full_save: bool = False,
        steady_state_check: bool = False,
        progress_bar: bool = True,
        step_rtol: float = 1e-3,
        step_atol: float = 1e-6,
    ):
        deriv = self.system_deriv
        self.init_sampler = self.generate_init_sampler()
        self.term = ODETerm(deriv)
        self.save_method = self.generate_full_save(deriv) if full_save else self.generate_metric_save(deriv)
        self.steady_state_check = Event(cond_fn=self.generate_steady_state_check()) if steady_state_check else None
        self.pid_controller = PIDController(rtol=step_rtol, atol=step_atol)
        self.progress_bar = TqdmProgressMeter() if progress_bar else NoProgressMeter()

    def generate_init_sampler(self) -> Callable:
        raise NotImplementedError

    def system_deriv(self) -> Callable:
        raise NotImplementedError

    def generate_metric_save(self) -> Callable:
        raise NotImplementedError

    def generate_full_save(self) -> Callable:
        raise NotImplementedError

    def generate_steady_state_check(self) -> Callable:
        raise NotImplementedError

    # @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=10)
    def integrate(
        self,
        config,
        *,
        M,
        key,
        T_init,
        T_max,
        T_step,
        solver,
    ):
        result = diffeqsolve(
            self.term,
            solver,
            y0=self.init_sampler(config, M, key),
            args=config.as_args,
            t0=T_init,
            t1=T_max,
            dt0=T_step,
            stepsize_controller=self.pid_controller,
            max_steps=int(1e6),
            saveat=SaveAt(t0=True, ts=jnp.arange(T_init, T_max, T_step), fn=self.save_method),
            progress_meter=self.progress_bar,
            event=self.steady_state_check,
            adjoint=RecursiveCheckpointAdjoint(checkpoints=1)
        )
        return result
