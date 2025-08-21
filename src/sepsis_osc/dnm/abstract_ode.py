from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Callable, TypeVar

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
    diffeqsolve,
)
from jax import tree as jtree
from numpy.typing import DTypeLike

from sepsis_osc.utils.jax_config import setup_jax

setup_jax()

ConfigT = TypeVar("ConfigT")
MetricT = TypeVar("MetricT")
StateT = TypeVar("StateT")


class ConfigArgBase(ABC, eqx.Module):
    pass

class ConfigBase(ABC, eqx.Module):
    @property
    @abstractmethod
    def as_args(self):
        NotImplementedError

    @property
    @abstractmethod
    def as_index(self):
        NotImplementedError

    @staticmethod
    @abstractmethod
    def batch_as_index():
        NotImplementedError


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
        return jtree.map(lambda x: jnp.reshape(x, shape) if x is not None else None, self)

    def astype(self, dtype: jnp.dtype = jnp.float32) -> StateT:
        return jtree.map(lambda x: x.astype(dtype) if x is not None else None, self)

    def squeeze(self) -> StateT:
        return jtree.map(lambda x: x.squeeze() if x is not None else None, self)

    def copy(self) -> StateT:
        return StateBase(**{f.name: getattr(self, f.name) for f in fields(self)})


class StateBase(ABC, TreeBase):
    pass


class MetricsBase(ABC, TreeBase):
    @staticmethod
    def np_empty(shape: tuple[int, ...], dtype: DTypeLike = np.float32) -> MetricT:
        initialized_fields = {f.name: np.zeros(shape, dtype=dtype) for f in fields(MetricsBase)}
        return MetricsBase(**initialized_fields)


class ODEBase(ABC):
    def __init__(self, full_save: bool = False):
        deriv = self.system_deriv
        self.term = ODETerm(deriv)
        self.save_method = self.generate_full_save(deriv) if full_save else self.generate_metric_save(deriv)
        self.steady_state_check = None

    def generate_init_conditions(self, config, M, key) -> StateT:
        raise NotImplementedError

    def system_deriv(self) -> Callable:
        raise NotImplementedError

    def generate_metric_save(self) -> Callable:
        raise NotImplementedError

    def generate_full_save(self) -> Callable:
        raise NotImplementedError

    def generate_steady_state_check(self) -> Callable:
        raise NotImplementedError

    def integrate(
        self,
        config,
        M,
        key,
        T_init,
        T_max,
        T_step,
        solver=Tsit5(),
        progress_bar=True,
        step_rtol=1e-6,
        step_atol=1e-3,
    ):
        result = diffeqsolve(
            self.term,
            solver,
            y0=self.generate_init_conditions(config, M, key),
            args=config.as_args,
            t0=T_init,
            t1=T_max,
            dt0=T_step,
            stepsize_controller=PIDController(rtol=step_rtol, atol=step_atol),
            max_steps=int(1e6),
            saveat=SaveAt(t0=True, ts=jnp.arange(T_init, T_max, T_step), fn=self.save_method),
            progress_meter=TqdmProgressMeter() if progress_bar else NoProgressMeter(),
            event=Event(cond_fn=self.generate_steady_state_check()) if self.steady_state_check is not None else None,
        )
        return result
