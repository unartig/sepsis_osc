# Replica Implementation of Adaptively Coupled Phase Oscillators for Sepsis Modeling

In this repository I will gather and share my custom/replica implementations of the Sepsis models described in [Berner et al.](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2021.730385/full) and [Sawicki et al.](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2022.904480/full).
The current implementation of the simulation is using Jax and [diffrax](https://github.com/patrick-kidger/diffrax) to (GPU) accelerate the numerical integration calculations (files simulation.py and run_simulation.py) but can also be run on the CPU.
The files viz_*.py can be used to visualize ensemble systems or single instances of the initial value problems, larger visualization (like the parameter space) need the SystemMetrics to be saved via the storage interface beforehand.

!!! The whole repository is WIP !!!
