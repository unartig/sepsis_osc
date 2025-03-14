# Replica Implementation of Adaptively Coupled Phase Oscillators for Sepsis Modeling

In this repository I will gather and share my custom/replica implementations of the Sepsis models described in [Berner et al.](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2021.730385/full) and [Sawicki et al.](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2022.904480/full).
The current implementation of the simulation is using Jax to (GPU) accelerate the numerical integration calculations (file simulation.py) but can also be run on the CPU.
In the file viz.py an early version of the visualizations can be found.

The whole repository is WIP. As of now it is full of errors and does **not** reproduce the findings of the resources.

## Open Questions

#### Regarding the visualisations

1. At what time-step are the "snapshots" of the phases $\phi^\mu$ taken?
2. Is $\langle \dot{\phi}^\mu_{100/300}\rangle$ the mean velocity between the time-steps 100/300 or 1100/1300?
3. Are the cytokine matrices $\kappa^\mu$ also sorted? By the same key as the phases?
