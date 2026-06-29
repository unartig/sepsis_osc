
#import "../thesis_env.typ": *
= Introduction
Sepsis is a life-threatening condition in which the human body's response to an infection turns against itself, causing a cascade of organ dysfunction that remains a leading cause of death in intensive care units worldwide.
With nearly 20% of annual worldwide deaths, sepsis poses an enormous clinical and economic burden @rudd2020global.
Despite decades of research and increasingly sophisticated clinical monitoring, early identification of patients at risk remains genuinely difficult due to its inherent complexity and multi-level pathogenicity.
The pathophysiology of sepsis is heterogeneous, its onset is gradual and not clearly defined, and the clinical signals that precede it are shared with many other conditions.

Recent work by Sawicki et al. @Sawicki2022DNM and Berner et al. @Berner2022Critical introduced the #acr("PNM"), a functional model capturing sepsis-related organ dysfunction through the lens of coupled oscillator dynamics, has been introduced.
In this model, the coordinated metabolic activity of organ and immune cells is represented as synchronization in a two-layer network of phase oscillators, and the breakdown of that coordination, as observed in sepsis, manifests as desynchronization and the formation of multifrequency clusters.
Two interpretable parameters govern this transition: a biological age parameter $beta$, summarizing comorbidities and baseline inflammatory state, and an interlayer coupling strength $sigma$, representing the interaction between organ tissue and the immune system.

The #acr("LDM") @backes2026, a deep learning pipeline, integrates the #acr("PNM") for online sepsis onset prediction.
It is a physics-informed deep learning architecture that is trained on #acr("EHR") from the #acr("MIMIC")-IV intensive care database.
Rather than treating sepsis prediction as a black-box classification problem, the #acr("LDM") projects patient trajectories into the two-dimensional parameter space of the #acr("PNM"), where the position and movement of a patient over time carry physiological meaning.
The model decomposes the Sepsis-3 criterion into separate infection and organ-dysfunction modules, and achieves competitive predictive performance while providing an interpretable trajectory-based representation of patient state.

While the original paper demonstrates the #acr("LDM")'s feasibility, several open questions remain.
The contribution of individual loss terms has not been systematically quantified.
The performance cost of the discrete #acr("PNM") lookup relative to differentiable alternatives is unknown.
And it is not clear whether the learned latent representation carries useful information or generalizes across patient populations.
This project addresses these gaps through a series of targeted analyses and experiments.
The central goals are: (i) to characterize what information the learned latent space actually encodes and whether it aligns with the intended physiological interpretation; (ii) to quantify the contribution of individual architectural and loss components through systematic ablation; (iii) to investigate whether the #acr("PNM") latent space and lookup mechanism confer any advantage over simpler alternatives; and (iv) to assess how well the model generalizes to an independent clinical dataset. 

The project is structured as follows:
In @sec:bg the necessary foundation is laid out by introducing the #acr("PNM") and #acr("LDM"), @sec:experiments describes various experiments that are performed for this project, with results presented in @sec:results and discussed in @sec:disc.
Closing remarks are taken in @sec:conc.

