#import "../thesis_env.typ": *

= Introduction
Sepsis, a dysregulated response to an infection, accounts for nearly 20% of all deaths worldwide—approximately 11 million deaths annually @rudd2020global.
Despite its enormous clinical and economic burden, sepsis remains notoriously difficult to predict, yet early recognition is critical, each hour of delayed treatment increases mortality risk @seymour2017time.
The difficulty in sepsis prediction stems its inherent complexity.
It is not a single disease but a heterogeneous syndrome involving infection, immune dysregulation, and multi-organ dysfunction across multiple biological scales.
Currently, automated alert systems suffer from limited clinical adoption and further research for improving performance is necessary @Alshaeba2025Effect.

Methodologically, approaches to sepsis prediction fall into two categories, each with fundamental limitations.
Data-driven methods using machine learning on electronic health records show promising performance but function as black boxes, offering little mechanistic insight into why a patient is at risk @Bomrah2024Review@Moor2021Review.
Mechanistic models explicitly encode biological processes, providing interpretability but require detailed parameterization and high-resolution measurements rarely available in clinical practice @Relouw2024@Cockrell2022Model.

This work proposes a hybrid approach that combines the strengths of both paradigms.
The foundation is the #acr("DNM") @osc1@osc2, a functional model that describes sepsis-related organ dysfunction through coupled oscillator networks.
The #acr("DNM") represents immune and organ systems as interacting layers of phase oscillators, where synchronization patterns correspond to physiological states.
Two key parameters, a biological age and the strength of organ-immune linkage, govern transitions between healthy synchronized states and pathological desynchronized regimes.

While the #acr("DNM") demonstrates rich theoretical behavior aligned with clinical understanding, it has never been validated against real patient data.
This thesis addresses that gap by developing the #acr("LDM"), a novel neural architecture that embeds the #acr("DNM") within a machine learning pipeline for online sepsis prediction.
The #acr("LDM") learns to map electronic health record time series to interpretable components aligned with the technical sepsis definition.
Rather than treating sepsis as a black-box classification problem, the model represents patient organ states as trajectories through the #acr("DNM")s parameter space, where desynchronization serves as a continuous proxy for acute organ failure.
This design provides both competitive predictive performance, evaluated on real world data and compared to existing baseline methods, and interpretable intermediate outputs that align with clinical reasoning.

The thesis is structured as follows:
@sec:sepsis provides medical background on sepsis, from cellular immune response and cytokine storms to the Sepsis-3 clinical definition, and closing by establishing the clinical need for better prediction systems.
@sec:dnm introduces the #acr("DNM"), starting with Kuramoto oscillators and building up complexity to the full #acr("DNM") architecture.
Simulation results demonstrate how the parameter space captures transitions between healthy and pathological states.
@sec:sota reviews the state of the art in sepsis prediction, contrasting mechanistic and data-driven approaches and identifying the gap that motivates hybrid modeling.
@sec:ldm presents the #acr("LDM") methodology in detail, task formalization, architecture design, training objectives, and the differentiable #acr("DNM") embedding strategy.
@sec:experiment describes the complete experimental setup (data-basis, cohort definition, implementation) and presents quantitative results alongside qualitative analysis of learned patient trajectories.
@sec:disc summarizes findings, discusses limitations, and proposes directions for future research.
