#import "../thesis_env.typ": *

#reset-acronym("DNM")
#reset-acronym("LDM")

= Conclusion
Sepsis remains the leading cause of in-hospital death globally, while early predictions are notoriously difficult due to its complex, heterogeneous pathophysiology. 
This work demonstrates that hybrid modeling, integrating biologically grounded dynamics with data-driven learning, can overcome limitations of both purely mechanistic and purely black-box approaches to sepsis prediction.

The starting point was the #acr("DNM") @osc1@osc2, a functional model of coupled oscillators, representing organ and immune cell populations that can describe both healthy and pathological states related to sepsis.
While theoretically promising, the #acr("DNM") had never been validated against real patient data.
This work closes that gap by developing the #acr("LDM"), a novel neural architecture that embeds the #acr("DNM") within a machine learning pipeline for online sepsis prediction.

The #acr("LDM") decomposes binary sepsis prediction into its underlying Sepsis-3 components: suspected infection and acute organ dysfunction.
Crucially, it represents patient organ system states as trajectories through the #acr("DNM")s parameter space, modulating the degree of frequency synchronization, serving as a proxy for acute organ failure.
This design enables the model to learn mappings from electronic health records to physiologically meaningful latent coordinates while preserving the structured inductive bias of the #acr("DNM").

Trained and evaluated retrospectively on real world patient data, the #acr("LDM") achieved an #acr("AUROC") of $aurocp$ and #acr("AUPRC") of $auprcp$, outperforming baseline models.
Beyond improved predictive accuracy, the #acr("LDM") provides interpretable intermediate outputs, in contrast to purely data-driven black-box models.
Qualitative analysis of patient trajectories through the latent space demonstrated clinically plausible patterns of deterioration, recovery, and stability, with latent coordinates systematically aligning with physiological progressions.


This proof-of-concept establishes hybrid physics-informed deep learning as a promising pathway for interpretable clinical decision support, offering an alternative to purely black-box models while maintaining competitive predictive performance.
Finally, this work addresses current limitations, such as external validation and the lack of systematic latent-space analyses, but also provides directions for future research in terms of model improvements and extensions.
