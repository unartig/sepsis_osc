#import "../thesis_env.typ": *

= Conclusion <sec:conc>
This project systematically untangled the structural and predictive mechanisms of the #acr("LDM"), confirming its utility as an interpretable framework for online sepsis onset prediction.

By analyzing the latent structures, the $beta$ axis reliably recovered a proxy for chronic organ burden and biological age, aligning with renal and metabolic markers without any direct feature supervision and doing so consistently across 25 independent training runs and all model variants with directional surface structure.
The $sigma$ axis encodes a genuine but weaker and less stable signal for which the clinical interpretation as immune-organ coupling remains plausible but cannot be established from standard EHR features alone.

Through experimentation with ablations and model variations, it was demonstrated that the model's predictive performance is asymmetric, primarily driven by the infection module, while its clinical interpretability is anchored entirely by the #acr("SOFA") regression loss.
Though the performance differences can be attributed to unfavorable labeling strategies and do not invalidate the acute organ dysfunction prediction.

Further, the ablation studies have shown that $lambda_"sofa"$ is the singular term responsible for grounding the latent space in physiological meaning: removing it leaves Sepsis-3 prediction intact while completely destroying latent interpretability, confirming a clean dissociation between task performance and representational structure.
$lambda_"sep"$ is necessary for calibrating the joint sepsis output; without it, the two branches optimize independently and the combined prediction collapses in precision.
The remaining three terms, $lambda_"dec"$, $lambda_"spread"$, and $lambda_"boundary"$, are pure geometric regularizers that reduce the probability of degenerate solutions without affecting mean performance or representational quality.

Replacing the discrete #acr("PNM") grid lookup with any differentiable surface consistently improves acute organ dysfunction detection by 10 #acr("AUROC") points, while leaving Sepsis-3 discrimination unchanged.
The paired radial comparison isolates the cause: the softmax grid discretization attenuates the gradient signal flowing through the surface to the encoder, and this attenuation is the binding constraint on #acr("SOFA") detection performance.
This is a deliberate and coherent design choice, but it limits the model's ceiling on organ dysfunction detection tasks.

The model achieves competitive performance on the #acr("eICU") cohort in zero-shot transfer, without any dataset-specific fine-tuning.
More importantly, the feature alignment patterns and decoder reconstruction profiles transfer consistently across the two databases, demonstrating that the learned representations abstract physiological structure rather than #acr("MIMIC")-IV-specific artifacts.

Future work should focus on regularizing continuous differentiable surfaces to preserve this clinical mapping without imposing the operational costs of discrete lookup quantization.
Additionally, revising either the evaluation protocol for organ dysfunction or improve the prediction definition remain an open challenge.
