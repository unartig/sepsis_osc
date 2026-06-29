#import "../thesis_env.typ": *

= Discussion <sec:disc>
This chapter reiterates key findings of the experiments and discusses its implication on the understanding of the #acr("LDM") performance.
The empirical evaluation of the #acr("LDM") uncovers an architectural tension in physics-informed machine learning: the trade-off between raw predictive capacity and structural interpretability.
While deep learning models traditionally operate as unconstrained black boxes to maximize performance, the #acr("LDM") deliberately constrains its latent space using a biophysically grounded #acr("PNM").

The findings systematically map the consequences of this constraint across ablation, variation, and external validation regimes.
Each strain of experiments adds some piece to the puzzle, resulting in a deepened understanding of the #acr("LDM").

== Latent Space Interpretability
The theoretical expectation that $beta$ is encoding chronic physiological compromise and $sigma$ encoding acute organ-immune coupling is partially confirmed.
In the standard scenario, the $beta$ axis robustly encodes renal and metabolic burden, separates high from low #acr("SOFA") patients with large effect sizes, and does so consistently across all 25 splits and all scenarios with directional surface structure.
Critically, these associations emerge without any direct feature supervision, only the #acr("SOFA") regression loss anchors the latent coordinate to clinical meaning, and the features themselves are never directly optimized against $beta$ or $sigma$.
The model independently discovers a latent axis that resembles a proxy for cumulative organ burden and biological age, consistent with the intended #acr("PNM") interpretation.

The $sigma$ axis is more ambiguous.
It contributes to #acr("SOFA") separation in the standard scenario ($d = 0.73$) and its feature alignment is weaker and less stable across splits.
This suggests $sigma$ encodes a genuine but weak signal.
The underlying reason is structural, because standard clinical protocols collect no direct measurement capturing the physical coupling between parenchymal and immune cells through the basal lamina, so the encoder is forced to infer this axis from indirect, global surrogate markers.
The clinical interpretation of $sigma$ as immune-organ coupling therefore remains plausible but should be treated with caution as Pearson's $r$ values does not account for nonlinear relationships.

== Mechanistic Roles of Loss Terms
The ablation analysis reveals a clear hierarchy among the multi-objective loss terms.

The two load-bearing terms are $lambda_"sep"$ and $lambda_"sofa"$, and they serve complementary roles.
$lambda_"sep"$ calibrates the multiplicative interaction between the infection and organ branches for the joint sepsis prediction objective.
Removing it leaves each branch well-optimized on its own target but miscalibrated for their product, collapsing #acr("AUPRC") by more than half despite only a five-point #acr("AUROC") drop.
$lambda_"sofa"$, by contrast, is the sole anchor that ties latent coordinates to physiological meaning.
removing it leaves Sepsis-3 performance entirely intact (as expected, since the infection branch carries the prediction), but completely destroys the clinically interpretable spatial structure of the latent space.
Without it, subgroup separation along both axes collapses to near zero with enormous inter-split variance, and feature-axis alignment becomes inconsistent and arbitrary across runs.
Neither of these losses can be compensated by any combination of the remaining four terms.

The three geometric terms, $lambda_"spread"$, $lambda_"boundary"$, and $lambda_"dec"$ are necessary but purely regularizing.
They constrain which of many equivalent predictive solutions the model converges to, without contributing to the objective itself, prediction performance or latent organization.
Of these, $lambda_"dec"$ is the most useful in practice as it adds a decodable feature representation and modestly amplifies latent separation, at no cost to performance.
$lambda_"spread"$ and $lambda_"boundary"$ serve as purely geometric and spatial regularizers, and help the model to avoid latent collapse and degenerate latent structures.

== The Performance-Interpretability bottleneck
The variation experiments reveal an unexpected finding that by replacing the exact #acr("PNM") lookup with almost any differentiable alternative improves $Delta$#acr("SOFA") detection by 10+ #acr("AUROC") points.
The standard scenario's discrete softmax lookup attenuates the gradient flowing from the organ branch back to the encoder, and this bottleneck appears to be the limiting factor for #acr("SOFA") discrimination.

The paired radial comparison isolates the mechanism cleanly.
The continuous radial surface achieves $Delta$#acr("SOFA") #acr("AUROC") $80.06$, while the radial_modest_latent_lookup scenario is using the identical surface but accessed through the discrete grid and drops to $77.15$.
The surface geometry is held constant and only the lookup mechanism changes.
This confirms that the performance gap is caused by gradient attenuation through the discretization step, not by any property of the #acr("PNM") landscape itself.
The softmax interpolation over a fixed grid quantizes the encoder output before it reaches the organ branch, limiting the sharpness and reliability of the gradient signal flowing back through $tilde(s)^1$ to the encoder weights.

The lookup is a design choice that ties the latent space to the #acr("PNM") grid and enables the clinical interpretation of $(beta, sigma)$ coordinates.
Replacing it with a smooth function improves task performance but severs that connection.
Whether this trade-off is acceptable depends on the intended use of the model, for a prediction tool, the lookup is suboptimal; for an interpretable representation, it is desirable.

Taken as a pure sepsis predictor, the standard scenario is competitive, the #acr("AUROC") $84.11$ is matched or approached by every other scenario, and no variation meaningfully outperforms it on the primary task.
Taken as an organ dysfunction detector, it is clearly suboptimal, ranking last among model variation scenarios by a wide margin.
Taken as an interpretable latent representation, it is the only scenario that combines stable feature alignment, consistent subgroup separation on both axes, and a biologically grounded surface topology, the variations either sacrifice $sigma$ structure (linear, approx) or produce uninterpretable organizations across runs (radial).
Interestingly, a performance drop in one branch—either organ failure or infection—is almost entirely offset by the other.
Yet, when both branches perform well, the combined performance stagnation suggests an inherent upper bound to sepsis prediction on the #acr("MIMIC")-IV dataset.

The standard scenario is therefore best understood as design compromise, because it accepts reduced #acr("SOFA") detection performance in exchange for a latent space that is consistently organized, decodable, and interpretable in terms of the #acr("PNM")'s biophysical parameters.
If that compromise is worthwhile depends on whether the latent representation is itself a deliverable, if the goal is prediction alone, a differentiable surface with the same supervision losses would dominate.

== Labeling Constraints on Branch Performance
A recurring observation throughout the results is the apparent weakness of the organ dysfunction branch.
Apparently, it contributes little to sepsis discrimination in the standard scenario, and its direct $Delta$#acr("SOFA") #acr("AUROC") of $65.82$ appears low relative to the infection branch.
This observation requires careful contextualization.

The #acr("YAIB") labeling procedure spreads the sepsis onset label $plus.minus 6$ hours around the documented #acr("SOFA") increase, but the acute organ dysfunction indicator $A_t$ is instantaneous, it fires only at the precise timestep where the #acr("SOFA") increase occurs.
Because the label definitions are mismatched, using the ground-truth $A_t$ directly as a predictor for the temporally-spread sepsis label yields #acr("AUROC") only $54.5$.
The organ branch modestly outperforms this ground-truth oracle (#acr("AUROC") $58.41$), meaning the causal smoothing window recovers some of the temporal misalignment.
The branch is not failing at organ dysfunction detection, it is limited by an incompatibility between the acute nature of its signal and the spread labeling strategy used for the primary outcome.

The infection branch, by contrast, benefits from the temporal spread, the smoothed infection surrogate used during training is explicitly designed to ramp up over the 48 hours before documented onset, creating a gradual signal that aligns naturally better with the sepsis label window.
This structural asymmetry, not a fundamental difference in model capacity, explains why the infection branch dominates sepsis performance across all scenarios.
The practical implication is that the organ branch is undervalued as a clinical signal by the current evaluation protocol.

== External Validation
The external validation on the #acr("eICU") has produced mixed results.
The prediction performance results are competitive and once again show its applicability for online sepsis onset prediction, though, unlike for the #acr("MIMIC") database, this time the #acr("LDM") is not able to significantly outperform the #acr("YAIB") benchmarks @yaib.
Crucially, the model demonstrates a robust capacity for cross-cohort generalization, with stable decoder reconstruction patterns and consistent latent feature alignments when transferred directly to the #acr("eICU") environment.
For example, the decoder's performance measured via Pearson's $r$ metrics remain highly synchronized across both environments, with the #acr("eICU") validation cohort exhibiting even tighter, highly robust tracking of specific focal markers, such as urine output.
This robust translation between different medical data sources suggests that the #acr("LDM") successfully abstracts fundamental, high-level physiological representations.

It must be noted, that a comparative analysis of this cross-cohort translation remains constrained by a limitation in current literature since there are presently no baseline values for cross-cohort transferability.
Consequently, while these results empirically demonstrate that the #acr("LDM") extracts stable, generalizable, and physiological principles across distinct intensive care networks, developing standardized benchmarks remains future research.
