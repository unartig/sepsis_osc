#import "../thesis_env.typ": *

// + *Analysis of the latent positioning*
//   - What information does the latent position carry?
//     - What input features are the largest contributors to the latent positioning?
//   - How well are different triggers separated in latent space?
//   - How does that vary between the different splits?
//   - Does the assumed interpretation of $beta$ (biological age) and $sigma$ (immune-organ connection strength) hold?
// + *Analysis of the Decoder performance*
//   - How well are the input features reconstructed?
//   - What features does the decoder focus on?
// + *Performance on an external dataset*
//   - #acr("eICU")
// + *Ablation studies*
//   - How does the removal of different loss-parts influence the performance.
//     - Removing $lambda_"sep"$
//     - Removing $lambda_"boundary"$
//     - Removing $lambda_"dec"$
//     - Removing $lambda_"spread"$
//     - Removing the infection branch
//     - Removing the Organ dysfunction branch
//   - What is the contribution of the #acr("PNM") latent space and lookup mechanism?
//     - How does a learnable MLP perform?
//     - How does a differentiable surrogate model of the #acr("PNM") latent space perform?
//     - How does a differentiable analytical approximation of the #acr("PNM") perform?
//     - How do other analytical functions perform?
//       - Linear step-like
//       - Linear smooth
//       - Radial
//     - How does a quantization of the Radial analytical function perform?

= Experiments <sec:experiments>
This chapter describes the experimental setup and analysis methodology used in this project.
It is organized into two parts.
The first defines the model variants, referred to as _scenarios_, that are trained and evaluated.
The second describes the analysis methods applied to those scenarios, including the mathematical formulation and motivation of each.

If not stated otherwise, all scenarios share the same data, preprocessing, cross-validation splits, hyperparameters, and evaluation protocol as the baseline #acr("LDM") described in the background chapter adopted from @backes2026.

There are three types of experiments performed: *ablations*, *model variations* and *external validation*.
Ablations remove individual loss terms to test which components of the training objective are necessary for the model to learn an interpretable, physiologically structured latent space.
Model variations replace the #acr("PNM") lookup with alternative functions, ranging from learnable approximations to simple closed form surfaces, to test how much of the #acr("LDM")'s performance and interpretability depends on the specific structure of the #acr("PNM")'s $(beta, sigma)$ landscape, versus any sufficiently smooth, structured mapping.
Additionally this tests the latent-lookup methodology, compared to other mechanisms of gradient provision.
External validations retrain the original model on a different dataset, providing insights in the generalizability of the method.

== Standard (baseline)
The full #acl("LDM") as described in the original paper, trained with all six loss components.
It serves as the reference point for all comparisons and is referred to as the 'standard' scenario.
The latent space, computed directly from the #acr("PNM") simulations, is depicted in @fig:latent-standard.
See @backes2026 for more details on the numerical integration of the #acr("PNM").
For the purpose of this project, the #acr("LDM") standard scenario has been retrained on the #acr("MIMIC")-IV dataset for 5 iterations of 5-fold cross-validation.

#figure(
  image("../images/project/ablations/standard/svg_latent_space.svg"),
  caption: flex-caption(
    long: [Default parameter space or latent space of the #acs("PNM"), adopted from @Berner2022Critical, for simulation details see supplemental materials of @backes2026.],
    short: [Default parameter space of the #acs("PNM")],
  ),
) <fig:latent-standard>

== Ablations

Each ablation removes a single loss term, by setting the respective loss weight to $0.0$, while keeping all others and their weights unchanged.
Since the loss balance changes, these variants require retraining across all 25 splits.
The latent space topology of the ablations are identical to the 'standard' scenario.

The five ablations correspond to four non-primary loss terms and the primary sepsis loss.
The primary sepsis loss $lambda_"sep"$ is included as a baseline ablation because it directly supervises the training target; the remaining four test each auxiliary objective independently.
Branch-level ablations (removing the infection or organ branch entirely) are not performed here, as they change the fundamental model topology rather than the training objective.
The infection ablation is not performed as it does not directly act on the latent space, the primary target of this project is about.

*No sepsis loss ($lambda_"sep"$)*\
The primary binary cross-entropy loss for sepsis onset prediction is removed.
The model is trained only on the auxiliary objectives: infection supervision, #acr("SOFA") regression, decoder reconstruction, spreading, and boundary. This scenario is called 'standard_no_sep'.

*No SOFA loss ($lambda_"sofa"$)*\
The mean-squared error supervision on #acr("SOFA") score magnitude is removed, and is called 'standard_no_sofa'.
This directly breaks the connection between latent position and the #acr("PNM") synchronization landscape during training, since the #acr("SOFA") loss is what guides $(beta, sigma)$ toward physiologically meaningful regions of the grid.

*No decoder loss ($lambda_"dec"$)*\
The auxiliary reconstruction loss is removed, and is referred to as 'standard_no_recon'.
The decoder network may still be present in the architecture but receives no training signal.

*No spreading loss ($lambda_"spread"$)*\
The generalized variance term that discourages latent collapse is removed.
This scenario is called 'standard_no_spread'.

*No boundary loss ($lambda_"boundary"$)*\
The soft penalty that discourages latent predictions near the edges of the parameter grid is removed, and is called 'standard_no_boundary'.

== Model Variations
The variants below replace the precomputed #acr("PNM") lookup-and-interpolation mechanism with alternative functions that map latent coordinates $(z_1, z_2)$ to a #acr("SOFA") proxy.
All other components of the architecture and training procedure remain identical to the standard scenario, and each variation requires retraining across all 25 splits.
For consistency, the new latent dimensions are also labeled $(beta, sigma)$, though they do not necessarily share the same clinical interpretation.

The variations span a deliberate spectrum from maximum fidelity to the #acr("PNM") (surrogate) to complete decoupling #acr("MLP"), with three closed-form surfaces in between.
This ordering allows the contribution of the #acr("PNM") structure, versus any smooth monotonic gradient, to be isolated progressively.

*Learnable #acr("MLP")*\
A small randomly-initialized multi-layer perceptron replaces the #acr("PNM") lookup and is trained jointly with the rest of the model.
This tests the upper bound of performance when the surface is fully unconstrained and jointly optimized.
This scenario is called 'mlp' and is already reported in the original paper. #footnote[For the purpose of this project, the #acr("LDM")-mlp scenario has been retrained from scratch.].
An exemplary latent space can be seen in @fig:latent-mlp, though here the new latent dimensions do not share the same clinical interpretation.
The model architecture is shown in the appendix in @fig:arch-mlp.

#figure(
  image("../images/project/variations/mlp/svg_latent_space.svg"),
  caption: flex-caption(
    short: [Exemplary parameter space of the #acs("MLP") scenario.],
    long: [Exemplary parameter space of the #acr("MLP") scenario. Because the #acr("MLP") parameters are trainable, this surface is subject to change over the course of training and varies across splits.],
  ),
) <fig:latent-mlp>

*Differentiable #acr("PNM") surrogate*\
A small neural network is first trained to approximate the $s^1(beta, sigma)$ surface on the precomputed grid, then frozen and used as a fixed replacement for the grid lookup.
This provides a smooth and fully differentiable approximation of the same functional form as the #acr("PNM") without requiring the discrete interpolation step of the latent lookup.
This tests whether the #acr("PNM") surface geometry itself is beneficial, independent of the gradient attenuation caused by discrete lookup.
This latent space of this 'surrogate' scenario, can be seen in @fig:latent-surrogate, where the dimensions share interpretation with the standard scenario.

#figure(
  image("../images/project/variations/surrogate/svg_latent_space.svg"),
  caption: flex-caption(
    short: [Latent space of the surrogate scenario.],
    long: [The latent space of the surrogate scenario. A small #acr("MLP") learns to reproduce the #acr("PNM") parameter space and is frozen during training, providing smooth differential gradients.],
  ),
) <fig:latent-surrogate>

*Analytical #acr("PNM") approximation*\
A closed-form expression that captures gross qualitative behavior of the $s^1$ surface, specifically the smooth transition from low to high desynchronization as $beta$ increases, without reproducing its fine-grained structure.
This scenario tests whether a qualitatively similar but analytically exact approximation of the #acr("PNM") surface preserves performance and interpretability.

The expression was obtained via symbolic regression using PySR @pysr, which searches over a space of algebraic expression to minimize the #acr("MSE") between the candidate expression and the precomputed $s^(beta, sigma)$ grid from the #acr("PNM").
The specific functional form is:
$
  "bump"(x) = x exp(-x)
$
$ A(beta, sigma) = "bump"[sin(sin(sigma / 0.60103625 - (1.1470096 beta)/pi)) + 0.4785785] $
$ B(beta, sigma) = (sin(beta / (pi (0.50825256 - A(beta, sigma)))) - 0.10480727)^4 $

The final expression is:

$ tilde(s)^1(beta, sigma) = sin(sin^2(B) + 0.11145829 sigma) $ <eq:approx>
and the resulting latent space can be seen in @fig:latent-approx.
The bump function creates a soft peak that avoids saturation at large $x$, and was given as an predefined operator in the search space, improving the symbolic regression convergence speed.

As this approximation provides analytical gradients, no lookup mechanism is needed.
This scenario is referred to as 'approx'.

#figure(
  image("../images/project/variations/approx/svg_latent_space.svg"),
  caption: flex-caption(
    short: [Latent space of the approx scenario.],
    long: [The latent space of the approx scenario. Here the #acr("PNM") parameter space approximated via a closed form function.],
  ),
) <fig:latent-approx>

*Linear smooth*\
A sigmoid ramp along the $beta$ axis, providing a monotonic but featureless gradient with no interaction between $beta$ and $sigma$, and is called 'linear_smooth'.
The function is implemented in its closed form to provide direct gradients:
$
  "linear"(beta, sigma) = "sigmoid"(k (beta - beta_"mid"))
$ <eq:linear>
where the parameter $k$ controls the steepness of the gradient, and $beta_"mid" = (0.7 - 0.4)/2 = 0.55$ is the midpoint of the $beta$ dimension of the grid.
For the smooth linear transition, a value of $k=25$ is chosen, the corresponding latent space is shown in @fig:latent-lin-smooth.
As this is a closed form, no lookup mechanism is needed.

#figure(
  image("../images/project/variations/linear_smooth/svg_latent_space.svg"),
  caption: flex-caption(
    short: [Latent space of the linear_smooth scenario.],
    long: [The latent space of the linear_smooth scenario.],
  ),
) <fig:latent-lin-smooth>

*Linear step-like*\
Similar to the 'linear_smooth' scenario, a but with a sharp transition along the $beta$ axis, and is called 'linear_step'.
For that a value of $k=100$ is used, see @eq:linear.
Both of the 'linear' tests are designed to provide latent spaces with structures as simple as possible, to see whether more involved and structurally complicated like the #acr("PNM")-space hinder the learning.
The gradient differences test whether sharp phase transitions influence the performance.

#figure(
  image("../images/project/variations/linear_step/svg_latent_space.svg"),
  caption: flex-caption(
    short: [Latent space of the linear_step scenario.],
    long: [The latent space of the linear_step scenario.],
  ),
) <fig:latent-lin-step>

*Radial*\
A radially symmetric function centered in the parameter space, where distance from the center encodes desynchronization, and is called 'radial_modest'.
Unlike all other surfaces, this has no directional structure along $beta$ or $sigma$ individually.
This tests the case where neither axis has individual directional meaning, to assess whether the structural assignment of $beta$ and $sigma$ roles matters.
The closed form is given by:
$
  beta_"symm"(beta) = (beta - beta_min) / (beta_max - beta_min)\
  sigma_"symm"(sigma) = (sigma - sigma_min) / (sigma_max - sigma_min)\
  r = sqrt((beta_"symm"(beta) - 0.5)^2 + (sigma_"symm"(sigma) - 0.5)^2)\
  "radial"(beta,sigma) = "sigmoid"(k (0.2 - r))\
$
where the parameter $k$ controls the steepness of the gradient, and $beta_"min/max"$ and $sigma_"min/max"$ are the boundaries of the space.
Here a value of $k=25$ is chosen.
The resulting latent space can be seen in @fig:latent-rad, and because this scenario is implemented as a closed form, no lookup mechanism is need.

#figure(
  image("../images/project/variations/radial_modest/svg_latent_space.svg"),
  caption: flex-caption(
    short: [Latent space of the radial_modest scenario.],
    long: [The latent space of the radial scenario, a radially symmetric surface with no directional structure along either axis.],
  ),
) <fig:latent-rad>

*Quantized radial*\
This variant evaluates model behavior using the same mathematically continuous radial surface defined above, but accesses it through the discrete softmax interpolation mechanism used by the standard #acr("PNM") lookup grid.
By keeping the underlying surface topology identical to the continuous radial model, this scenario isolates the operational impact of grid quantization and discrete lookup routines.
Any performance drop or gradient attenuation observed here can be confidently attributed to the discretization process itself rather than the landscape geometry.
This scenario is termed 'radial_modest_latent_lookup'.

== External validation
To assess how well the #acr("LDM") generalizes beyond its training distribution, external validation is performed on the #acr("eICU") Collaborative Research Database @pollard2018eicu, a large multi-centre #acr("ICU") database covering over 200 US hospitals.
This contrasts with #acr("MIMIC")-IV, which is drawn from a single academic medical centre (Beth Israel Deaconess Medical Center, Boston).
The two cohorts differ meaningfully in size, patient demographics, and data collection protocols, key characteristics are summarized in @tab:mimic and @tab:eicu, making the eICU a non-trivial test of cross-cohort robustness.

Using the #acr("YAIB") framework @yaib, a compatible cohort and replication task is constructed that mirrors the Sepsis-3 labeling strategy and feature definitions used in the #acr("MIMIC")-IV experiments.
The shared feature set (with per-feature missingness rates for both cohorts) is listed in @a:feat.

Performance is evaluated under three paradigms:
- *eICU in-domain*: The #acr("LDM") is retrained from scratch on the #acr("eICU") cohort across the same 25 cross-validation splits, establishing an empirical performance ceiling for within-database learning on this population.
- *Zero-shot MIMIC $->$ eICU*: The #acr("MIMIC")-IV-trained model is applied directly to the #acr("eICU") cohort without any fine-tuning or adaptation.
- *Zero-shot eICU $->$ MIMIC*: The #acr("eICU")-trained model is applied directly to the #acr("MIMIC")-IV cohort, providing the reverse transfer direction.

Both zero-shot directions are included because transfer performance is not symmetric: a #acr("MIMIC")-trained model may generalize to #acr("eICU") differently than an #acr("eICU")-trained model generalizes to #acr("MIMIC"), due to differences in cohort size, class balance, and feature patterns.
For each paradigm, the full suite of analyse, predictive performance, latent feature alignment, subgroup separation, and decoder reconstruction quality is reported, allowing the question of generalization to be answered at the level of latent representation as well as prediction score.

== External validation
To evaluate the generalization ability of the original #acr("LDM") (trained on #acr("MIMIC")-IV), external validation on the #acr("eICU") Collaborative Research Database @pollard2018eicu is performed.

Using the #acr("YAIB") framework (a standardized cohort extraction and evaluation platform for #acr("ICU") prediction tasks), a compatible cohort and replication task that mirrors the Sepsis-3 labeling and feature definitions from the #acr("MIMIC")-IV experiments is constructed.
The shared list of these features, with missingness values, is available in @a:feat, while the cohort characteristics for both the #acr("MIMIC") and #acr("eICU") datasets are contrasted in @tab:mimic and @tab:eicu, respectively.

The performance is evaluated under two paradigms:
- *In-domain baseline*: Training the #acr("LDM") from scratch on #acr("eICU") to establish an empirical performance upper bound.
- *Zero-shot transfer*: Directly applying the #acr("MIMIC")-IV-trained model to the #acr("eICU") cohort, and vice-versa.

== Analysis methods
The following analyses are defined independently of the specific model variants they are applied to.
Note that not every analysis method is used for every scenario, as this would go beyond the scope.

Where analyses of a single split are of interest a _representative_ split is chosen as the closest split to the mean in terms of validation performance over all splits.

=== Predictive performance
The primary quantitative outcome for all variants is predictive performance, reported as #acr("AUROC") and #acr("AUPRC") for sepsis onset prediction, averaged over the 25 cross-validation splits with standard deviation.

To understand what each component of the model contributes to the final prediction, additionally an inference-time decomposition into individual module contributions is performed.
The sepsis-3 prediction is the product of the infection probability and the organ dysfunction risk.
By setting one factor to $1.0$ at inference time, it can be estimated what the model would predict using the remaining branch alone:
- *Infection branch only*: the organ dysfunction term is fixed to $1.0$, so sepsis risk is driven solely by $p_theta (I_t | bold(mu)^i_(1:t))$.
- *Organ dysfunction branch only*: the infection term is fixed to $1.0$, so sepsis risk is driven solely by $p_theta (A^i_t | bold(mu)^i_(1:t))$.

These approximations do not require retraining and are therefore available for
all scenarios. They provide a lightweight estimate of how much each branch
contributes to predictive performance.

Performance changes compared to #acr("YAIB") baseline models are analyzed, for the standard scenarios of both datasets (#acr("MIMIC")-IV and #acr("eICU")).
To test for significant changes in terms of prediction performance the two-side Welch's $t$-test @welch1947 with Holm-Bonferroni correction @Holm1979 across all #acr("YAIB") comparisons is used.
Significance is reached at a level of $p<0.05$.

=== Feature alignment
To understand what information the latent coordinates carry, it is measured how strongly each input feature is linearly associated with the predicted $beta_t$ and $sigma_t$ values. Specifically, for each input feature $x_f$ and each latent coordinate $z in {beta, sigma}$, the Pearson correlation coefficient $r$:
$
  r(x_f, z) = frac(
    sum_t (x_f^t - overline(x)_f)(z^t - overline(z)),
    sqrt(sum_t (x_f^t - overline(x)_f)^2) dot sqrt(sum_t (z^t - overline(z))^2)
  )
$ <eq:pearson>
where the sum runs over all valid timesteps across all patients in the test set.
Pearson $r$ is scale-invariant and does not require features to share a common unit or distribution, making it appropriate for a heterogeneous set of clinical measurements.
The measure captures linear associations; a near-zero $r$ does not rule out a nonlinear relationship between a feature and the latent coordinate.

The theoretical interpretation of the latent space assigns $beta$ to biological age and comorbidity burden, and $sigma$ to the strength of organ-immune coupling.
If the learned mapping recovers these meanings, $beta$ should correlate most strongly with slowly varying, chronic markers such as age, albumin (alb), creatinine (crea), and bilirubin (bili), while $sigma$ should respond more strongly to acute inflammatory signals such as temperature (temp) and white blood cell count (wbc).
A divergence from this pattern does not invalidate the model, but sets important limits on how the latent coordinates should be communicated clinically.

=== Subgroup separation
To assess whether the latent space organizes patients according to clinical severity, the distributions of $beta$ and $sigma$ are compared across three pairs of clinically defined subgroups: sepsis-positive versus sepsis-negative timesteps, timesteps from patients with peak SOFA score $>=8$ versus those with peak SOFA $< 8$, and timesteps associated with a suspected infection versus
those without.

Separation is quantified using Cohen's $d$, the standardized mean difference between two groups:
$
  d = frac(overline(z)_A - overline(z)_B, s_"pooled"), quad
  s_"pooled" = sqrt(frac(s_A^2 + s_B^2, 2))
$ <eq:cohens-d>
where $overline(z)_A$ and $overline(z)_B$ are the group means of a latent coordinate and $s_A$, $s_B$ are their standard deviations.
By convention, $|d| approx 0.2$ is considered a small effect, $|d| approx 0.5$ medium, and $|d| >= 0.8$ large @cohen1988.
Positive values indicate that group $A$ (e.g. sepsis-positive) has a higher mean latent value than group $B$.

In addition to this scalar summary, the joint distribution of $(beta_t, sigma_t)$ is visualized for each subgroup pair using scatter plots with marginal histograms overlaid on the #acr("PNM") synchronization landscape, so that systematic differences in latent positioning can be interpreted relative to the underlying surface.

=== Cross-split consistency
To assess whether the latent space structure is stable across training runs, the subgroup separation (Cohen's $d$) and feature alignment (Pearson $r$) is evaluated independently for each of the 25 cross-validation splits and report means and standard deviations across splits.
Additionally, the latent distributions for the best and worst performing splits are visualized to check whether performance differences are reflected in qualitatively different spatial organizations.

The key question is whether variability across splits is bounded and structurally consistent, retaining the broad organization of the space even if absolute coordinates shift, or whether it is effectively arbitrary.
The answer determines whether individual-level latent positions can be interpreted, or whether only population-level statistics and directional trajectory changes are reliable.

=== Decoder reconstruction quality
The decoder maps the two-dimensional latent state $(beta_t, sigma_t)$ to a reconstruction of all 52 input features.
Reconstruction quality is assessed per feature using the Pearson correlation coefficient between the reconstructed and ground-truth values (see @eq:pearson).
This metric is chosen over mean-squared error and $R^2$ for the following reasons.
#acr("MSE") is dominated by features with large absolute values and is not comparable across features with different scales (even though the data has been standardized, the values still vary significantly between features).
$R^2$ assumes a linear relationship between predicted and actual values; when the decoder learns a nonlinear mapping, a poor $R^2$ may reflect curvature rather than poor reconstruction quality.
Pearson $r$, being scale-invariant and sensitive to monotonic linear relationships, provides a consistent basis for comparison across features with different distributions.

Features closely connected to #acr("SOFA") and organ dysfunction are expected to be better reconstructed, since the #acr("SOFA") supervision directly links them to the latent position.
Features with high missingness or weak connection to sepsis physiology are expected to reconstruct poorly, and this is interpretable rather than a failure of the model.

