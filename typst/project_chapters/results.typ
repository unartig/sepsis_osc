#import "../thesis_env.typ": *


= Results <sec:results>
This chapter reports findings for each scenario in the order established in @sec:experiments.
The standard baseline first, followed by the five ablations, the seven model variations, and the external validation.
For each scenario, predictive performance is reported first, followed by latent organization analyses (subgroup separation, feature alignment, cross-split consistency) where applicable.

Before inspecting the latent alignment directly, it is useful to establish a baseline: how strongly do the raw input features correlate with the outcome labels themselves?
@fig:feat-label shows the Pearson correlation between each input feature and the three supervision signals Sepsis-3, inflammation, and #acr("SOFA") score with bar color indicating which label a feature is most associated with physiologically (e.g. creatinine directly enters the renal component of the #acr("SOFA") score).
Hatching distinguishing the three labels.

Several patterns stand out.
The Sepsis-3 label shows near-zero correlation with almost all individual features, which is expected, as it is a binary event that emerges from the conjunction of infection and organ dysfunction rather than any single measurement.
The inflammation label correlates weakly but mostly selective with its corresponding feature set.
The #acr("SOFA") score shows stronger correlations with its constituent features, and  additionally with the blood urea nitrogen concentration (bun), hemoglobin concentration (hgb), and the mean arterial pressure (map).
A notable exception is the fraction of inspired oxygen (fio2), which contributes directly to the
respiratory sub-score but shows little correlation with the aggregate #acr("SOFA") label, likely because its effect is mediated through the O2 partial pressure (po2)/fio2 ratio rather than the raw value.

#figure(
  image("../images/project/label_alignment.svg"),
  caption: flex-caption(
    short: [Pearson correlation between input features and outcome labels.],
    long: [Pearson correlation between each input feature and the three supervision signals: Sepsis-3, inflammation, and #acr("SOFA") score. Bar color indicates the physiological feature group; hatching distinguishes the three labels. Features are sorted by combined absolute correlation.],
  ),
) <fig:feat-label>

Note that inflammation-associated features are expected to show only weak alignment with the latent space, and #acr("SOFA")/age-associated ones are expected to align the strongest.
This is due to the Sepsis-3 decomposition done in the #acr("LDM"), where the inflammation risk prediction is detached from the latent space.

== Standard scenario <sec:standard>
The standard scenario, adopted from @backes2026, trains with all loss terms active and serves as the reference for all ablations and variations.
The analysis below examines predictive performance, latent organization, and cross-split stability to establish what the full model learns and how reliably it learns it.

=== Predictive performance
The full model achieves #acr("AUROC") $84.11 plus.minus 0.82$ and #acr("AUPRC") $9.88 plus.minus 0.93$ on the Sepsis-3 label across 25 splits.
An analysis of the branch decomposition reveals an unambiguous asymmetry: the isolated infection branch preserves nearly identical performance (#acr("AUROC") $83.54$, #acr("AUPRC") $9.62$), while the organ branch alone collapses to #acr("AUROC") $58.41$ and #acr("AUPRC") $1.86$.
Sepsis discrimination is thus almost entirely carried by the infection branch, with the organ branch contributing only marginally.
The performance for all components is given in @tab:perf-standard.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],
    [Sepsis-3], [$84.11 plus.minus 0.82$], [$9.88 plus.minus 0.93$],
    [$Delta$#acr("SOFA") $>=$ 2], [$65.82 plus.minus 2.47$], [$9.84 plus.minus 1.08$],
    [Infection], [$72.57 plus.minus 1.08$], [$20.06 plus.minus 0.90$],
    [Sepsis-3 using Organ branch only], [$58.41 plus.minus 3.19$], [$1.86 plus.minus 0.18$],
    [Sepsis-3 using Infection branch only], [$83.54 plus.minus 0.91$], [$9.62 plus.minus 1.01$],
  ),
  caption: flex-caption(
    short: [Standard scenario predictive performance.],
    long: [Predictive performance of the standard model across 25 cross-validation splits (mean $plus.minus$ std).
      Branch-only rows isolate each component by fixing the other branch's output to $1.0$.],
  ),
  kind: table,
) <tab:perf-standard>

The weak organ-branch performance can largely be explained by the sepsis-labeling performed by #acr("YAIB").
The target sepsis label spreads $plus.minus$ 6 hours around the #acr("SOFA") increase, using the ground truth $A_t$ as predictor for sepsis yields #acr("AUROC") of $54.5$ and #acr("AUPRC") of $1.61$.
This is due to the sparse/instaneous $A_t$ label, not being spread around the actual onset, like the sepsis label.
In this light, the performance of the organ branch seems reasonable as it slightly outperforms the ground truth in terms of predictive power (increase in performance here can be attributed to the causal smoothing, allowing to match more sepsis labels than the instantaneous increase).
Conversely, using the ground truth $I_t$ as predictor for sepsis yields #acr("AUROC") of $95.83$ and an #acr("AUPRC") of $15.34$, and the organ branch is not able to achieve this kind of prediction strength.

@tab:performance shows the prediction performance of both the #acr("YAIB")-baseline models and of the standard scenario.
Similar to the paper, the #acr("LDM") outperforms all baseline models significantly in both metrics, at a significance level of $p<0.05$.

#figure(table(
    align: (left, right, right, right, right, right, right),
  columns: 7,
    [Model],
    table.vline(stroke: .5pt),
    [AUROC$plus.minus$std],
    [$t_"AUROC"$],
    [$p_"AUROC"$],
    table.vline(stroke: .5pt),
    [AUPRC$plus.minus$std],
    [$t_"AUPRC"$],
    [$p_"AUPRC"$],

  table.vline(stroke: .5pt),
  [Reg. Logistic Regression], [$77.1 plus.minus 0.4$], [37.26], [$<0.001$], [$4.6 plus.minus 0.1$], [28.50], [$<0.001$],
  [LightGBM], [$77.5 plus.minus 0.3$], [36.62], [$<0.001$], [$5.9 plus.minus 0.2$], [21.02], [$<0.001$],
  [Transformer], [$80.0 plus.minus 0.8$], [17.53], [$<0.001$], [$6.6 plus.minus 0.2$], [17.26], [$<0.001$],
  [LSTM], [$82.0 plus.minus 0.3$], [11.59], [$<0.001$], [$8.0 plus.minus 0.2$], [9.74], [$<0.001$],
  [TCN], [$82.7 plus.minus 0.3$], [7.71], [$<0.001$], [$8.8 plus.minus 0.2$], [5.44], [$<0.001$],
  [GRU], [$83.6 plus.minus 0.3$], [2.70], [$0.011$], [$9.1 plus.minus 0.3$], [3.72], [$<0.001$],
    table.hline(stroke: .5pt),
  [LDM], [*$84.11$$plus.minus$$0.82$*], [--], [--], [*$ 9.88 plus.minus  0.93$*], [--], [--],
),
  caption: flex-caption(
    short: [Performance comparison between the baselines and the #acs("LDM") for the #acs("MIMIC")-IV cohort.],
    long: [Performance comparison between the @yaib baselines and the #acr("LDM") for the #acr("MIMIC")-IV cohort. Bold values indicate the best performing model.],
  ),
) <tab:performance>


=== Subgroup separation
@fig:subgroup-standard shows the latent distribution of the representative split overlaid on the #acr("PNM") surface, for the three subgroups, i.e. high vs. low #acr("SOFA"), infection vs. no infection and sepsis vs. no sepsis.
Marginal histograms are added to the axis and group centroids indicted by the crosses.
Patient trajectories exhibit an operational preference for the upper latent topology ($sigma gt.tilde 0.36$), leaving the lower parameter space unexploited, while $beta$ is tightly centered near $0.55$.

The high versus low #acr("SOFA") comparison produces the largest effects by far ($beta$: $d = 0.90$, $sigma$: $d = 0.77$, @tab:cohen-single-standard), with severe patients shifted toward higher values on both axes, in line with $beta$ encoding physiological compromise and $sigma$ encoding acute organ-immune coupling.
In contrast, infection status separates weakly ($d = 0.07$ on $beta$, $0.20$ on $sigma$), suggesting the infection branch learns its signal independent of the latent geometry.
The sepsis versus no sepsis comparison sits between the two ($beta$: $d = 0.30$, $sigma$: $d = −0.28$), where the negative $sigma$ shift may reflect early-onset or resolving episodes at lower coupling values.

#figure(
  image("../images/project/ablations/standard/svg_subgroup_separation.svg"),
  caption: flex-caption(
    short: [Subgroup separation in the standard latent space.],
    long: [Latent scatter of the representative split for three subgroup pairs (Sepsis vs No Sepsis, Infection vs No Infection, Peak SOFA $>=$ 8 vs $<$ 8) overlaid on the #acr("PNM") synchronization surface.
      Crosses mark group centroids; marginal histograms show the $beta$ and $sigma$ distributions per group.],
  ),
) <fig:subgroup-standard>

#(
  figure(
    table(
      columns: 3,
      [], [$beta$], [$sigma$],
      [High SOFA vs Low SOFA], [0.90], [0.77],
      [Infection vs No infection], [0.07], [0.20],
      [Sepsis vs No sepsis], [0.30], [-0.28],
    ),
    caption: flex-caption(
      short: [Cohen's $d$ for subgroup separation, representative split.],
      long: [Cohen's $d$ for separation along $beta$ and $sigma$ for three clinically defined subgroup pairs, computed on the representative split.
        Positive values indicate the first-named group has a higher mean latent value.],
    ),
    kind: table,
  )
) <tab:cohen-single-standard>



=== Feature alignment
@fig:alignment-standard shows the Pearson $r$ between each input feature and the two latent coordinates, alongside the decoder reconstruction quality.
The latent dimension $beta$ is strongly associated with renal and metabolic markers, blood urea nitrogen (bun) and creatinine concentration (crea) exhibit the highest positive Pearson $r$ with $beta$ among all features, while hemoglobin (hgb) and urine output correlates negatively.
This is collectively consistent with the theoretical interpretation of $beta$ as a proxy for comorbidity burden and reduced physiological fitness.
The $sigma$ correlations are generally slightly weaker and more diffuse, with correlations, mostly agreeing in direction and relative strength with $beta$.
Notable exceptions are the bicarbonate (bicar) and base excess concentration (be) as well as the inflammation associated feature lactate, where the correlation directions are opposite.
Decoder reconstruction quality tracks $beta$ alignment closely, where there is a high alignment with $beta$, the decoder performs better, reflecting the bottleneck's preference for the $beta$-axis variance.

#figure(
  image("../images/project/ablations/standard/svg_alignment_recon.svg"),
  caption: flex-caption(
    short: [Feature alignment with $beta$ and $sigma$, standard scenario.],
    long: [Pearson $r$ between each input feature and $beta$ (solid bars) or $sigma$ (hatched bars) for the representative split, grouped by physiological category (#acr("SOFA"): orange; Inflammation: purple; Age/Other: yellow).
      The blue line (right axis) shows decoder reconstruction $r$ per feature. Features are ordered by mean decoder reconstruction $r$.],
  ),
) <fig:alignment-standard>

@fig:recon-dist-standard shows the ground truth and reconstruction distributions for the ten best reconstructed features.
The decoder captures mean values mostly accurate and somewhat matches the feature distribution, yet the spread often times underestimated, most prominently for hemoglobin (hgb) and age.
This suggests, that the 2-dimensional latent bottleneck compresses the inter-patient variance while preserving central tendencies.

#figure(
  image("../images/project/ablations/standard/svg_recon_dist.svg"),
  caption: flex-caption(
    short: [Reconstruction distributions for top-10 features.],
    long: [Ground-truth (blue) and decoder-reconstructed (orange) distributions for the ten features with the highest reconstruction Pearson $r$ in the standard scenario.],
  ),
) <fig:recon-dist-standard>

@fig:recon-scatter-standard maps the six best-reconstructed latent distribution across the latent space, with each pixel colored by the mean feature value of that ($beta$, $sigma$) coordinate.
Physiological marker values tend towards the desynchronized high-$beta$/high-$sigma$ region, confirming that latent position carries clinically meaningful information about overall patient state, extending beyond the #acr("SOFA")-proxy $s^1$ it was directly trained on.
However, because multiple correlated features jointly determine specific positioning, individual coordinates cannot be cleanly attributed to any single input.
Essentially, the space encodes a compressed clinical summary rather than a readable feature decomposition.

#figure(
  image("../images/project/ablations/standard/svg_latent_scatter.svg"),
  caption: flex-caption(
    short: [Latent space colored by reconstructed feature values.],
    long: [Mean value of the six best-reconstructed features at each $(beta, sigma)$ coordinate in the representative standard split.
      Unhealthy feature values (color scale per panel) tend to occupy the desynchronized upper-right region of the #acr("PNM") surface.],
  ),
) <fig:recon-scatter-standard>

=== Cross-split consistency

In @fig:cv-stability-standard, the mean and standard deviation of the latent distributions per latent-dimension and across the 25 splits is shown.
The $beta$ marginal is highly stable: split means span $approx 0.54-0.56$ with overlapping error bars throughout.
On the other hand, the marginals on the $sigma$ dimensions vary substantially between splits, while most splits center between $0.7-0.8$ with a standard deviation of $approx 0.11$, but several cases shift the mean around $1.0$, and split four collapses to a thin strip.
Notably, this collapse of split number four does not seem to compromise the prediction performance substantially, with an #acr("AUROC") of $83.53$ and an #acr("AUPRC") of $10.4$.
As the clinical interpretation of $sigma$ is more vague compared to $beta$ (immune-organ link vs. biological age and comorbidity).
This baseline ambiguity stems from a fundamental clinical reality: standard clinical protocols collect no direct physiological feature capturing the physical or functional immune-organ link represented by the basal lamina. Consequently, the network is forced to infer this axis dynamically from surrogate global markers.
Therefore, the weak geometrical signal over the $sigma$ dimension does not invalidate the #acr("PNM") interpretation.

#figure(
  image("../images/project/ablations/standard/svg_latent_stability.svg"),
  caption: flex-caption(
    short: [Latent marginal stability across 25 splits.],
    long: [Mean $plus.minus$ std of $beta$ (left) and $sigma$ (right) for each of the 25 cross-validation splits in the standard scenario.
      The dashed line marks the overall mean across splits.],
  ),
) <fig:cv-stability-standard>

In @fig:cv-dists-standard, the log-scaled latent distributions densities are displayed for the two best and two worst performing splits in terms of #acr("AUROC") performance.
All four panels splits show the same qualitative structure: a high density core and then spread and thinning out towards all directions.
This suggests, that specific latent structure does not correlate with performance.

#figure(
  image("../images/project/ablations/standard/svg_dists_cv.svg"),
  caption: flex-caption(
    short: [Latent densities for best and worst splits.],
    long: [Log-scaled latent density contours for the two best- and two worst-performing splits in the standard scenario, overlaid on the #acr("PNM") surface. Darker red indicates higher density.],
  ),
) <fig:cv-dists-standard>

The cross-split Cohen's $d$ values (@tab:cohens-cv-standard) closely match the representative split: high vs. low #acr("SOFA") separation remains large and consistent ($beta$: $0.89 plus.minus 0.15$, $sigma$: $0.73 plus.minus 0.22$), while infection and sepsis subgroups remain weakly separated across all runs ($beta$ separates the sepsis subgroup modestly).
The subgroup structure is therefore a systematic property of the model rather than an artifact of any particular data partition.

#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],

    [High SOFA vs Low SOFA], [$0.89 plus.minus 0.15$], [$0.73 plus.minus 0.22$],
    [Infection vs No infection], [$0.11 plus.minus 0.07$], [$0.19 plus.minus 0.09$],
    [Sepsis vs No sepsis], [$0.30 plus.minus 0.08$], [$-0.14 plus.minus 0.13$],
  ),
  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, standard scenario.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation along $beta$ and $sigma$, averaged over all 25 cross-validation splits in the standard scenario.],
  ),
  kind: table,
) <tab:cohens-cv-standard>


Lastly, @fig:cv-heat-standard confirms these observations at a feature level.
The cross-split $beta$ heatmap mirrors the topology of the representative split in @fig:alignment-standard, confirming that these structural associations are invariant to data partitioning.
The $sigma$ rows are lower in saturation but maintain consistent sign, with the sole clear exception being split four, where $sigma$ alignment is inverted; mirroring its distributional collapse seen in @fig:cv-stability-standard.
Likewise, the decoder reconstruction $r$ (top panel) seem to be stable over the 25 splits.

#figure(
  image("../images/project/ablations/standard/svg_alignment_heatmap_splits.svg"),
  caption: flex-caption(
    short: [Feature alignment stability across splits, standard scenario.],
    long: [Heatmap of Pearson $r$ between each input feature and $beta$ (upper block) or $sigma$ (lower block) for all 25 splits in the standard scenario, sorted by feature category (color bar).
      The top panel shows mean $plus.minus$ std of decoder reconstruction $r$ across splits.],
  ),
) <fig:cv-heat-standard>


== Ablations

=== No sepsis loss ($lambda_"sep"$)
Removing $lambda_"sep"$ eliminates the direct binary cross-entropy supervision on the Sepsis-3 label.
The model must instead learn to predict sepsis indirectly, through the SOFA and infection branch losses that remain active.
This ablation tests whether the joint supervision of the two constituent branches is sufficient to recover sepsis discrimination without an explicit end-to-end signal.

Dropping $lambda_"sep"$ causes a clear drop in Sepsis-3 performance (AUROC $79.11 plus.minus 0.93$, AUPRC $4.36 plus.minus 0.37$), a loss of roughly five #acr("AUROC") points and more than half the #acr("AUPRC") relative to standard (@tab:perf-no-sep).
Notably, the individual branch tasks improve, both the infection branch #acr("AUROC") rises to $82.38$ (from $72.57$) and #acr("AUPRC") to $35.46$ (from $20.06$), and also the $Delta$#acr("SOFA") detection is marginally better.
This suggests that without the sepsis objective, the branches overspecialize on their own targets rather than learning a jointly calibrated product.
The branch decomposition pattern is preserved, the infection branch still dominates Sepsis-3 (AUROC $78.61$ vs $58.75$ for the organ branch), but the infection branch is now miscalibrated for the combined task, explaining the #acr("AUPRC") collapse.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],

    [Sepsis-3], [$79.11 plus.minus 0.93$], [$4.36 plus.minus 0.37$],
    [$Delta$ #acr("SOFA") $>=$ 2], [$66.57 plus.minus 2.26$], [$10.62 plus.minus 1.66$],
    [Infection], [$82.38 plus.minus 0.95$], [$35.46 plus.minus 2.83$],
    [Sepsis-3 using Organ branch only], [$58.75 plus.minus 2.45$], [$1.89 plus.minus 0.19$],
    [Sepsis-3 using Infection branch only], [$78.61 plus.minus 0.96$], [$4.14 plus.minus 0.34$],
  ),
  caption: flex-caption(
    short: [standard_no_sep predictive performance.],
    long: [Predictive performance with $lambda_"sep"$ removed, across 25 cross-validation splits (mean $plus.minus$ std).
      Branch-only rows fix the complementary branch output to $1.0$.],
  ),
  kind: table,
) <tab:perf-no-sep>

As shown in @tab:cohens-cv-no-sep, the latent subgroup structure is largely preserved relative to standard.
High vs low #acr("SOFA") separation remains the dominant effect ($beta$: $d = 0.88$, $sigma$: $d = 0.68$), nearly identical to the baseline values of $0.89$ and $0.73$.
Infection and Sepsis separations are similarly unchanged.
Removing the sepsis supervision therefore does not disrupt the #acr("SOFA")-driven spatial organization of the latent space, since $lambda_"sofa"$ remains active and directly supervises the two coordinates relevant for subgroup positioning.
This confirms that the #acr("SOFA")-driven latent structure is stable across training runs even without sepsis supervision.
A lower variance for the #acr("SOFA") groups compared to the standard scenario suggests marginally more consistent organization, though the difference is negligible.

#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],
    [High SOFA vs Low SOFA], [$0.88 plus.minus 0.10$], [$0.68 plus.minus 0.20$],
    [Infection vs No infection], [$0.08 plus.minus 0.07$], [$0.16 plus.minus 0.09$],
    [Sepsis vs No sepsis], [$0.30 plus.minus 0.05$], [$-0.09 plus.minus 0.16$],
  ),
  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, standard_no_sep.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation along $beta$ and $sigma$, averaged over all 25 splits in the standard_no_sep scenario.],
  ),
  kind: table,
) <tab:cohens-cv-no-sep>

The feature alignment heatmap (@fig:cv-heat-no-sep) is nearly indistinguishable from the standard scenario (@fig:cv-heat-standard); no qualitative differences are visible.

#figure(
  image("../images/project/ablations/standard_no_sep/svg_alignment_heatmap_splits.svg"),
  caption: flex-caption(
    short: [Feature alignment stability across splits, standard_no_sep.],
    long: [Heatmap of Pearson $r$ between input features and $beta$
      (upper block) or $sigma$ (lower block) for all 25 splits in
      the standard_no_sep scenario. Top panel shows mean $plus.minus$
      std of decoder reconstruction $r$ across splits.],
  ),
) <fig:cv-heat-no-sep>


=== No #acr("SOFA")-loss ($lambda_"sofa"$)
Removing $lambda_"sofa"$ eliminates the direct #acr("MSE") supervision on #acr("SOFA") score magnitude, the loss term that most directly connects the latent coordinates to the #acr("PNM")'s clinical interpretation.
This is the most theoretically consequential ablation: if #acr("SOFA") supervision is what anchors $beta$ and $sigma$ to physiological meaning, its removal should degrade latent organization even if sepsis prediction is partially preserved through the remaining losses.

Looking at the performances provided in @tab:perf-no-sofa, the Sepsis-3 performance is nearly unchanged (#acr("AUROC") $83.98 plus.minus 0.78$, #acr("AUPRC") $9.67 plus.minus 0.83$), within one standard deviation of the standard scenario.
The infection branch likewise remains intact (#acr("AUROC") $72.02$, #acr("AUPRC") $19.55$), and the branch decomposition is preserved, again the infection branch alone accounts for almost all sepsis discrimination (#acr("AUROC") $83.20$), while the organ branch collapses to near chance (#acr("AUROC") $50.64$, AUPRC $1.35$).
Without $lambda_"sofa"$ it receives no direct supervision signal, but it confirms that the organ branch contributes negligibly to the sepsis prediction in the standard scenario anyway.
The large increase in $Delta$#acr("SOFA") #acr("AUROC") variance ($50.48 plus.minus 5.20$ vs $65.82 plus.minus 2.47$) and the near-chance mean indicate that #acr("SOFA") detection without #acr("SOFA") supervision is effectively random.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],
    [Sepsis-3], [$83.98 plus.minus 0.78$], [$9.67 plus.minus 0.83$],
    [$Delta$#acr("SOFA") $>=$ 2], [$50.48 plus.minus 5.20$], [$5.75 plus.minus 1.69$],
    [Infection], [$72.02 plus.minus 1.22$], [$19.55 plus.minus 1.01$],
    [Sepsis-3 (organ branch only)], [$50.64 plus.minus 0.46$], [$1.35 plus.minus 0.07$],
    [Sepsis-3 (infection branch only)], [$83.20 plus.minus 0.85$], [$9.34 plus.minus 0.96$],
  ),
  caption: flex-caption(
    short: [standard_no_sofa predictive performance.],
    long: [Predictive performance with $lambda_"sofa"$ removed, across 25 cross-validation splits (mean $plus.minus$ std).
      The organ branch alone collapses to near chance, consistent with the absence of any direct #acr("SOFA") supervision.],
  ),
  kind: table,
) <tab:perf-no-sofa>

The cross-split Cohen's $d$ values (@tab:cohens-cv-no-sofa) reveal the importance of the $lambda_"sofa"$ loss, and that without it all subgroup separations collapse to near zero on both axes.
Most notably, the high vs low #acr("SOFA") effect drops from $d = 0.89$ (standard) to $0.00 plus.minus 0.36$ on $beta$ and $-0.09 plus.minus 0.75$ on $sigma$.
With standard deviations large enough to span the full range from strongly negative to strongly positive means, indicate that the latent space organizes #acr("SOFA") severity arbitrarily and inconsistently across splits.
Infection and sepsis subgroups, already weakly separated in the standard scenario, similarly vanish.
Removing $lambda_"sofa"$ therefore completely destroys the clinically interpretable spatial structure of the latent space.

#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],
    [High SOFA vs Low SOFA], [$0.00 plus.minus 0.36$], [$-0.09 plus.minus 0.75$],
    [Infection vs No infection], [$-0.02 plus.minus 0.10$], [$-0.01 plus.minus 0.15$],
    [Sepsis vs No sepsis], [$-0.02 plus.minus 0.10$], [$-0.04 plus.minus 0.19$],
  ),
  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, standard_no_sofa.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation along $beta$ and $sigma$, averaged over all 25 splits in the standard_no_sofa scenario.],
  ),
  kind: table,
) <tab:cohens-cv-no-sofa>

The collapse of latent-space structure can also be observed in @fig:cv-dists-no-sofa, where the log-scaled densities of the two best- and worst-performing splits is shown.
Here, all splits occupy large amounts of the latent space without a clear structure, compared to the standard scenario, where most splits share the same structure.

#figure(
  image("../images/project/ablations/standard_no_sofa/svg_dists_cv.svg"),
  caption: flex-caption(
    short: [Latent densities for best and worst splits.],
    long: [Log-scaled latent density contours for the two best- and two worst-performing splits in the standard scenario, overlaid on the #acr("PNM") surface. Darker red indicates higher density.],
  ),
) <fig:cv-dists-no-sofa>

The feature alignment heatmap (@fig:cv-heat-no-sofa) reflects the same breakdown.
Unlike the standard scenario, the $beta$ block shows no consistent feature correlation, features that are dominant in all other scenarios, alternate between red and blue across splits, and the overall
pattern is visually noisy with no stable feature-axis associations.
The $sigma$ block is even more variable, with many features flipping sign between splits.
The wide error bars on decoder reconstruction $r$ for several features (notably mean arterial pressure (map), mean corpuscular volume (mcv), and urine in the top panel) are consistent with the model finding different, unstable solutions across runs.
This confirms that $lambda_"sofa"$ is the principal anchor that stabilizes which features are encoded along $beta$ and $sigma$.

#figure(
  image("../images/project/ablations/standard_no_sofa/svg_alignment_heatmap_splits.svg"),
  caption: flex-caption(
    short: [Feature alignment stability across splits, standard_no_sofa.],
    long: [Heatmap of Pearson $r$ between input features and $beta$ (upper block) or $sigma$ (lower block) for all 25 splits in the standard_no_sofa scenario.
      Unlike the standard scenario, no features show consistent alignment across splits.
      Top panel shows mean $plus.minus$ std of decoder reconstruction $r$.],
  ),
) <fig:cv-heat-no-sofa>

Removing $lambda_"sofa"$ leaves Sepsis-3 prediction essentially intact but completely dismantles latent organization, the subgroup separation collapses to near zero across all 25 splits, and feature-axis alignment becomes inconsistent and arbitrary.
This dissociation, stable task performance with destroyed latent structure, demonstrates that the infection branch is sufficient to sustain sepsis prediction independently of the latent space, and that $lambda_"sofa"$ is the critical loss term for grounding $beta$ and $sigma$ in physiological meaning.

=== No decoder ($lambda_"dec"$)
Removing $lambda_"dec"$ eliminates the decoder reconstruction loss, meaning the model no longer needs to map $(beta, sigma)$ back to the 52 input features. Without this constraint, the latent coordinates are free to organize purely around the predictive objectives, with no pressure to preserve feature-level information.

Performance is virtually indistinguishable from standard across all metrics (@tab:perf-no-recon). Sepsis-3 #acr("AUROC") ($84.00 plus.minus 0.81$) and #acr("AUPRC") ($9.86 plus.minus 1.00$) are within noise, the branch decomposition is unchanged, and even $Delta$#acr("SOFA") detection drops only marginally (#acr("AUROC") $63.97$ vs $65.82$).
The reconstruction loss therefore contributes nothing to predictive performance, its role must be purely structural.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],
    [Sepsis-3], [$84.00 plus.minus 0.81$], [$9.86 plus.minus 1.00$],
    [$Delta$#acr("SOFA") $>=$ 2], [$63.97 plus.minus 2.25$], [$9.20 plus.minus 1.13$],
    [Infection], [$72.43 plus.minus 1.18$], [$20.00 plus.minus 0.98$],
    [Sepsis-3 (organ branch only)], [$58.65 plus.minus 2.25$], [$1.87 plus.minus 0.14$],
    [Sepsis-3 (infection branch only)], [$83.42 plus.minus 0.92$], [$9.59 plus.minus 1.09$],
  ),
  caption: flex-caption(
    short: [standard_no_recon predictive performance.],
    long: [Predictive performance with $lambda_"dec"$ removed, across 25 cross-validation splits (mean $plus.minus$ std).],
  ),
  kind: table,
) <tab:perf-no-recon>

#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],
    [High SOFA vs Low SOFA], [$0.95 plus.minus 0.21$], [$0.75 plus.minus 0.47$],
    [Infection vs No infection], [$0.18 plus.minus 0.09$], [$0.24 plus.minus 0.13$],
    [Sepsis vs No sepsis], [$0.33 plus.minus 0.10$], [$-0.17 plus.minus 0.17$],
  ),
  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, standard_no_recon.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation
      along $beta$ and $sigma$, averaged over all 25 splits in the
      standard_no_recon scenario.],
  ),
  kind: table,
) <tab:cohens-cv-no-recon>

Subgroup separation is preserved and, if anything, slightly stronger than in the standard scenario (@tab:cohens-cv-no-recon).
The high vs low #acr("SOFA") effect increases marginally on $beta$ ($0.95$ vs $0.89$), while infection and sepsis separations are also modestly larger.
The elevated standard deviation on $sigma$ for the SOFA comparison ($0.47$ vs $0.22$) indicates more inter-split variability in how $sigma$ handles severity separation, but the mean effect remains comparable.
Removing the reconstruction constraint does not degrade, and may slightly sharpen, the clinically relevant latent structure.

The feature alignment heatmap (@fig:cv-heat-no-recon) shows the consequence of removing $lambda_"dec"$, the decoder reconstruction $r$ collapses to near zero for all features (top panel), with means clustered around $0.0$ and wide error bars reflecting random variation around chance.
Despite this, the $beta$ and $sigma$ alignment blocks are qualitatively similar to the standard scenario.
Blood urea nitrogen concentration (bun) and creatinine (crea) form consistently warm (positively correlated) $beta$ columns, and the directional pattern across other features is broadly maintained, though with somewhat more inter-split noise visible in the $sigma$ block.
Interestingly, the latent coordinates encode feature-relevant structure even when not explicitly trained to reconstruct those features, indicating that $lambda_"sofa"$ and $lambda_"sep"$ alone are sufficient to induce a physiologically organized latent space.
The reconstruction loss adds decodability and strengthens the latent separation, but is not the source of that organization.

#figure(
  image("../images/project/ablations/standard_no_recon/svg_alignment_heatmap_splits.svg"),
  caption: flex-caption(
    short: [Feature alignment stability across splits, standard_no_recon.],
    long: [Heatmap of Pearson $r$ between input features and $beta$ (upper block) or $sigma$ (lower block) for all 25 splits in the standard_no_recon scenario.
      Top panel shows mean $plus.minus$ std of decoder reconstruction $r$, which collapses to near zero across all features in the absence of $lambda_"dec"$.],
  ),
) <fig:cv-heat-no-recon>

Removing $lambda_"dec"$ has no effect on predictive performance and leaves the clinically interpretable latent structure largely intact.
The only consequence is the collapse of decoder reconstruction quality.
This confirms that the reconstruction loss serves a purely auxiliary role.
It adds a decodable feature representation on top of the latent space, even though it is not responsible for the spatial organization induced by the supervision losses, though it amplifies the latent separation.

=== No spreading loss ($lambda_"spread"$)
Removing $lambda_"spread"$ eliminates the variance penalty that prevents latent collapse.
Performance is essentially unchanged from standard across all metrics (@tab:perf-no-spread), confirming the spreading loss contributes nothing to the predictive objective.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],
    [Sepsis-3], [$84.03 plus.minus 0.91$], [$9.92 plus.minus 0.97$],
    [$Delta$#acr("SOFA") $>=$ 2], [$67.75 plus.minus 3.50$], [$10.63 plus.minus 1.63$],
    [Infection], [$72.27 plus.minus 1.26$], [$19.97 plus.minus 0.95$],
    [Sepsis-3 (organ branch only)], [$60.41 plus.minus 3.78$], [$1.98 plus.minus 0.24$],
    [Sepsis-3 (infection branch only)], [$83.46 plus.minus 0.96$], [$9.65 plus.minus 1.00$],
  ),
  caption: flex-caption(
    short: [standard_no_spread predictive performance.],
    long: [Predictive performance with $lambda_"spread"$ removed, across
      25 cross-validation splits (mean $plus.minus$ std).],
  ),
  kind: table,
) <tab:perf-no-spread>

The mean subgroup separations are close to standard, but the standard deviations are roughly double (@tab:cohens-cv-no-spread): $0.40$ on $beta$ and $0.59$ on $sigma$ for the #acr("SOFA") comparison versus $0.15$ and $0.22$ in the baseline.
The latent stability plot (@fig:cv-stability-no-spread) reflects this, $beta$ means are comparable to standard, but $sigma$ shows wider excursions, with two splits near split 2 and 9 drifting notably above the overall mean.

#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],
    [High SOFA vs Low SOFA], [$0.92 plus.minus 0.40$], [$0.72 plus.minus 0.59$],
    [Infection vs No infection], [$0.14 plus.minus 0.13$], [$0.16 plus.minus 0.15$],
    [Sepsis vs No sepsis], [$0.28 plus.minus 0.09$], [$-0.08 plus.minus 0.19$],
  ),
  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, standard_no_spread.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation
      along $beta$ and $sigma$, averaged over all 25 splits in the
      standard_no_spread scenario.],
  ),
  kind: table,
) <tab:cohens-cv-no-spread>

#figure(
  image("../images/project/ablations/standard_no_spread/svg_latent_stability.svg"),
  caption: flex-caption(
    short: [Latent marginal stability across 25 splits, standard_no_spread.],
    long: [Mean $plus.minus$ std of $beta$ (left) and $sigma$ (right)
      per split in the standard_no_spread scenario. Dashed line marks
      the grand mean.],
  ),
) <fig:cv-stability-no-spread>


The density plots (@fig:cv-dists-no-spread) make the instability concrete, the best-performing split (Rep. 5, Fold 1; AUROC $0.86$) collapses to a narrow near-vertical strip at low $beta$, while the remaining three splits show the usual structure.
Crucially, the collapsed split achieves identical #acr("AUROC") to the best standard splits.

#figure(
  image("../images/project/ablations/standard_no_spread/svg_dists_cv.svg"),
  caption: flex-caption(
    short: [Latent densities for best and worst splits, standard_no_spread.],
    long: [Log-scaled latent density contours for the two best- and two worst-performing splits in the standard_no_spread scenario.
      The best split (Rep. 5, Fold 1) collapses to a narrow vertical strip despite achieving #acr("AUROC") $0.86$.],
  ),
) <fig:cv-dists-no-spread>

The feature alignment heatmap (@fig:cv-heat-no-spread) mirrors this picture, the dominant $beta$ associations are mostly preserved across splits, but with more inter-split noise than standard, particularly in the $sigma$ block.

#figure(
  image("../images/project/ablations/standard_no_spread/svg_alignment_heatmap_splits.svg"),
  caption: flex-caption(
    short: [Feature alignment stability across splits, standard_no_spread.],
    long: [Heatmap of Pearson $r$ between input features and $beta$ (upper block) or $sigma$ (lower block) for all 25 splits in the standard_no_spread scenario.
      Top panel shows mean $plus.minus$ std of decoder reconstruction $r$ across splits.],
  ),
) <fig:cv-heat-no-spread>

$lambda_"spread"$ is irrelevant to task performance but acts as a regularizer on the solution space.
Without it, the model more often collapses the latent distribution without any predictive penalty, and
subgroup separation becomes unreliable across splits.
This mechanism that makes interpretable organization emerge more consistently rather.


=== No boundary loss ($lambda_"boundary"$)
Removing $lambda_"boundary"$ eliminates the soft penalty against edge-of-grid predictions, leaving the model free to place representations anywhere in the latent space including at its boundaries.
Performance is again essentially unchanged from standard (@tab:perf-no-boundary) and requires no further discussion.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],
    [Sepsis-3], [$84.01 plus.minus 0.90$], [$9.96 plus.minus 0.91$],
    [$Delta$ #acr("SOFA") $>=$ 2], [$67.08 plus.minus 2.83$], [$10.82 plus.minus 1.26$],
    [Infection], [$72.12 plus.minus 1.15$], [$19.78 plus.minus 0.84$],
    [Sepsis-3 (organ branch only)], [$59.69 plus.minus 2.90$], [$1.93 plus.minus 0.20$],
    [Sepsis-3 (infection branch only)], [$83.40 plus.minus 0.96$], [$9.60 plus.minus 0.90$],
  ),
  caption: flex-caption(
    short: [standard_no_boundary predictive performance.],
    long: [Predictive performance with $lambda_"boundary"$ removed, across 25 cross-validation splits (mean $plus.minus$ std).],
  ),
  kind: table,
) <tab:perf-no-boundary>


Subgroup separation is well preserved and notably more consistent than in the standard_no_spread scenario (@tab:cohens-cv-no-boundary).
The high vs low #acr("SOFA") effect is $0.86$ on $beta$ and $0.85$ on $sigma$, the $sigma$ value is actually higher than in standard ($0.73$), and standard deviations remain relatively small throughout.

#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],
    [High SOFA vs Low SOFA], [$0.86 plus.minus 0.15$], [$0.85 plus.minus 0.17$],
    [Infection vs No infection], [$0.10 plus.minus 0.09$], [$0.23 plus.minus 0.06$],
    [Sepsis vs No sepsis], [$0.32 plus.minus 0.06$], [$-0.11 plus.minus 0.21$],
  ),
  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, standard_no_boundary.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation along $beta$ and $sigma$, averaged over all 25 splits in the standard_no_boundary scenario.],
  ),
  kind: table,
) <tab:cohens-cv-no-boundary>

The latent density plots (@fig:cv-dists-no-boundary) reveal the consequence of removing the boundary penalty directly.
The worst split (Rep. 3, Fold 5) pushes the entire patient mass against the right edge of the grid, forming a tall narrow strip pressed against the high-$beta$ boundary.
The three remaining splits show the usual structure.

#figure(
  image("../images/project/ablations/standard_no_boundary/svg_dists_cv.svg"),
  caption: flex-caption(
    short: [Latent densities for best and worst splits,
      standard_no_boundary.],
    long: [Log-scaled latent density contours for the two best- and two worst-performing splits in the standard_no_boundary scenario.
      The worst split (Rep. 3, Fold 5) collapses against the high-$beta$ grid boundary, the precise failure mode $lambda_"boundary"$ is designed to prevent.],
  ),
) <fig:cv-dists-no-boundary>

The feature alignment heatmap and feature reconstruction(@fig:cv-heat-no-boundary) is qualitatively indistinguishable from standard.
The boundary-pressing split does not produce a visibly anomalous row, suggesting that even when the distribution migrates to the grid edge, the relative feature ordering within the latent space is preserved.

#figure(
  image("../images/project/ablations/standard_no_boundary/svg_alignment_heatmap_splits.svg"),
  caption: flex-caption(
    short: [Feature alignment stability across splits,
      standard_no_boundary.],
    long: [Heatmap of Pearson $r$ between input features and $beta$ (upper block) or $sigma$ (lower block) for all 25 splits in the standard_no_boundary scenario.
      Top panel shows mean $plus.minus$ std of decoder reconstruction $r$ across splits.],
  ),
) <fig:cv-heat-no-boundary>

$lambda_"boundary"$ successfully prevents the latent distribution from drifting against the grid edges, which occurs in a minority of splits when it is removed.
Like $lambda_"spread"$, it acts purely as a geometric regularizer, the task performance and feature alignment are unaffected, but without it the spatial interpretation of the #acr("PNM") surface can break down in individual runs.

=== Summary

@tab:ablation-summary collects the predictive performance across all ablation scenarios.
@fig:ablation-comparison visualizes the joint (#acr("AUROC"), #acr("AUPRC")) distribution per scenario as confidence ellipses (95% interval), making the (non-)separation between scenarios immediately apparent.

#figure(
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    stroke: .5pt,
    table.header(
      [Scenario],
      table.cell(colspan: 2)[*Sepsis-3*],
      table.cell(colspan: 2)[*$Delta$#acr("SOFA") $>=$ 2*],
      table.cell(colspan: 2)[*Infection*],
      [],
      [AUROC], [AUPRC],
      [AUROC], [AUPRC],
      [AUROC], [AUPRC],
    ),
    [standard],
    text(10pt)[*84.11* $plus.minus$ 0.82],
    text(10pt)[9.88 $plus.minus$ 0.93],
    text(10pt)[65.82 $plus.minus$ 2.47],
    text(10pt)[9.84 $plus.minus$ 1.08],
    text(10pt)[72.57 $plus.minus$ 1.08],
    text(10pt)[20.06 $plus.minus$ 0.90],

    [\*\_no_sep],
    text(10pt)[79.11 $plus.minus$ 0.93],
    text(10pt)[4.36 $plus.minus$ 0.37],
    text(10pt)[66.57 $plus.minus$ 2.26],
    text(10pt)[10.62 $plus.minus$ 1.66],
    text(10pt)[*82.38* $plus.minus$ 0.95],
    text(10pt)[*35.46* $plus.minus$ 2.83],

    [\*\_no\_sofa],
    text(10pt)[83.98 $plus.minus$ 0.78],
    text(10pt)[9.67 $plus.minus$ 0.83],
    text(10pt)[50.48 $plus.minus$ 5.20],
    text(10pt)[5.75 $plus.minus$ 1.69],
    text(10pt)[72.02 $plus.minus$ 1.22],
    text(10pt)[19.55 $plus.minus$ 1.01],

    [\*\_no\_recon],
    text(10pt)[84.00 $plus.minus$ 0.81],
    text(10pt)[9.86 $plus.minus$ 1.00],
    text(10pt)[63.97 $plus.minus$ 2.25],
    text(10pt)[9.20 $plus.minus$ 1.13],
    text(10pt)[72.43 $plus.minus$ 1.18],
    text(10pt)[20.00 $plus.minus$ 0.98],

    [\*\_no\_spread],
    text(10pt)[84.03 $plus.minus$ 0.91],
    text(10pt)[9.92 $plus.minus$ 0.97],
    text(10pt)[*67.75* $plus.minus$ 3.50],
    text(10pt)[10.63 $plus.minus$ 1.63],
    text(10pt)[72.27 $plus.minus$ 1.26],
    text(10pt)[19.97 $plus.minus$ 0.95],

    text(10pt)[\*\_no\_boundary],
    text(10pt)[84.01 $plus.minus$ 0.90],
    text(10pt)[*9.96* $plus.minus$ 0.91],
    text(10pt)[67.08 $plus.minus$ 2.83],
    text(10pt)[*10.82* $plus.minus$ 1.26],
    text(10pt)[72.12 $plus.minus$ 1.15],
    text(10pt)[19.78 $plus.minus$ 0.84],
  ),
  caption: flex-caption(
    short: [Ablation predictive performance summary.],
    long: [Mean $plus.minus$ std of #acr("AUROC") and #acr("AUPRC") across 25 cross-validation splits for all ablation scenarios.
      Values $times 100$.
      Bold entries indicate the best performing ablation for each metric.],
  ),
  kind: table,
) <tab:ablation-summary>

#figure(
  image("../images/project/ablation_comparison.svg"),
  caption: flex-caption(
    short: [Ablation experiment comparison.],
    long: [Joint (AUROC, AUPRC) distributions across 25 splits for all
      ablation scenarios, shown as scatter points with 95% confidence
      ellipses. Left: Sepsis-3; center: $Delta$SOFA $>=$ 2; right:
      suspected infection.],
  ),
) <fig:ablation-comparison>

The performance picture is clear and consistent with the individual analyses.
Only two ablations produce meaningful changes.
Removing $lambda_"sep"$ causes a five-point #acr("AUROC") drop and halves the Sepsis-3 #acr("AUPRC"), with the ellipse in @fig:ablation-comparison shifting visibly left and downward and tightening, reflecting not just lower performance but reduced variance, as the branches optimize independently on simpler targets.
Removing $lambda_"sofa"$ leaves Sepsis-3 intact but collapses $Delta$#acr("SOFA") detection to near chance (#acr("AUROC") $50.48 plus.minus 5.20$), with a dramatically enlarged ellipse in the center panel reflecting the high instability of a task optimized without its primary supervision signal.
Interestingly, without the $lambda_"sofa"$-loss, the infection branch is almost fully compensating for the missing organ dysfunction predictions.
The remaining four ablations, no reconstruction, no spread, no boundary are indistinguishable from standard on all three tasks, their ellipses overlapping almost perfectly.

@fig:ablation-alignment-summary and @fig:ablation-decoder-summary extend this comparison to the latent space and decoder.
The decoder summary (@fig:ablation-decoder-summary) reveals, that only standard\_no\_recon produces a uniformly white mean row and a bright std row, isolating the reconstruction loss as the only term responsible for decodability.
The standard_no_sofa configuration yields improved reconstruction performance and higher decoder variance, suggesting that the reconstruction loss is effectively suppressed by the competing #acr("SOFA") objective, at least to some extent.

#figure(
  image("../images/project/ablation_decoder_summary.svg"),
  caption: flex-caption(
    short: [Decoder reconstruction summary across ablations.],
    long: [Mean (top) and standard deviation across splits (bottom) of
      decoder reconstruction Pearson $r$ per feature, for all six
      ablation scenarios.],
  ),
) <fig:ablation-decoder-summary>


The feature alignment heatmap (@fig:ablation-alignment-summary) tells a similar story.
Five of the six scenarios, standard, no\_sep, no\_recon, no\_spread, no\_boundary, show virtually identical $beta$ and $sigma$ alignment patterns in sign and value structure throughout.
standard\_no\_sofa is the sole exception, its rows are visually washed out relative to all others, confirming the latent disorganization observed in the cross-split Cohen's $d$ analysis.

#figure(
  image("../images/project/ablation_alignment_summary.svg"),
  caption: flex-caption(
    short: [Feature alignment summary across ablations.],
    long: [Pearson $r$ between each input feature and $beta$ (upper block) or $sigma$ (lower block), averaged over 25 splits, for all six ablation scenarios.
      Feature categories follow the color bar at the bottom.],
  ),
) <fig:ablation-alignment-summary>

Taken together, the ablation analysis identifies a clear two-tier structure among the loss terms.
$lambda_"sep"$ and $lambda_"sofa"$ are load-bearing, the former for task calibration, the latter for latent interpretability.
The remaining terms, i.e. $lambda_"dec"$, $lambda_"spread"$, $lambda_"boundary"$, $lambda_"recon"$, are geometric regularizers that shape the solution space without contributing to either predictive performance or (except for the reconstruction loss) the feature-level organization of the latent coordinates.

== Variations

=== MLP
The mlp scenario replaces the frozen #acr("PNM") surface with a small jointly-trained #acr("MLP"), meaning the latent space topology is no longer fixed, it rather evolves during training alongside the encoder and prediction branches.

Sepsis-3 performance is essentially identical to standard (#acr("AUROC") $84.03 plus.minus 0.77$, #acr("AUPRC") $9.85 plus.minus 1.01$), and the infection branch decomposition is unchanged (@tab:perf-mlp).
The most striking difference is in organ dysfunction detection, the $Delta$#acr("SOFA") #acr("AUROC") rises to $78.28 plus.minus 1.40$ (from $65.82$ in standard) and #acr("AUPRC") to $22.65$ (from $9.84$), and the organ branch alone achieves #acr("AUROC") $67.75$ versus $58.80$.
The jointly-trained surface apparently learns a geometry that better supports #acr("SOFA") discrimination, since the #acr("MLP") is free to shape the latent space specifically around the #acr("SOFA") supervision signal rather than being constrained to the #acr("PNM")'s biophysically-motivated topology.


#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],

    [Sepsis-3], [$84.03 plus.minus 0.77$], [$9.85 plus.minus 1.01$],
    [$Delta$ #acr("SOFA") $>=$ 2], [$78.28 plus.minus 1.40$], [$22.65 plus.minus 1.20$],
    [Infection], [$72.87 plus.minus 1.18$], [$20.40 plus.minus 1.07$],
    [Sepsis-3 using Organ branch only], [$67.75 plus.minus 1.45$], [$2.50 plus.minus 0.24$],
    [Sepsis-3 using Infection branch only], [$83.59 plus.minus 0.84$], [$9.59 plus.minus 0.96$],
  ),
  caption: flex-caption(
    short: [mlp scenario predictive performance.],
    long: [Predictive performance of the #acr("MLP") variation across 25 cross-validation splits (mean $plus.minus$ std).
      The latent space topology is jointly learned rather than fixed to the #acr("PNM") surface.],
  ),
  kind: table,
) <tab:perf-mlp>

The latent stability plot (@fig:cv-stability-mlp) reveals substantially higher inter-split variability than any ablation scenario.
This reflects the added degrees of freedom: with no fixed surface to anchor the latent geometry, different training runs converge to qualitatively different spatial organizations.

#figure(
  image("../images/project/variations/mlp/svg_latent_stability.svg"),
  caption: flex-caption(
    short: [Latent marginal stability across 25 splits, mlp scenario.],
    long: [Mean $plus.minus$ std of $beta$ (left) and $sigma$ (right) per split in the mlp scenario.
      The dashed line marks the overall mean.
      Both marginals show substantially higher inter-split variability than in any ablation scenario.],
  ),
) <fig:cv-stability-mlp>

The density plots (@fig:cv-dists-mlp) confirm this, as all four panels show a qualitatively different latent structure from the standard scenario.
The patient mass forms a diagonal band running from low-$beta$/high-$sigma$ to high-$beta$/low-$sigma$, tracing the anti-correlation structure the #acr("MLP") has learned between the two coordinates.
This diagonal organization is consistent across the four displayed splits and suggests the #acr("MLP") converges to a low-dimensional manifold rather than filling the 2-dimensional space.
But the specific orientation and extent of that manifold varies across splits, consistent with
the stability plot.
The background surface also differs from split to split, as expected for a jointly-trained function.

#figure(
  image("../images/project/variations/mlp/svg_dists_cv.svg"),
  caption: flex-caption(
    short: [Latent densities for best and worst splits, mlp scenario.],
    long: [Log-scaled latent density contours for the two best- and two worst-performing splits in the mlp scenario, overlaid on the learned #acr("MLP") surface (which differs per split).
      The patient mass consistently forms a diagonal band absent in the standard scenario.],
  ),
) <fig:cv-dists-mlp>

=== Surrogate
Sepsis-3 performance is identical to standard (#acr("AUROC") $84.03 plus.minus 0.79$, #acr("AUPRC") $9.90 plus.minus 0.87$), and the branch decomposition is unchanged (@tab:perf-surrogate).
Notably, the organ dysfunction detection improves substantially over the standard scenario, with $Delta$#acr("SOFA") #acr("AUROC") $74.41 plus.minus 2.35$ and #acr("AUPRC") $16.77$, and the organ branch alone reaching #acr("AUROC") $64.74$.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],
    [Sepsis-3], [$84.03 plus.minus 0.79$], [$9.90 plus.minus 0.87$],
    [$Delta$ #acr("SOFA") $>=$ 2], [$74.41 plus.minus 2.35$], [$16.77 plus.minus 2.14$],
    [Infection], [$72.81 plus.minus 1.03$], [$20.25 plus.minus 0.92$],
    [Sepsis-3 using Organ branch only], [$64.74 plus.minus 2.47$], [$2.50 plus.minus 0.27$],
    [Sepsis-3 using Infection branch only], [$83.53 plus.minus 0.90$], [$9.58 plus.minus 0.90$],
  ),
  caption: flex-caption(
    short: [Surrogate scenario predictive performance.],
    long: [Predictive performance of the surrogate scenario across 25 cross-validation splits (mean $plus.minus$ std).
      The latent space is defined by a frozen #acr("MLP") approximation of the #acr("PNM") surface.],
  ),
) <tab:perf-surrogate>

The subgroup separation tells a notable story (@tab:cohens-cv-surrogate).
The high vs low #acr("SOFA") effect on $beta$ rises to $1.29 plus.minus 0.30$, the largest value observed across all ablations and variations, while the $sigma$ effect remains comparable to standard ($0.83$) but with high variance ($plus.minus 1.01$), indicating inconsistent $sigma$-based organization across splits.
Infection separation on $beta$ also increases to $0.36$, modestly above the standard value of $0.10$, suggesting the surrogate surface creates a geometry that better exposes infection-related structure along $beta$.
The sepsis comparison is unchanged.


#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],
    [High SOFA vs Low SOFA], [$1.29 plus.minus 0.30$], [$0.83 plus.minus 1.01$],
    [Infection vs No infection], [$0.36 plus.minus 0.09$], [$0.23 plus.minus 0.29$],
    [Sepsis vs No sepsis], [$0.29 plus.minus 0.08$], [$0.08 plus.minus 0.23$],
  ),
  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, surrogate scenario.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation along $beta$ and $sigma$, averaged over all 25 splits in the surrogate scenario.],
  ),
) <tab:cohens-cv-surrogate>

The latent stability plot (@fig:cv-stability-surrogate) shows $beta$ is well-behaved, the split means cluster tightly between $0.555$ and $0.580$ with only two outliers near $0.60$, comparable to the standard scenario.
The $sigma$ marginal is more variable, with two splits showing means above $1.1$ and very wide error bars.

#figure(
  image("../images/project/variations/surrogate/svg_latent_stability.svg"),
  caption: flex-caption(
    short: [Latent marginal stability across 25 splits, surrogate scenario.],
    long: [Mean $plus.minus$ std of $beta$ (left) and $sigma$ (right) per split in the surrogate scenario.
      The dashed line marks the overall mean.
      Two outliers drive the elevated $sigma$ variance.],
  ),
) <fig:cv-stability-surrogate>

The density plots (@fig:cv-dists-surrogate) reveal a qualitatively distinct latent structure from the standard scenario.
Here, the patient mass forms a compact diagonal band oriented from low-$beta$/high-$sigma$ to high-$beta$/low-$sigma$, closely resembling the #acr("MLP") scenario.
This diagonal structure is consistent across three of the four displayed splits, with the
worst split (Rep. 2, Fold 2; AUROC $0.83$) showing a more compact, vertically oriented mass at higher $beta$ values.

#figure(
  image("../images/project/variations/surrogate/svg_dists_cv.svg"),
  caption: flex-caption(
    short: [Latent densities for best and worst splits, surrogate scenario.],
    long: [Log-scaled latent density contours for the two best- and two worst-performing splits in the surrogate scenario, overlaid on the frozen surrogate surface.
      Three splits show a diagonal band structure; the worst split (Rep. 2, Fold 2) deviates toward a more compact vertical organization.],
  ),
) <fig:cv-dists-surrogate>

The feature alignment heatmap (@fig:cv-heat-surrogate) is broadly consistent with standard.
Yet, two notable differences are visible, the $beta$ saturation for the leading features is somewhat lower than in standard (lighter red), consistent with the larger spread of the patient mass weakening per-feature correlations; and the $sigma$ block shows more inter-split stripe noise.
Decoder reconstruction $r$ values are stable and slightly lower overall than standard, with blood urea nitrogen concentration (bun) and creatinine near $r approx 0.45$–$0.50$ rather than $0.60$.

#figure(
  image("../images/project/variations/surrogate/svg_alignment_heatmap_splits.svg"),
  caption: flex-caption(
    short: [Feature alignment stability across splits, surrogate scenario.],
    long: [Heatmap of Pearson $r$ between input features and $beta$ (upper block) or $sigma$ (lower block) for all 25 splits in the surrogate scenario.
      Top panel shows mean $plus.minus$ std of decoder reconstruction $r$ across splits.],
  ),
) <fig:cv-heat-surrogate>

The surrogate scenario matches standard on Sepsis-3 while improving organ dysfunction detection, and produces the strongest $beta$-based #acr("SOFA") separation of any scenario.
The frozen approximate surface is sufficient to recover the main structural properties of the standad latent space, though with greater inter-split variability in $sigma$ and a diagonal
rather than triangular latent structure.

=== Approximation
Sepsis-3 and infection performance are again indistinguishable from standard (@tab:perf-approx). Organ dysfunction detection matches the mlp scenario closely (#acr("AUROC") $77.98 plus.minus 1.43$, #acr("AUPRC") $19.57$; organ branch alone #acr("AUROC") $67.81$), confirming that the improvement over standard is a property of the approximate surface.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],
    [Sepsis-3], [$84.05 plus.minus 0.83$], [$9.65 plus.minus 1.00$],
    [$Delta$ #acr("SOFA") $>=$ 2], [$77.98 plus.minus 1.43$], [$19.57 plus.minus 2.08$],
    [Infection], [$73.16 plus.minus 0.75$], [$20.53 plus.minus 0.85$],
    [Sepsis-3 using Organ branch only], [$67.81 plus.minus 1.33$], [$2.63 plus.minus 0.23$],
    [Sepsis-3 using Infection branch only], [$83.59 plus.minus 0.86$], [$9.44 plus.minus 1.03$],
  ),
  caption: flex-caption(
    short: [Approx scenario predictive performance.],
    long: [Predictive performance of the approx scenario across 25 cross-validation splits (mean $plus.minus$ std).
      The latent space is defined by a frozen closed-form approximation of the #acr("PNM") surface.],
  ),
) <tab:perf-approx>

The subgroup separation results are striking (@tab:cohens-cv-approx).
The high vs low #acr("SOFA") effect on $beta$ reaches $1.44 plus.minus 0.08$, the largest value across all scenarios and, crucially, with a standard deviation of only $0.08$, far tighter than the surrogate ($plus.minus 0.30$) or any other scenario.
This indicates that the analytic surface consistently and reliably separates #acr("SOFA") severity along $beta$ across all 25 splits.
The $sigma$ #acr("SOFA") effect, collapses to near zero ($0.17 plus.minus 1.05$) with enormous variance, suggesting that $sigma$ carries no stable #acr("SOFA") signal in this scenario.
The analytic surface apparently concentrates all severity information along the $beta$ axis.
Infection separation on $beta$ is $0.36 plus.minus 0.06$, identical to the surrogate and again
more stable.

#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],
    [High SOFA vs Low SOFA], [$1.44 plus.minus 0.08$], [$0.17 plus.minus 1.05$],
    [Infection vs No infection], [$0.36 plus.minus 0.06$], [$0.15 plus.minus 0.21$],
    [Sepsis vs No sepsis], [$0.28 plus.minus 0.05$], [$0.06 plus.minus 0.16$],
  ),
  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, approx scenario.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation along $beta$ and $sigma$, averaged over all 25 splits in the approx scenario.],
  ),
) <tab:cohens-cv-approx>

The latent stability plot (@fig:cv-stability-approx) is the most stable seen across all variation scenarios.
The $beta$ marginal spans only $approx 0.535$–$0.555$ for the majority of splits, with tight error bars throughout and only one minor outlier near split 0.
More remarkably, the $sigma$ marginal shows a clear bimodal structure, roughly the first sixi splits have means near $0.65$–$0.75$, while splits 7 onward cluster tightly around $0.62$ with very small error bars.
This discrete shift likely reflects a bifurcation in the analytic surface map.

#figure(
  image("../images/project/variations/approx/svg_latent_stability.svg"),
  caption: flex-caption(
    short: [Latent marginal stability across 25 splits, approx scenario.],
    long: [Mean $plus.minus$ std of $beta$ (left) and $sigma$ (right) per split in the approx scenario.
      The $beta$ marginal is highly stable; the $sigma$ marginal shows a discrete shift between       early and later splits.],
  ),
) <fig:cv-stability-approx>

The density plots (@fig:cv-dists-approx) confirm the $sigma$ bifurcation directly.
The best split (Rep. 2, Fold 1; #acr("AUROC") $0.85$) shows a compact horizontal ellipse at mid-$beta$ and low $sigma$, occupying a very small region of the space.
The three remaining splits display the diagonal band structure shared with the surrogate and mlp scenarios, oriented from low-$beta$/mid-$sigma$ to high-$beta$/low-$sigma$.

#figure(
  image("../images/project/variations/approx/svg_dists_cv.svg"),
  caption: flex-caption(
    short: [Latent densities for best and worst splits, approx scenario.],
    long: [Log-scaled latent density contours for the two best- and two worst-performing splits in the approx scenario, overlaid on the analytic surface.
      The best split collapses to a compact horizontal ellipse; the remaining splits show the diagonal band structure.],
  ),
) <fig:cv-dists-approx>

The feature alignment heatmap (@fig:cv-heat-approx) mirrors the standard scenario more closely than the surrogate.
The $sigma$ block maintains a similar directional pattern to standard, though with horizontal striping in several rows reflecting the bimodal split structure.
Decoder reconstruction $r$ values are comparable to the surrogate.

#figure(
  image("../images/project/variations/approx/svg_alignment_heatmap_splits.svg"),
  caption: flex-caption(
    short: [Feature alignment stability across splits, approx scenario.],
    long: [Heatmap of Pearson $r$ between input features and $beta$ (upper block) or $sigma$ (lower block) for all 25 splits in the approx scenario.
      Horizontal striping in the $sigma$ block reflects the bimodal split structure visible in the stability plot.],
  ),
) <fig:cv-heat-approx>

=== Linear Smooth
Performance is unchanged from standard on Sepsis-3 and infection (@tab:perf-linear-smooth). Organ dysfunction detection improves over the standard scenario (#acr("AUROC") $77.05 plus.minus .21$) though falls short of the approx and mlp scenarios, and the organ branch alone reaches #acr("AUROC") $63.47$.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],
    [Sepsis-3], [$84.05 plus.minus 0.82$], [$9.83 plus.minus 0.96$],
    [$Delta$ #acr("SOFA") $>=$ 2], [$77.05 plus.minus 1.21$], [$16.61 plus.minus 1.17$],
    [Infection], [$72.37 plus.minus 1.31$], [$20.00 plus.minus 0.91$],
    [Sepsis-3 using Organ branch only], [$63.47 plus.minus 1.46$], [$2.21 plus.minus 0.17$],
    [Sepsis-3 using Infection branch only], [$83.53 plus.minus 0.89$], [$9.61 plus.minus 1.07$],
  ),
  caption: flex-caption(
    short: [Linear smooth scenario predictive performance.],
    long: [Predictive performance of the linear_smooth scenario across 25 cross-validation splits (mean $plus.minus$ std).],
  ),
) <tab:perf-linear-smooth>

The subgroup separation results are consistent with the $sigma$-free surface design (@tab:cohens-cv-linear-smooth).
The $beta$ SOFA effect reaches $1.44 plus.minus 0.04$, matching the approx scenario and the tightest standard deviation observed across all scenarios, while $sigma$ produces
no reliable separation ($-0.25 plus.minus 0.60$).
Infection and sepsis subgroups follow the same pattern, with strong and stable $beta$ effects and near-zero $sigma$ effects.
The latent space has effectively collapsed to one informative dimension.

#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],
    [High SOFA vs Low SOFA], [$1.44 plus.minus 0.04$], [$-0.25 plus.minus 0.60$],
    [Infection vs No infection], [$0.37 plus.minus 0.06$], [$-0.07 plus.minus 0.12$],
    [Sepsis vs No sepsis], [$0.30 plus.minus 0.05$], [$-0.06 plus.minus 0.13$],
  ),
  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, linear_smooth scenario.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation along $beta$ and $sigma$, averaged over all 25 splits.],
  ),
) <tab:cohens-cv-linear-smooth>

The $beta$ marginal (@fig:cv-stability-linear-smooth) is the most stable seen across all scenarios, the split means are essentially fixed at $approx 0.489$ with negligible variation, though within-split spread remains large (error bars spanning $approx 0.46$–$0.51$).
The $sigma$ marginal shows the usual inter-split variability, given the surface provides no organizing gradient along that axis.

#figure(image("../images/project/variations/linear_smooth/svg_latent_stability.svg"), caption: flex-caption(
  short: [Latent marginal stability across 25 splits, linear_smooth.],
  long: [Mean $plus.minus$ std of $beta$ (left) and $sigma$ (right) per split.
    The $beta$ marginal is the most stable observed across all scenarios; $sigma$ remains variable due to the flat surface along that axis.],
)) <fig:cv-stability-linear-smooth>

The density plots (@fig:cv-dists-linear-smooth) show a consistent near-vertical elongated structure cross all four splits, the patient mass stretches along $sigma$ with a fixed $beta$ center, directly reflecting the $sigma$-flat surface forcing all $beta$ variation to be driven by the supervision signal alone.

#figure(
  image("../images/project/variations/linear_smooth/svg_dists_cv.svg"),
  caption: flex-caption(
    short: [Latent densities for best and worst splits, linear_smooth.],
    long: [Log-scaled latent density contours for the two best- and two worst-performing splits in the linear_smooth scenario.
      All splits show a vertically elongated structure, consistent with the $sigma$-flat surface providing no lateral gradient.],
  ),
) <fig:cv-dists-linear-smooth>


=== Linear Step
Performance is again unchanged from standard on Sepsis-3 (@tab:perf-linear-step), with organ dysfunction detection matching linear_smooth (#acr("AUROC") $77.32 plus.minus 1.10$).
The sharper ramp does not improve over the smooth variant on any metric.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],
    [Sepsis-3], [$84.08 plus.minus 0.77$], [$9.74 plus.minus 1.00$],
    [$Delta$ #acr("SOFA") $>=$ 2], [$77.32 plus.minus 1.10$], [$14.86 plus.minus 0.55$],
    [Infection], [$72.47 plus.minus 1.04$], [$19.99 plus.minus 0.85$],
    [Sepsis-3 using Organ branch only], [$63.94 plus.minus 1.38$], [$2.27 plus.minus 0.20$],
    [Sepsis-3 using Infection branch only], [$83.47 plus.minus 0.79$], [$9.50 plus.minus 1.01$],
  ),
  caption: flex-caption(
    short: [Linear step scenario predictive performance.],
    long: [Predictive performance of the linear_step scenario across 25 cross-validation splits (mean $plus.minus$ std).],
  ),
) <tab:perf-linear-step>

Subgroup separation mirrors linear_smooth closely (@tab:cohens-cv-linear-step): strong, consistent $beta$ #acr("SOFA") effect ($1.38 plus.minus 0.06$) and unreliable $sigma$ effects.
The slightly lower $beta$ #acr("SOFA") $d$ compared to linear_smooth ($1.38$ vs $1.44$) and the larger $sigma$ variance ($-0.44 plus.minus 0.95$) most likely reflects the smaller gradient section in which patients are positioned, and larger neutral regions where the positioning does not influence #acr("SOFA") separation.

#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],
    [High SOFA vs Low SOFA], [$1.38 plus.minus 0.06$], [$-0.44 plus.minus 0.95$],
    [Infection vs No infection], [$0.35 plus.minus 0.06$], [$-0.08 plus.minus 0.20$],
    [Sepsis vs No sepsis], [$0.35 plus.minus 0.06$], [$-0.09 plus.minus 0.19$],
  ),
  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, linear_step scenario.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation along $beta$ and $sigma$, averaged over all 25 splits.],
  ),
) <tab:cohens-cv-linear-step>


The $beta$ stability (@fig:cv-stability-linear-step) is similarly tight to linear_smooth, with means locked at $approx 0.527$ and negligible inter-split drift.

#figure(
  image("../images/project/variations/linear_step/svg_latent_stability.svg"),
  caption: flex-caption(
    short: [Latent marginal stability across 25 splits, linear_step.],
    long: [Mean $plus.minus$ std of $beta$ (left) and $sigma$ (right) per split in the linear_step scenario.],
  ),
) <fig:cv-stability-linear-step>

The density plots (@fig:cv-dists-linear-step) show the same vertically elongated morphology, but the mass sits at a slightly higher $beta$ value than linear_smooth and occupies a narrower $beta$ and, consistent with patients clustering just above the sharp step boundary where
the gradient signal is concentrated.

#figure(
  image("../images/project/variations/linear_step/svg_dists_cv.svg"),
  caption: flex-caption(
    short: [Latent densities for best and worst splits, linear_step.],
    long: [Log-scaled latent density contours for the two best- and two worst-performing splits in the linear_step scenario.
      The vertically elongated morphology is preserved, with the mass positioned at a slightly higher $beta$ than linear_smooth.],
  ),
) <fig:cv-dists-linear-step>

=== Radial
Sepsis-3 performance is unchanged from standard (@tab:perf-radial-modest).
Organ dysfunction detection reaches the highest value across all scenarios (#acr("AUROC") $80.06 plus.minus 1.19$, #acr("AUPRC") $24.26$; organ branch alone #acr("AUROC") $66.40$), suggesting the radially symmetric surface provides a particularly effective latent space.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],
    [Sepsis-3], [$84.07 plus.minus 0.81$], [$9.72 plus.minus 0.82$],
    [$Delta$ #acr("SOFA") $>=$ 2], [$80.06 plus.minus 1.19$], [$24.26 plus.minus 1.70$],
    [Infection], [$73.42 plus.minus 0.87$], [$20.74 plus.minus 0.81$],
    [Sepsis-3 using Organ branch only], [$66.40 plus.minus 1.35$], [$2.40 plus.minus 0.19$],
    [Sepsis-3 using Infection branch only], [$83.64 plus.minus 0.85$], [$9.56 plus.minus 0.85$],
  ),
  caption: flex-caption(
    short: [Radial modest scenario predictive performance.],
    long: [Predictive performance of the radial_modest scenario across 25 cross-validation splits (mean $plus.minus$ std).],
  ),
) <tab:perf-radial-modest>

The subgroup separation picture is, however, dramatically different from all
other scenarios (@tab:cohens-cv-radial-modest).
As the radial latent space does not provide structural guidance to the latent positioning, the resulting latent structures emerge completely independent, and no structural correlation is observed.
#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],
    [High SOFA vs Low SOFA], [$-0.66 plus.minus 1.09$], [$-0.44 plus.minus 1.28$],
    [Infection vs No infection], [$-0.22 plus.minus 0.36$], [$-0.14 plus.minus 0.36$],
    [Sepsis vs No sepsis], [$-0.14 plus.minus 0.29$], [$-0.12 plus.minus 0.27$],
  ),
  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, radial_modest scenario.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation along $beta$ and $sigma$, averaged over all 25 splits.
      The large standard deviations reflect a bimodal split distribution rather than genuine uncertainty.],
  ),
) <tab:cohens-cv-radial-modest>

The latent stability plot (@fig:cv-stability-radial-modest) makes this clear, as the 25 splits bifurcate into well-separated regimes.
Roughly half the splits cluster at $beta approx 0.61$ and $sigma approx 1.07$, while the other half sit at $beta approx 0.50$ and $sigma approx 0.42$–$0.53$.
The radially symmetric surface has equivalent regions at equal distance from the center, and the model converges to one or the other arbitrarily depending on initialization.
Within each regime the latent organization may be consistent, but the sign of subgroup separation flips between them, producing the low mean Cohen's $d$ with inflated variance.

#figure(
  image("../images/project/variations/radial_modest/svg_latent_stability.svg"),
  caption: flex-caption(
    short: [Latent marginal stability across 25 splits, radial_modest.],
    long: [Mean $plus.minus$ std of $beta$ (left) and $sigma$ (right) per split.
      The clear bimodal structure reflects convergence to two equivalent but sign-flipped solutions on the symmetric surface.],
  ),
) <fig:cv-stability-radial-modest>


#figure(image("../images/project/variations/radial_modest/svg_dists_cv.svg"), caption: flex-caption(
  short: [Latent densities for best and worst splits, radial_modest.],
  long: [Log-scaled latent density contours for the two best- and two
    worst-performing splits in the radial_modest scenario.
    Splits place the patient mass in different quadrants of the symmetric surface, reflecting the rotational degeneracy of the latent space.],
)) <fig:cv-dists-radial-modest>

=== Radial Modest Latent Lookup
Performance is again unchanged on Sepsis-3 (@tab:perf-radial-lookup).
Organ dysfunction detection drops to AUROC $77.15 plus.minus 2.04$ relative to the continuous radial scenario ($80.06$), suggesting the discretization slightly degrades the organ branch signal despite the identical underlying surface.

#figure(
  table(
    columns: 3,
    [Metric], [AUROC], [AUPRC],
    [Sepsis-3], [$84.09 plus.minus 0.83$], [$9.81 plus.minus 0.96$],
    [$Delta$ #acr("SOFA") $>=$ 2], [$77.15 plus.minus 2.04$], [$15.25 plus.minus 1.42$],
    [Infection], [$72.56 plus.minus 0.98$], [$20.17 plus.minus 0.89$],
    [Sepsis-3 using Organ branch only], [$65.19 plus.minus 1.83$], [$2.32 plus.minus 0.21$],
    [Sepsis-3 using Infection branch only], [$83.49 plus.minus 0.87$], [$9.54 plus.minus 1.01$],
  ),
  caption: flex-caption(
    short: [radial_modest_latent_lookup scenario predictive performance.],
    long: [Predictive performance of the radial_modest_latent_lookup scenario across 25 cross-validation splits (mean $plus.minus$ std).],
  ),
) <tab:perf-radial-lookup>

Subgroup separation is similarly uninformative as the radial_modest scenario (@tab:cohens-cv-radial-lookup), with all effects near zero and large standard deviations.

#figure(
  table(
    columns: 3,
    [], [$beta$], [$sigma$],
    [High SOFA vs Low SOFA], [$0.16 plus.minus 1.32$], [$-0.16 plus.minus 0.94$],
    [Infection vs No infection], [$0.02 plus.minus 0.31$], [$-0.05 plus.minus 0.39$],
    [Sepsis vs No sepsis], [$0.05 plus.minus 0.28$], [$-0.05 plus.minus 0.24$],
  ),

  caption: flex-caption(
    short: [Cohen's $d$ across 25 splits, radial latent lookup scenario.],
    long: [Mean $plus.minus$ std of Cohen's $d$ for subgroup separation along $beta$ and $sigma$, averaged over all 25 splits.
      As with the continuous radial scenario, large standard deviations reflect a bimodal split distribution.],
  ),
) <tab:cohens-cv-radial-lookup>

The latent stability plot (@fig:cv-stability-radial-lookup) shows the same bimodal regime structure as the continuous radial_modest scenario, but more extreme.
Both two clusters are more widely separated in both $beta$ ($approx 0.50$ vs $approx 0.61$) and $sigma$ ($approx 0.40$–$0.45$ vs $approx 1.07$–$1.12$).
The discretization appears to sharpen the bifurcation by restricting the encoder to a finite set of grid positions, reducing within-split variance for the $sigma$ axis and larger variances for the $beta$ axis, while preserving the rotational degeneracy of the symmetric surface.

#figure(
  image("../images/project/variations/radial_modest_latent_lookup/svg_latent_stability.svg"),
  caption: flex-caption(
    short: [Latent marginal stability across 25 splits, radial latent lookup.],
    long: [Mean $plus.minus$ std of $beta$ (left) and $sigma$ (right) per split.
      The bimodal cluster structure is more pronounced than in the continuous radial scenario, with broader tighter within-cluster spread for $beta$ and tighter spread for $sigma$.],
  ),
) <fig:cv-stability-radial-lookup>

The density plots (@fig:cv-dists-radial-lookup) show the same quadrant bifurcation as the continuous radial scenario, but the patient mass is less compact and confined to a larger region of the grid in each split, consistent with the discretization constraining encoder outputs to less precise grid approximations.

#figure(
  image("../images/project/variations/radial_modest_latent_lookup/svg_dists_cv.svg"),
  caption: flex-caption(
    short: [Latent densities for best and worst splits, radial latent lookup.],
    long: [Log-scaled latent density contours for the two best- and two worst-performing splits in the radial_modest_latent_lookup scenario.
      The bimodal quadrant structure is preserved and but less sharpened relative to the continuous radial scenario.],
  ),
) <fig:cv-dists-radial-lookup>

=== Summary
@tab:variation-summary collects predictive performance across all variation scenarios, with @fig:variation-comparison showing the joint (#acr("AUROC"), #acr("AUPRC")) distributions.

#figure(
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    stroke: .5pt,
    table.header(
      [Scenario],
      table.cell(colspan: 2)[*Sepsis-3*],
      table.cell(colspan: 2)[*$Delta$#acr("SOFA") $>=$ 2*],
      table.cell(colspan: 2)[*Infection*],
      [],
      [AUROC], [AUPRC],
      [AUROC], [AUPRC],
      [AUROC], [AUPRC],
    ),
    text(10pt)[standard],
    text(10pt)[*84.11* $plus.minus$ 0.82],
    text(10pt)[9.88 $plus.minus$ 0.93],
    text(10pt)[65.82 $plus.minus$ 2.47],
    text(10pt)[9.84 $plus.minus$ 1.08],
    text(10pt)[72.57 $plus.minus$ 1.08],
    text(10pt)[20.06 $plus.minus$ 0.90],

    text(10pt)[mlp],
    text(10pt)[84.03 $plus.minus$ 0.77],
    text(10pt)[9.85 $plus.minus$ 1.01],
    text(10pt)[78.28 $plus.minus$ 1.40],
    text(10pt)[22.65 $plus.minus$ 1.20],
    text(10pt)[72.87 $plus.minus$ 1.18],
    text(10pt)[20.40 $plus.minus$ 1.07],

    text(10pt)[surrogate],
    text(10pt)[84.03 $plus.minus$ 0.79],
    text(10pt)[*9.90* $plus.minus$ 0.87],
    text(10pt)[74.41 $plus.minus$ 2.35],
    text(10pt)[16.77 $plus.minus$ 2.14],
    text(10pt)[72.81 $plus.minus$ 1.03],
    text(10pt)[20.25 $plus.minus$ 0.92],

    text(10pt)[approx],
    text(10pt)[84.05 $plus.minus$ 0.83],
    text(10pt)[9.65 $plus.minus$ 1.00],
    text(10pt)[77.98 $plus.minus$ 1.43],
    text(10pt)[19.57 $plus.minus$ 2.08],
    text(10pt)[73.16 $plus.minus$ 0.75],
    text(10pt)[20.53 $plus.minus$ 0.85],

    text(10pt)[linear_smooth],
    text(10pt)[84.05 $plus.minus$ 0.82],
    text(10pt)[9.83 $plus.minus$ 0.96],
    text(10pt)[77.05 $plus.minus$ 1.21],
    text(10pt)[16.61 $plus.minus$ 1.17],
    text(10pt)[72.37 $plus.minus$ 1.31],
    text(10pt)[20.00 $plus.minus$ 0.91],

    text(10pt)[linear_step],
    text(10pt)[84.08 $plus.minus$ 0.77],
    text(10pt)[9.74 $plus.minus$ 1.00],
    text(10pt)[77.32 $plus.minus$ 1.10],
    text(10pt)[14.86 $plus.minus$ 0.55],
    text(10pt)[72.47 $plus.minus$ 1.04],
    text(10pt)[19.99 $plus.minus$ 0.85],

    text(10pt)[radial_modest],
    text(10pt)[84.07 $plus.minus$ 0.81],
    text(10pt)[9.72 $plus.minus$ 0.82],
    text(10pt)[*80.06* $plus.minus$ 1.19],
    text(10pt)[*24.26* $plus.minus$ 1.70],
    text(10pt)[*73.42* $plus.minus$ 0.87],
    text(10pt)[*20.74* $plus.minus$ 0.81],

    text(10pt)[radial_modest\ \_latent_lookup],
    text(10pt)[84.09 $plus.minus$ 0.83],
    text(10pt)[9.81 $plus.minus$ 0.96],
    text(10pt)[77.15 $plus.minus$ 2.04],
    text(10pt)[15.25 $plus.minus$ 1.42],
    text(10pt)[72.56 $plus.minus$ 0.98],
    text(10pt)[20.17 $plus.minus$ 0.89],
  ),
  caption: flex-caption(
    short: [Ablation predictive performance summary.],
    long: [Mean $plus.minus$ std of #acr("AUROC") and #acr("AUPRC") across 25 cross-validation splits for all variation scenarios.
      Values $times 100$.
      Bold entries indicate the best performing ablation for each metric.],
  ),
) <tab:variation-summary>

The Sepsis-3 panel of @fig:variation-comparison is strikingly uniform, all eight ellipses overlap almost perfectly, confirming that no surface substitution meaningfully affects sepsis discrimination.
The standard scenario is not the best performer on this metric; it is simply indistinguishable from the rest.
The $Delta$#acr("SOFA") panel is where the scenarios diverge.
Standard is a clear outlier at the bottom left (#acr("AUROC") $65.82$), isolated from all variations, which cluster between #acr("AUROC") $74$–$80$.
The radial_modest scenario achieves the highest #acr("SOFA") detection and the surrogate the lowest among variations, with the linear and approx scenarios occupying the middle range.
The infection panel shows modest but consistent improvements in most variations over standard, with radial_modest again leading marginally.

#figure(
  image("../images/project/variation_comparison.svg"),
  caption: flex-caption(
    short: [Variation experiment comparison.],
    long: [Joint (#acr("AUROC"), #acr("AUPRC")) distributions across 25 splits for all variation scenarios, shown as scatter points with 95% confidence ellipses.
      Left: Sepsis-3; center: $Delta$SOFA $>=$ 2; right: suspected infection.],
  ),
) <fig:variation-comparison>


The decoder summary (@fig:variation-decoder-summary) shows that reconstruction quality is broadly preserved across all variation scenarios.
The linear-variation rows stand out with a distinctly darker mean values, suggesting that the simpler latent space encourage decoder quality, they are the only variations exceeding the standard model.
Additionally, the radial_modest_latent_lookup exhibits better reconstruction quality, indicating that discretized softmax lookup marginally improves the decoder's ability to recover feature values.
This is possibly because the grid quantization reduces the effective degrees of freedom available to the encoder, forcing tighter feature-to-position associations.
The std panel shows elevated variance for linear_smooth on several mid-range features, consistent with the $sigma$-uninformative surface producing less stable decoder solutions for features that would otherwise load on both axes.

#figure(
  image("../images/project/variation_decoder_summary.svg"),
  caption: flex-caption(
    short: [Decoder reconstruction summary across variation scenarios.],
    long: [Mean (top) and standard deviation across splits (bottom) of decoder reconstruction Pearson $r$ per feature, for all variation scenarios and standard.],
  ),
) <fig:variation-decoder-summary>

The feature alignment heatmap (@fig:variation-alignment-summary) reveals a consistent pattern across the structured scenarios.
Standard, mlp, surrogate, approx, linear_smooth, and linear_step all show the same warm $beta$ columns, and broadly similar $sigma$ alignment patterns.
The two radial scenarios are the exception since their $beta$ and $sigma$ blocks are visually washed out relative to all others, reflecting the rotational degeneracy that prevents consistent feature-axis
assignment across splits.
A notable difference between the linear scenarios and the rest is the near-absence of $sigma$ structure in their alignment rows, this is consistent with a $sigma$-flat surface providing no organizing
gradient along that axis.

#figure(
  image("../images/project/variation_alignment_summary.svg"),
  caption: flex-caption(
    short: [Feature alignment summary across variation scenarios.],
    long: [Mean Pearson $r$ between each input feature and $beta$ (upper block) or $sigma$ (lower block) across 25 splits, for all variation scenarios and standard.
      Feature categories follow are shown in the top legend.],
  ),
) <fig:variation-alignment-summary>

Taken together, the variation results establish three findings.
First, Sepsis-3 prediction is completely surface-agnostic, any smooth function in the latent space is sufficient.
Second, organ dysfunction detection improves substantially when the standard #acr("PNM") lookup is replaced by a differentiable surface.
The standard scenario uses a discrete grid with softmax lookup, which quantizes the encoder output before it reaches the organ branch.
Replacing the lookup with any smooth, differentiable function, whether learned (#acr("MLP")), approximated (surrogate, approx), or analytic (linear, radial), consistently improves $Delta$#acr("SOFA") #acr("AUROC") by 10+ points.
The radial_modest_latent_lookup scenario, which applies the same softmax discretization to the radial surface, confirms this.
Here, the $Delta$#acr("SOFA") #acr("AUROC") drops from $80.06$ to $77.15$ relative to the continuous radial scenario despite an identical underlying function.
Interestingly, the lookup discretization appears beneficial for decoder reconstruction, likely because constraining encoder outputs to a fixed grid forces tighter and more stable feature-to-position associations, a direct trade-off between organ branch gradient quality and representation stability.
Third, interpretable and consistent latent organization requires a surface with directional structure.
The radial scenarios produce high predictive performance but latent spaces that are uninterpretable across runs, while the linear and approximate scenarios recover strong $beta$-based clinical organization at the cost of an uninformative $sigma$ axis.

== External Validation
The results for the external validation with the #acr("eICU") dataset turned out somewhat differently than the ones for the standard scenario on the #acr("MIMIC")-IV dataset (@tab:external shows mean, standard deviation and test statistics across all 25 splits).
While the #acr("LDM") did improve upon the #acr("MIMIC")-IV dataset against all baselines in both metrics, for the #acr("eICU") the Transformer-baseline is the strongest model in terms of #acr("AUROC"), while the #acr("LDM") is the strongest in terms of #acr("AUPRC").
Both of these effects are statistically significant in at $p<0.05$.

#figure(
  table(
    align: (left, right, right, right, right, right, right),
    columns: 7,
    [Model],
    table.vline(stroke: .5pt),
    [AUROC$plus.minus$std],
    [$t_"AUROC"$],
    [$p_"AUROC"$],
    table.vline(stroke: .5pt),
    [AUPRC$plus.minus$std],
    [$t_"AUPRC"$],
    [$p_"AUPRC"$],
    [Reg. Logistic Regression],
    [$71.8 plus.minus 0.3$],
    [29.4751],
    [$<0.001$],
    [$2.9 plus.minus 0.1$],
    [27.9903],
    [$<0.001$],
    [LightGBM], [$69.1 plus.minus 0.3$], [44.95], [$<0.001$], [$3.3 plus.minus 0.1$], [23.37], [$<0.001$],
    [Transformer], [*$77.4 plus.minus 0.2$*], [-2.71], [$0.035$], [$5.1 plus.minus 0.1$], [2.57], [$0.035$],
    [LSTM], [$74.0 plus.minus 0.2$], [17.45], [$<0.001$], [$4.0 plus.minus 0.1$], [15.28], [$<0.001$],
    [TCN], [$76.7 plus.minus 0.1$], [1.47], [$0.154$], [$4.9 plus.minus 0.1$], [4.89], [$<0.001$],
    [GRU], [$76.2 plus.minus 0.1$], [4.50], [$<0.001$], [$4.6 plus.minus 0.1$], [8.35], [$<0.001$],
    table.hline(stroke: .5pt),
    [LDM], [$76.9 plus.minus 0.8$], [--], [--], [*$5.3 plus.minus 0.4$*], [--], [--],
  ),
  caption: flex-caption(
    short: [Performance comparison between the baselines and the #acs("LDM") for the #acs("eICU") cohort.],
    long: [Performance comparison between the @yaib baselines and the #acr("LDM") for the #acr("eICU") cohort. Bold values indicate the best performing model.],
  ),
) <tab:external>

Additional to the training and testing on each individual dataset a zero-shot cross-dataset validation has been performed.
This procedure is done for both directions, the results are named "TrainDataset-TestDataset".
Individual split and mean for #acr("AUROC") and #acr("AUPRC") are shown in @fig:external-comparison, the ellipse indicates the 95% confidence.
The cross-dataset validations perform worse than the homogeneously trained and validated counterparts.
Yet, the #acr("LDM") seems to generalize reasonably when validated on a new dataset, as some degradation is expected since different hospitals have varying measurement and documentation protocols.

#figure(
  image("../images/project/external_comparison.svg"),
  caption: flex-caption(
    short: [External validation comparison.],
    long: [Joint (#acr("AUROC"), #acr("AUPRC")) distributions across 25 splits for the external validation, shown as scatter points with 95% confidence ellipses.
      Left: Sepsis-3; center: $Delta$SOFA $>=$ 2; right: suspected infection.],
  ),
) <fig:external-comparison>

In @fig:external-decoder-summary the decoders reconstruction performance in terms of Pearson's $r$ and in @fig:external-alignment-summary the latent-to-feature alignment are shown, for each in- and cross-dataset validation.
Generally they show the same systematics, with the #acr("eICU") being stronger in urine reconstruction and having slightly higher reconstruction standard deviations.
This suggests, that the #acr("LDM") is able to robustly abstract clinical features, regardless of the underlying dataset.

#figure(
  image("../images/project/external_decoder_summary.svg"),
  caption: flex-caption(
    short: [Decoder reconstruction summary across variation scenarios.],
    long: [Mean (top) and standard deviation across splits (bottom) of decoder reconstruction Pearson $r$ per feature, for all variation scenarios and standard.],
  ),
) <fig:external-decoder-summary>

#figure(
  image("../images/project/external_alignment_summary.svg"),
  caption: flex-caption(
    short: [Feature alignment summary across external validation.],
    long: [Mean Pearson $r$ between each input feature and $beta$ (upper block) or $sigma$ (lower block) across 25 splits, for the external validation.
      Feature categories are shown in the top legend.],
  ),
) <fig:external-alignment-summary>
