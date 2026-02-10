#import "../thesis_env.typ": *

// Discussion
// +++++++
// Competitive performance - DNM seems beneficial
// Already more informative and interpretable predictions
// Small and straight forward
// Modular - change infection module - or add more
// -------
// Sensitivity regarding initializations (random seed)
// Pseudo-interpretability of latent space (we still guess semantics) - problem of projection
// No real probabilities, "only" heuristic risk scores
//
//
// Outlook
// Latent Space analysis, does the decoder reg really work? Is the gating informative?
// Actually backprop through ODE solver
// More parameters to tune
// Validate on external data
// Incorporate treatment?
// Offline Predictions

#reset-acronym("DNM")
#reset-acronym("LDM")
#reset-acronym("EHR")

= Discussion <sec:disc>

This work introduced the #acr("LDM"), a novel architecture that embeds the #acr("DNM") as an interpretable latent space for short-term sepsis prediction from #acr("EHR").
By combining physics-inspired structure with data-driven learning, this hybrid approach demonstrates competitive predictive performance while offering enhanced interpretability compared to conventional models.
This chapter discusses the key findings in relation to the research questions posed in @sec:problemdef, addresses limitations of the current implementation, and outlines directions for future research.

== Addressing the Research Questions

This work was motivated by two central questions regarding the clinical utility of the #acr("DNM"):
#quote(block: true)[
  _*1) Usability of the #acr("DNM")*: How and to what extent can #acr("ML")-determined trajectories of the #acr("DNM") be used for detection and prediction, especially of critical infection states?_

  _*2) Comparison with data-based approaches*: How can the model-based predictions be compared with those of purely data-based approaches in terms of predictive power?_
]

Experimental results provide encouraging answers to both questions.
Regarding usability, it was demonstrated that #acr("DNM") parameters ($beta$, $sigma$) can be inferred from clinical #acr("EHR") time-series data, and that trajectories through this parameter space capture clinically meaningful disease progression.
Systematic correspondence between latent positions and clinical severity suggests that #acr("DNM") parameters learned from data do correlate with disease trajectory, support the physiological relevance of the model.

Regarding comparison with data-driven approaches, #acr("LDM") achieved an #acr("AUROC") of $aurocp%$ and #acr("AUPRC") of $auprcp%$, outperforming all baseline models from the #acr("YAIB") benchmark, including the best-performing standard #acr("GRU") ($83.6%$ #acr("AUROC"), and $9.1%$ #acr("AUPRC")).
Critically, this performance was achieved while providing medically interpretable intermediate representations, whereas baseline models operated as complete black boxes.
This suggests that embedding #acr("DNM") structure does not sacrifice predictive power and may actually provide useful inductive biases that facilitate learning clinically relevant patterns.

== Key Strengths

Perhaps the most significant finding is that incorporating physiologically-motivated structure does not compromise predictive accuracy.
Compared to most baseline models which provide only a single risk-score, whereas #acr("LDM") provides multiple clinically interpretable indicators, namely infection likelihood $tilde(I)_t$, organ desynchronization $s^1_t (hat(bold(z)))$, acute deterioration risk $tilde(A)_t$, and overall sepsis risk $tilde(S)_t$.
This richer output enables clinicians to understand _why_ a patient is flagged as high-risk, supporting more informed decision-making.
Importantly, the #acr("LDM") interpretability is not post-hoc rationalization but structurally embedded in the model architecture.
Furthermore, traditionally, predictive models undergo extensive hyperparameter optimization.
The #acr("LDM") parameters were manually tuned with emphasis on maintaining latent space interpretability rather than maximizing performance metrics alone.

As discussed in @sec:sota, explainability in current data-driven sepsis prediction systems predominantly relies on Shapley-value analyses, deriving importance factors of single input features or feature interactions @Stylianides2025Review@Sundararajan2020SHAP.
While valuable, such approaches explain which features influenced a prediction without revealing how those features interact dynamically to produce physiological states.
In contrast, #acr("DNM")-based trajectories show temporal evolution through a space with direct physiological interpretation, while still allowing conventional feature-importance analyses.

This interpretability operates at multiple levels.
At the population level, density plots (@fig:heat_space) reveal that the model systematically organizes patients according to severity.
At an individual patient level, trajectory shapes encode temporal dynamics.
Sharp directional changes correspond to acute clinical events, while gradual curves indicate slower progression or recovery (@fig:traj).
This multi-scale interpretability aligns with clinician preferences identified in @EiniPorat2022.
Survey participants emphasized that "the trend of a patient's trajectory itself should be the prediction target" and expressed preference for "trajectories over plain binary event predictions."

With roughly 21,000 parameters, the entire #acr("LDM") is a relatively small network.
Additionally, due to its modularity into functional clear roles, i.e. $f_theta_f$, $g_theta_g$ and $d_theta_d$, it provides the necessary flexibility to improve on individual aspects of the system in a straight forward manner.

== Limitations and Challenges

While the experimental results are encouraging, several important limitations need careful consideration.
Preliminary experiments revealed notable sensitivity to random seed initialization.
Different random seeds produced models with qualitatively different latent space organizations.
While final predictive performance remained relatively stable ($plus.minus 1%$ #acr("AUROC")), the specific geometric arrangement of patients in parameter space varied considerably.
This sensitivity likely stems from the multi-objective loss function, which creates a complex optimization landscape with multiple local minima, and globally not with a single optimal solution, rather a Pareto frontier of trade-offs.

Additionally, there is most likely no ground truth mapping from a high-dimensional patient state, represented by the #acr("EHR"), to the low dimensional parameter space of the #acr("DNM").
This ambiguity offers infinitely many mappings, while some might be more plausible than others, each training run converges most likely to a different mapping depending on the random seed.
Ultimately, when interpreting results, this variability in solutions should be acknowledged rather than treating a single trained model as definitive.
Ensemble methods aggregating predictions from multiple initializations could improve robustness while quantifying this uncertainty.

As discussed in @sec:problemdef, the #acr("DNM") faces inherent limitations.
Parameters like $beta$ (biological age) and $sigma$ (interaction strength between organ and immune system) do not correlate to any directly observable physiological quantity.
Furthermore, the fully connected topology may not reflect actual organ interaction patterns, and individual oscillators do not correspond to specific biological processes.
Currently, the #acr("LDM") does not directly implement the #acr("DNM") dynamical system, since it does not solve the coupled differential equations at each time step.
Instead, it learns to position patients in a parameter space abstracted from the #acr("DNM"), with the intent (encouraged by loss functions) that these positions correlate meaningfully with true physiological states.
It is learning a projection from high-dimensional #acr("EHR") data into a two-dimensional space.
The desynchronization metric $s^1(bold(z))$ computed from these projected positions may correlate with organ dysfunction without reflecting actual organ-level dynamics.
Experimental results provide some reassurance, the systematic correlation between latent position and #acr("SOFA") scores, the meaningful trajectory patterns in @fig:traj, and the competitive predictive performance all suggest the model has learned clinically relevant structure.
However, direct mechanistic interpretability cannot be claimed.
A more honest characterization might be "physiologically-motivated dimensionality reduction".

Right now, the #acr("LDM") produces risk-scores $tilde(S)_t$ indicating sepsis likelihood, but they have not been calibrated to represent true probabilities of sepsis onset.
This means that the prediction has to move from plain distinction between septic and non-septic to sensitive estimates, how critical patient states are compared to others.
For clinical deployment, well-calibrated probabilities would be essential, which could be achieved with post-training calibration, like simple Platt- or Temperature scaling or more involved calibration techniques @guo2017calibration.
// As noted in @sec:sepwhy, traditional sepsis screening relies on reactive clinical scores like #acr("SOFA"), while automated prediction systems aim to identify patients before organ failure develops.
// However, as highlighted by the meta-analysis in @Alshaeba2025Effect, many alert systems fail to improve patient outcomes, often due to alert fatigue from excessive false positives.

== Future Directions

All experiments relied on the #acr("MIMIC")-IV exclusively.
While this enables direct comparison with #acr("YAIB") benchmarks, it limits generalizability claims.
As noted in @sec:sota, applications trained on single datasources often do not generalize well to other datasources or real world settings.
This is why, external validation on independent datasets is essential to assess whether learned representations transfer across settings, but are out of this scope for this proof-of-concept thesis.
Some performance degradation on external data is expected, but the central question is whether the #acr("DNM")-structured latent space provides robust representations.
If the latent space captures fundamental physiological principles rather than dataset-specific patterns, these representations should be transferable.

The evaluation focused primarily on predictive metrics and trajectory visualization, more rigorous analysis of latent space structure and model behavior could provide deeper insights.
For example systematically assess whether $d_theta_d$ successfully regularizes the latent space, in a way that disentangles clinical concepts in a meaningful way.
If it does not, can it be achieved?
Could this information used to deduce practical patient-individual treatment? 

Furthermore, the encoder $g^e_theta_g^e$ uses sigmoid gates to weight input features.
Analyzing learned gate weights could reveal, which features the model considers most informative for positioning patients in #acr("DNM") space, whether gating patterns differ between septic and non-septic patients or if gates encode any known clinical knowledge.
Lastly quantitative analysis of latent trajectory curvature, velocity, and acceleration could give insights if these geometric properties correlate with clinical severity or outcomes.
Such analyses would either strengthen confidence in the #acr("DNM")s clinical validity or reveal specific weaknesses requiring architectural modification.

The current implementation uses a differential lookup methodology to retrieve the #acr("DNM") synchronization metrics.
A more principled approach would directly integrate the #acr("DNM") system of differential equations:
At each time step, given latent position $bold(hat(z))_t = (beta_t, sigma_t)$, solve this system numerically to obtain $s^1(bold(hat(z))_(t))$.
Modern differentiable #acr("ODE")-solvers, such as diffrax @kidger2022diffrax, enable backpropagation through the integration process, making this approach trainable end-to-end.
This approach would come with a tremendous increase in computational cost, but is the most promising path toward mechanistic and more nuanced interpretability.

This work focused on online prediction, where risk estimates update continuously as new measurements arrive.
As noted in @sec:sota, offline prediction tasks, meaning predicting sepsis risk at a fixed observation time for a specified future horizon $T$, are also clinically relevant.
Because of the #acr("LDM")s modularity, an extension to handle offline predictions is straightforward and would require minimal additions.
This would test whether learned latent dynamics capture sufficient structure to extrapolate forward in time, and whether #acr("DNM")-based trajectories provide better forecasting than purely data-driven alternatives.
Training a model on both online- and offline-prediction might offer cross-benefits for both tasks.

While the #acr("DNM") provides a physiologically motivated latent space, other spaces might also provide interpretable structures, as long as there are regions that could be assigned to healthy and septic patients and some interpretation behind the space dimensions.
Even not biologically motivated but artificially created spaces could be used for the systematic comparison, revealing which structures best capture sepsis pathophysiology, or if its solely the model built around it that drives prediction performance.

Lastly, some general concerns, not specific to the #acr("LDM").
Current sepsis prediction implementations treat patients as passively observed systems, ignoring medical interventions. 
Incorporating treatment variables could improve predictions and enable counterfactual reasoning about intervention timing, though this raises its own methodological challenges.
Furthermore, hourly resampling with forward-fill imputation may obscure rapid deterioration occurring between measurements.
More sophisticated approaches like irregular time series methods could better capture high-frequency dynamics.
