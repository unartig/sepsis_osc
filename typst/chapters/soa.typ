#import "../thesis_env.typ": *
#import "../figures/on_vs_off.typ": oo_fig

= State of the Art <sec:sota>
As @sec:sepsis concluded, sepsis represents a critical challenge in modern healthcare, it is both common and deadly, yet hard to diagnose.
This chapter provides a brief overview of the current approaches to automated sepsis prediction and the fundamental challenges when comparing these.

Sepsis prediction models for individual patients can be categorized into two major classes, the model-based and the data-driven approaches, each with their own distinct strengths and limitations.

== Model-Based Approaches
Biologically and medically inspired models of sepsis offer high interpretability and explainability, since they explicitly encode causal relationships.
However, due to the inherent complexity of sepsis pathophysiology, such mechanistic models remain rare @Schuurman2023Complex.

Most existing works focus on dynamic immune response on a cellular level @Relouw2024@An2024Model@Cockrell2022Model@McDaniel2019body.
Complicated signaling and production mechanisms influenced by varying cell concentration are typically modeled using large systems of coupled differential equations.

To derive risk estimates or disease trajectories, model parameters are fitted to individual patient observations.
By simulating physiological trajectories under hypothetical infection scenarios, these models enable estimation of the likelihood of sepsis development @An2024Model.
More advanced digital twins which incorporate bidirectional feedback between the mechanistic model and patient data have also been explored in @Cockrell2022Model.

Mechanistic sepsis models are usually validated by comparing simulation trajectories to repetitively measured cellular concentrations.
Since the required high-resolution immunological measurements are difficult and costly to obtain, only small patient samples have been validated.
To date, no model-based prediction approach has been evaluated on large-scale clinical datasets, limiting the insights into real-world performance and generalizability.

== Data-Driven Approaches <sec:dda>
With the increasing availability of #acl("EHR") and computational resources, #acr("ML") and #acr("DL") methods have become the dominant paradigm for sepsis prediction systems over the last decade.
Data-driven approaches can capture highly nonlinear relationships in heterogeneous clinical data.
Unlike mechanistic models, these methods do not require explicit specification of biological relationships, instead, they learn predictive patterns directly from historical data.

Research on data-driven sepsis prediction systems is highly active, in the past five years alone (2021-2026), six systematic reviews on data-driven sepsis predictions have been published @Bomrah2024Review@Moor2021Review@Yadgarov2024Review@Gao2024Review@Parvin2023Review@Stylianides2025Review.
// (21+ 29 + 73 + 7 + 39 + 11)
The reviews include a total of 180 studies (7 to 73 works per review), proposing over 50 distinct #acr("ML") and #acr("DL") methodologies that range from classical to highly specialized methods.
The following overview is based primarily on findings from @Bomrah2024Review@Moor2021Review@Yadgarov2024Review because, taken together, they provide a very comprehensive and complementary coverage of data-driven sepsis prediction research within the considered time frame.
By focusing on these three reviews, the analysis captures the majority of relevant studies and conclusions while avoiding redundancy.

Studies differ fundamentally in how they frame the prediction problem, most prominently in _online prediction_ versus _offline prediction_.
In online prediction newly arriving medical measurements are incorporated into a continuously updated risk estimate.
In offline prediction only the information available at a fixed observation time is used to predict the risk of sepsis within a pre specified time-horizon $T$.
Because these setups rely on different information structures and temporal assumptions, their reported performances are not directly comparable.

Online prediction is more clinically relevant but also more challenging.
Both schemes are shown in @fig:oo, note that in offline prediction the horizon $T$, the specific choice strongly influences the outcome, with smaller horizons the tasks becomes gradually easier.
For the online scheme, the choice of what time range around a diagnosed sepsis onset qualifies as positive label influences prediction accuracy.
#figure(
  scale(oo_fig, 80%),
  caption: flex-caption(
    short: [Offline and Online prediction],
    long: [Illustration of the two predictions schemes, _offline_ (*A*) vs. _online_ (*B*) (figure heavily inspired by @Moor2021Review).
    The main difference is the sepsis labeling, as well as provision and utilization and arrival of observation data.])
) <fig:oo>

Most models rely on routinely collected clinical data, including vital signs, laboratory measurements, demographics, and treatment variables aggregated and summarized in a #acr("EHR").
Publicly available #acr("ICU") datasets, for example the #acr("MIMIC") series @johnson2023mimic, serve as the predominant development and benchmarking platforms.
Differences in feature selection substantially influence both model performance and real-world usability.
While a broader set of features can increase predictive accuracy but risks again the clinical applicability, if required measurements are not routinely available.
Moreover, extensive feature sets increase the risk of label leakage, where the measurements and medical concepts used to derive the sepsis label are provided to the prediction model as feature input.
This way the model would learn the sepsis derivation but not underlying signals which are actually helpful for early sepsis recognition.
As feature selection is not standardized, the reviewed works deployed feature sets ranging in size from 2 to 100, again emphasizing the heterogeneous nature of the field of research.

Most sepsis prediction models are trained retrospectively and evaluated using offline prediction tasks, typically predicting sepsis onset $T=6–48$ hours in advance.
Model performance is commonly reported using #acr("AUROC") and #acr("AUPRC") (the metric derivation and interpretation is discussed in @sec:metrics).
Across studies, reported #acr("AUROC") values typically range from approximately 0.60 to 0.95, indicating modest to very good performance, though such values must be interpreted cautiously given differences in cohort definition, task formulation and evaluation protocols.
For comparison, classical assessment scores achieve #acr("AUROC")s of #acr("SOFA") $0.667$ and #acr("qSOFA") $0.612$ @Yadgarov2024Review. 

Methodologically, a wide range of supervised learning approaches has been explored.
Classical models such as logistic regression, Cox proportional hazards models, and random forest or gradient boosting remain strong baselines due to their robustness and interpretability.
Deep learning architectures, including #acr("RNN"), temporal convolutional networks, and more recently transformer-based models, have been proposed to capture complex temporal dependencies.
In general, explainability of these predictions predominantly relies on Shapley-values analyses, deriving post-hoc importance factors for single input-features or input-feature interactions @Stylianides2025Review@Sundararajan2020SHAP.

Finally, due to ambiguities in the Sepsis-3 definition, the deployed definitions vary widely across studies and greatly influence prevalence, cohort composition and therefore the task difficulty.
Intuitively, different sepsis definitions are not comparable since they might capture dissimilar medical concepts.
Even for the same conceptual definition and same dataset differences in implementation can yield different patient cohorts and therefore different prediction performances @Johnsons2018Data.
More restrictive definitions typically produce lower prevalence and greater class imbalances making #acr("ML")-based prediction more difficult but potentially increasing clinical relevance.
Less restrictive definitions can artificially inflate prediction performance while reducing practical applicability.
// Since less than 10% of works publish their code for label generation, hurting reproducibility and making comparisons often impossible @Moor2021Review.

Overall the field of research on data-driven sepsis prediction is highly relevant and active.
To date, it has generated numerous heterogeneous methodologies, where most of these works provide proof-of-concepts.
A major challenge remains in the incomparability and lack of standardization in model development and evaluation.
Though, works such as #acr("YAIB") @yaib attempt to address the current challenges by providing a common basis for evaluating models by standardizing the dataset, cohort definition, prediction task, and labeling strategy, thereby enabling fair and reproducible comparison of different approaches.


== Summary
Purely model-based and purely data-driven approaches come with their own sets of strengths and limitations.
Mechanistic models offer strong interpretability and encode physiological priors, yet their practical usefulness is limited by the scarcity of high-resolution immunological measurements and the lack of large-scale clinical validation.
In contrast, data-driven models show strong empirical performance on #acr("EHR") datasets, but their prediction behavior is often difficult to interpret and can show black-box behavior.

This work, therefore, aims to combine the strengths of both paradigms: mechanistic components of a physiologically inspired model is used to introduce structured physiological biases to help the learning process and provide more interpretable intermediate quantities.
At the same time, the data-driven components allow the model to adapt to real clinical variability and make use of information that is not explicitly captured by the mechanistic structure.
In this way, this work novel methodology seeks to make data-driven sepsis prediction models more transparent and potentially more robust.

The heterogeneity in prediction tasks, sepsis definitions and feature sets illustrates why each decision of one of these aspects regarding new prediction models should be taken with care.
For the sake of reproducibility the choices need to be reported as precisely as possible.

After introducing the methodology of this work in @sec:ldm, its performance validated on real clinical data in @sec:experiment.
The experimental setup makes use of the #acr("YAIB") framework, and its dataset, cohort definition, prediction task, and labeling implementation are described in detail in @sec:data, as these settings are adopted for training and evaluating the developed method.
