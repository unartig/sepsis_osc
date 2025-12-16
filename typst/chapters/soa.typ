#import "../thesis_env.typ": *
#import "../figures/on_vs_off.typ": oo_fig

= State of the Art <sec:sota>
This chapter provides a brief overview of the current state of the art in automated sepsis prediction and concludes in @sec:comp with some fundamental challenges found when assessing sepsis prediction systems.
Sepsis prediction models for individual patients can be categorized into two major classes, the model-based and the data-driven approaches.

== Model-Based Approaches
Biological and medical inspired models of sepsis offer high interpretability and explainability, since they explicitly encode causal relationships.
However, due to the inherent complexity of sepsis pathophysiology, such mechanistic models remain rare @Schuurman2023Complex.
Most existing works focus on dynamic immune response on a cellular @Relouw2024@An2024Model@Cockrell2022Model@McDaniel2019body, intricate signaling and production mechanisms influenced by varying cell concentration are typically modeled using large systems of coupled differential equations.

To derive risk estimates or disease trajectories, model parameters are fitted to individual patient observations.
By simulating physiological trajectories under hypothetical infection scenarios, these models enable to estimate the likelihood of sepsis development @An2024Model.
More advanced digital twins which incorporate bidirectional feedback between the mechanistic model and patient data have also been explored in @Cockrell2022Model.

Validation mechanistic sepsis models is usually done by comparing simulation trajectories to repetitively measured cellular concentrations.
Since the required high-resolution immunological measurements are difficult and costly to obtain only small samples of patients have been validated.
To date no model-based prediction approaches has yet been evaluated on large-scale clinical datasets, limiting their insight into real-world performance and generalizability.

== Data-Driven Approaches <sec:dda>
With the increasing availability of #acl("EHR") and computational resources, #acr("ML")- and #acr("DL")-methods have become the dominant paradigm for sepsis prediction systems over the last decade.
Data-driven approaches can capture highly nonlinear relationships in heterogeneous clinical data and have demonstrated strong empirical performance in real-world settings.
Numerous systems have been proposed and are summarized in surveys such as @Bomrah2024Review and @Moor2021Review.
The technology-landscape is highly diverse and includes classical #acr("ML")-models (nonlinear regression, random-forests, #acr("SVM")s, and gradient-boosted trees such as XGBoost @XGBoost), as well as #acr("DL")-architectures (standard #acr("MLP"), recurrent models e.g. #acr("GRU")/#acr("LSTM"), attention based e.g. transformers and more specialized models like Gaussian Process Temporal Convolutional Networks @Moor2023Retro.

== Challenges in comparing sepsis prediction models <sec:comp>
Although many of data-driven models report reasonable performance, direct comparison across studies is fundamentally limited, as highlighted by @Moor2021Review.
The main issues particularly relevant to the #acr("LDM") methodology are:

#list([*Prediction Tasks*\
Studies differ fundamentally in how they frame the prediction problem.
This includes _online training_, where newly arriving medical measurements are incorporated into a continuously updated risk estimate and _offline training_, where only the information available at a fixed observation time is used to predict the risk of sepsis within a prespecified horizon $T$.
Because these setups rely on different information structures and temporal assumptions, their reported performances are not directly comparable.
Online prediction is more clinically desirable but also more challenging to implement given the current state of available #acr("EHR").
Both schemes are shown in @fig:oo, notice in offline prediction the horizon $T$, the specific choice strongly influences the outcome, with smaller horizons the tasks becomes gradually easier.
For the online scheme choice of what time range around a sepsis onset qualifies as positive label also significantly changes prediction accuracy and prediction.
#figure(
  scale(oo_fig, 80%),
  caption: flex-caption(
    short: [Offline vs. Online prediction],
    long: [Illustration of the two predictions schemes, _offline_ (*A*) vs. _online_ (*B*) (figure heavily inspired by @Moor2021Review).
    The main difference is the provision and utilization and arrival of observation data.])
) <fig:oo>
To have a meaningful comparisons between two models, they must be evaluated on the _exact same task_.
This includes the scheme (offline or online), the same choice of $T$ for the offline scheme and the labeling window around the measured onset.
], 

[*Sepsis Definition and Implementation*\
Sepsis definitions are vary widely across studies but greatly influence the prevalence, cohort composition as well as task difficulty.
Intuitively different sepsis different sepsis definitions are not comparable with each other since they might capture dissimilar medical concepts.
Even for the same conceptual definition and same dataset differences in implementation can yield different patient cohorts and therefore different prediction performances @Johnsons2018Data.
More restrictive definitions typically produce lower prevalence and greater class imbalances making #acr("ML")-based prediction more difficult but potentially increasing clinical relevance.
Less restrictive definitions can artificially inflate prediction performance while reducing practical applicability.
],

[*Feature selection*\
Feature choice also influences both model performance and real-world usability.
Using a broader set of features can increase predictive accuracy but risks again the clinical applicability, since extensive measurements may be not routinely available.
It also increases the risk of label leakage, where the measurements and medical concepts used to derive the sepsis label are provided to the prediction model as feature input.
This way the model would learn the sepsis derivation but not underlying signals which are actually helpful for early sepsis recognition.
],)


== Summary of State of the Art
As of now, neither purely model-based nor purely data-driven approaches were able to fully address the challenges of sepsis prediction.
Mechanistic models offer strong interpretability and encode physiological priors, yet their practical usefulness is limited by the scarcity of high-resolution immunological measurements and the lack of large-scale clinical validation.
In contrast, data-driven models show strong empirical performance on #acr("EHR") datasets, but their prediction behavior is often difficult to interpret and show black-box behavior

This work, named the #acl("LDM") aims to combine the strengths of both paradigms: mechanistic components of the #acr("DNM") introduce structured physiological biases that can help stabilize the learning process, and provide more interpretable intermediate quantities.
At the same time, the data-driven components allow the model to adapt to real clinical variability and make use of information that is not explicitly captured by the mechanistic structure.
In this way, the #acr("LDM") seeks to make data-driven sepsis prediction models more transparent and more robust to the well-known issues discussed in @sec:comp.

The heterogeneity in prediction tasks, sepsis definitions and feature sets illustrates why each decision of one of these aspects regarding new prediction models should be taken with care.
For the sake of reproducibility the choices need to be reported as precisely as possible.
Works such as #acr("YAIB") @yaib attempt to address these challenges by providing a common basis for evaluating models by standardizing the dataset, cohort definition, prediction task, and labeling strategy, thereby enabling fair and reproducible comparison of different approaches.

After introducing the methodology in @sec:ldm, its performance validated on real clinical data in @sec:experiment.
The experimental setup makes use of the #acr("YAIB") framework, and its dataset, cohort definition, prediction task, and labeling implementation are described in detail in @sec:yaib, as these settings are adopted for training and evaluating the #acr("LDM").
