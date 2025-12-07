#import "../thesis_env.typ": *

= Experiment <sec:experiment>
To assess the potential benefits from embedding the #acl("DNM") into a short-term sepsis prediction system, the #acl("LDM") (see @sec:ldm) was trained and evaluated using real-world medical data.
This chapter presents the complete experimental setup, including the data basis (data source, cohort selection, preprocessing), the prediction task, and all implementation and training details.
Further the chapter will provide details on the implementation and training routine.
To begin with, a short overview of the state of the art of model- and data-driven short-term sepsis prediction systems is given.
Before introducing these elements, a brief overview of the current state of the art in model- and data-driven sepsis prediction is provided. This overview concludes with a detailed discussion of the baseline study used later for quantitative comparison in @sec:results.

== State of the Art <sec:sota>
Prediction models targeting the risk of developing sepsis for individual patients can be differentiated between model-based and data-driven approaches.
Following their short overview, this subsection will discuss fundamental challenges when assessing automated sepsis prediction systems.

=== Model Based Approaches
Compared with data-driven methods, model-based prediction systems remain relatively rare and are primarily centered around digital twins.
Digital twins attempt to mathematically and mechanistically model complex biological processes, such as cell production, signaling, and pathogen-host interactions.
These models are ultimately fit to individual patients and by simulating physiological trajectories under hypothetical infection scenarios, the enable the estimation of sepsis risk @An2024Model@Cockrell2022Model.
However, to my knowledge no model-based prediction approaches has been evaluated on large-scale clinical data-sets, limiting their insight into real-world performance and generalizability.

=== Data-Driven Approaches
With the increasing amount of #acr("EHR") and computational resources, #acr("ML")- and #acr("DL")-methods have become the dominant paradigm for sepsis prediction systems over the last decade.
Numerous data-driven sepsis risk prediction systems have been proposed and reviewed in surveys such as @Bomrah2024Review@Moor2021Review.
The technology-landscape is highly diverse and includes classical #acr("ML")-models (nonlinear regression, random-forests, #acr("SVM")s, and gradient-boosted trees such as XGBoost @Placeholder), as well as #acr("DL")-architectures (standard #acr("MLP"), recurrent models e.g. #acr("GRU")/#acr("LSTM"), attention based e.g. transformers and more specialized models like Gaussian Process Temporal Convolutional networks @Moor2023Retro.

=== Comment on Model Comparisons
Although many of data-driven models report reasonable performance, direct comparison across studies is fundamentally limited, as highlighted by @Moor2021Review.
The main issues particularly relevant to the #acr("LDM") methodology are:

#list([*Prediction Tasks*\
Studies differ fundamentally in how they frame the prediction problem.
This includes _online training_, where newly arriving medical measurements are incorporated into a continuously updated sepsis prediction and _offline training_ that uses the data available at an observation time and predicts the risk of sepsis up until a specified horizon.
Because these setups rely on different information structures and temporal assumptions, their reported performances are not directly comparable.
Online prediction is more clinically desirable but also more challenging to implement given the current state of available #acr("EHR").
Additionally, the specification of what time range around a sepsis onset qualifies as positive learning signal significantly influences precision and sensitivity of predictions.], 

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

The heterogeneity in prediction tasks, sepsis definitions and feature sets illustrates why each decision of one of these aspects regarding new prediction models should be taken with care.
For the sake of reproducibility the final choices need to be reported as precise as possible.
In the following section <sec:yaib>, a specific state-of-the-art study is presented and which is used as the baseline method for later performance comparisons.
The presentation includes its dataset, cohort definition, prediction task, and labeling implementation which are all adopted for the training of the #acr("LDM").

== Baseline <sec:yaib>
=== Data
#figure(
  image("../images/yaib_sets.svg", width: 100%),
  caption: [
    Sets of @yaib
  ],
)<fig:sets>
=== Task
RICU and YAIB use delta_cummin function, i.e. the delta #acr("SOFA") increase is calculated with respect to the lowest observed #acr("SOFA") to this point.
== Implementation Details
=== Latent Lookup Implementation <sec:impl_fsq> 
#todo[explain STE (straight through estimation)]
== Metrics (How to validate performance?)
