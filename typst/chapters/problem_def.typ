#import "../thesis_env.typ": *


= Problem definition <sec:problemdef>
This section provides some background on the specific research questions which are investigated in @sec:experiment using the methods introduced in @sec:dnm and @sec:ldm respectively.
As discussed in @sec:sepwhy, there is a substantial need for robust methods to identify patients sepsis onset and overall progression.
This work provides a proof of concept for such a prediction system.

The increasing availability of high-quality medical data, i.e. multiple physiological markers with high temporal resolution, enables both classical statistical and #acr("ML") (including #acr("DL")) methods (see @sec:sota).
While these purely data-driven approaches often achieve acceptable performance but the explainability of the prediction suffers and limits their adoption in clinical practice #todo[cite].

In parallel, recent advances in the field of network physiology have introduced new ways to model physiological systems as interacting subsystems rather than isolated organs @Ivanov2021Physiolome.
The #acr("DNM") introduced in @osc1 and adapted in @osc2, allows for a functional description of organ failure in sepsis and shows realistic system behavior in preliminary analysis.
An in-depth introduction to the #acr("DNM") is provided in @sec:dnm.
But up until now the dynamic model has not yet been verified on real data.
The goal is to investigate how real patients would translate to the model parameters, and how the temporal physiological evolution can be incorporated and if there is a benefit doing so.
// However, this model has not yet been validated against real-world observations, which will be addressed in this work #todo[eher project???].


To summarize, the specific research questions include:
#(
  list(
    [*Usability of the #acr("DNM")*: How and to what extent can the #acr("ML")-determined trajectories of the #acr("DNM") be used for detection and prediction, especially of critical infection states and mortality.],
    [*Comparison with data-based approaches*: How can the model-based predictions be compared with those of purely data-based approaches in terms of predictive power and interpretability.],
  )
)
#TODO[End this]
