#import "../thesis_env.typ": *

= Experiment <sec:experiment>
To assess the potential benefits from embedding the #acl("DNM") into a short-term sepsis prediction system, the #acl("LDM") (see @sec:ldm) was trained and evaluated using real-world medical data.
This chapter presents the complete experimental setup, including the data basis (data source, cohort selection, preprocessing), the prediction task, and all implementation and training details.
Further the chapter will provide details on the implementation and training routine.
To begin with, a short overview of the state of the art of model- and data-driven short-term sepsis prediction systems is given.

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
== Implementation Details <sec:impl>
=== Latent Lookup Implementation <sec:impl_fsq> 
#todo[explain STE (straight through estimation)]
== Metrics (How to validate performance?)
