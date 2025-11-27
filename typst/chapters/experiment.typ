#import "../thesis_env.typ": *

= Experiment <sec:experiment>
== Task - Definition of Ins and Outs
== State of the Art <sec:sota>
=== Model Based Methods
=== Data Based Methods
==== Selected Works
== Data
#figure(
  image("../images/yaib_sets.svg", width: 100%),
  caption: [
    Sets of @yaib
  ],
)<fig:sets>
RICU and YAIB use delta_cummin function, i.e. the delta #acr("SOFA") increase is calculated with respect to the lowest observed #acr("SOFA") to this point.
=== MIMIC-III/IV
=== YAIB + (Further) Preprocessing
==== ricu-Concepts
== Implementation Details
=== Latent Lookup Implementation <sec:impl_fsq> 
== Metrics (How to validate performance?)
