#import "../thesis_env.typ": *
#import "../figures/cohort.typ": cohort_fig

= Experiment <sec:experiment>
To assess the potential benefits from embedding the #acl("DNM") into a short-term sepsis prediction system, the #acl("LDM") (see @sec:ldm) was trained and evaluated using real-world medical data.
This chapter presents the complete experimental setup, including the data basis (data source, cohort selection, preprocessing), the prediction task, and provide details on the implementation and training routine.

== Data <sec:data>
As a basis the experiments relies exclusively on the #acl("MIMIC")-IV @johnson2023mimic database (version 2.3).
The #acr("MIMIC") database series collects #acr("EHR") information on the day-to-day clinical routines and include patient measurements, orders, diagnoses, procedures, treatments and free-text clinical notes.
Every part of the data has been de-identified and as a whole the data is publicly available to support research in electronic healthcare applications, with special focus in intensive care. 
Even though it is known that applications trained on the #acr("MIMIC") databases do not generalize well to other data-sources and real-world use, they still are the de facto open-data default resource when developing new sepsis prediction systems @Bomrah2024Review @Rockenschaub2023review.


#figure(
  cohort_fig,
  caption: [Cohort selection and exclusion process],
)<fig:cohort>






#figure(
  image("../images/yaib_sets.svg", width: 100%),
  caption: [
    Sets of @yaib
  ],
)<fig:sets>
=== Task
RICU and YAIB use delta_cummin function, i.e. the delta #acr("SOFA") increase is calculated with respect to the lowest observed #acr("SOFA") to this point.
== Implementation Details <sec:impl>
== Metrics (How to validate performance?)

