#import "../thesis_env.typ": *
#import "../figures/cohort.typ": cohort_fig

= Experiment <sec:experiment>
To assess the potential benefits from embedding the #acl("DNM") into a short-term sepsis prediction system, the #acl("LDM") (see @sec:ldm) was trained and evaluated using real-world medical data.
This chapter presents the complete experimental setup, including the data basis (data source, cohort selection, preprocessing), the prediction task, and provide details on the implementation and training routine.

== Data <sec:data>
This study relies exclusively on the #acl("MIMIC")-IV database (version 2.3) @johnson2023mimic.
The #acr("MIMIC") database series collects #acr("EHR") information on the day-to-day clinical routines and include patient measurements, orders, diagnoses, procedures, treatments, and free-text clinical notes.
All of the included #acr("EHR") were recorded in the Beth Israel Deaconess Medical Center in Boston, America, between 2008 and 2022.
Every part of the data has been de-identified and as a whole the data is publicly available to support research in electronic healthcare applications, with special focus in intensive care.
Even though it is known that applications trained on the #acr("MIMIC") databases do not generalize well to both other data-sources and real-world use, they still are the default open-data resource when developing sepsis prediction systems @Bomrah2024Review @Rockenschaub2023review.

=== Cohort Definition and Preprocessing
To derive a cohort from the raw data, as well as the definition, and preprocessing of clinical features, the #acr("YAIB") framework is used @yaib.
#acr("YAIB") seeks to standardize the cohort definition, feature derivation and data preprocessing for retrospective #acr("ICU") studies based off of different publicly available databases.
It additionally provides benchmark results for common #acr("ICU") prediction task, such as the online prediction of sepsis.
For this work, every step from the sepsis and cohort definition to the preprocessed data is adopted from their paper @yaib to enable the comparison of prediction results.

Their definition of sepsis is closely linked to the Sepsis-3 definition @Sepsis3:
#quote(block:true)[The onset of sepsis was defined using the Sepsis-3 criteria (Singer et al., 2016), which defines sepsis
as organ dysfunction due to infection. Following guidance from the original authors of Sepsis-3
(Seymour et al., 2016), organ dysfunction was defined as an increase in SOFA score ≥2 points
compared to the lowest value over the last 24 hours. Suspicion of infection was defined as the
simultaneous use of antibiotics and culture of body fluids. The time of sepsis onset was defined as
the first time of organ dysfunction within 48 hours before and 24 hours after suspicion of infection.
Time of suspicion was defined as the earlier antibiotic initiation or culture request. Antibiotics
and culture were considered concomitant if the culture was requested ≤24 hours after antibiotic
initiation or if antibiotics were started ≤72 hours after the culture was sent to the lab. Where available,
antibiotic treatment was inferred from administration records; otherwise, we used prescription data.
To exclude prophylactic antibiotics, we required that antibiotics were administered continuously for
≥3 days (Reyna et al., 2019). Antibiotic treatment was considered continuous if an antibiotic was
administered once every 24 hours for 3 days (or until death) or was prescribed for the entire time
spent in the ICU]
The cohort includes all adult patients (age at admission $>=$ 18, $N=73,181$), to ensure data volume and quality following patient were excluded:
#list(
// [1) invalid admission or discharge time defined as a missing value or negative calculated #acr("LOS").],
[Less than six hours spent in the #acr("ICU").],
[Less than four separate hours across the entire stay where at least one feature was measured.],
[Any time interval of $>=$ 12 consecutive hours throughout the stay during which no feature was measured.],
[Patients with a sepsis onset before the 6th hour in the #acr("ICU").])
Applying these criteria results in a total cohort size of $N=63,425$ patients, the process with corresponding exclusion numbers can be seen in @fig:cohort.


#figure(
  scale(cohort_fig, 75%),
  caption: [Cohort selection and exclusion process],
)<fig:cohort>


#figure(table(
  columns: 4,
  align: (left, center, center, center),
  stroke: 0.5pt,
  
  table.header(
    [*Characteristic*],
    [*All patients*],
    [*SEP-3 positive*],
    [*SEP-3 negative*]
  ),
  
  [N],
  [63425 (100.0%)],
  [3320 (5.2%)],
  [60105 (94.8%)],
  
  [Male],
  [35170 (55.5%)],
  [1881 (56.7%)],
  [33289 (55.4%)],  

  [Age at admission],
  [65.0 (53.0–76.0)],
  [65.0 (54.0–76.0)],
  [65.0 (53.0–76.0)],

  [Weight at admission],
  [77.6 (65.1–92.3)],
  [77.6 (65.6–94.0)],
  [77.6 (65.0–92.2)],
  
  table.cell(colspan: 4)[*Clinical Outcomes*],

  [SOFA median],
  [3.0 (1.0–5.0)],
  [3.0 (1.0–5.0)],
  [3.0 (1.0–5.0)],

  [SOFA max],
  [4.0 (2.0–6.0)],
  [5.0 (4.0–8.0)],
  [4.0 (2.0–6.0)],

  [Hospital #acr("LOS") hours],
  [157.7 (92.8–268.9)],
  [335.1 (194.2–548.6)],
  [150.3 (90.9–256.0)],

  [Hospital Mortality],
  [4828 (7.6%)],
  [879 (26.5%)],
  [3949 (6.6%)],

  [SEP-3 onset time],
  [-],
  [13.0 (8.0–34.0)],
  [-],
  
  table.cell(colspan: 4)[*Ethnicity*],

  [White],
  [40364 (63.6%)],
  [2087 (62.9%)],
  [38277 (63.7%)],

  [Black],
  [5809 (9.2%)],
  [262 (7.9%)],
  [5547 (9.2%)],

  [Asian],
  [721 (1.1%)],
  [42 (1.3%)],
  [679 (1.1%)],

  [Hispanic],
  [630 (1.0%)],
  [32 (1.0%)],
  [598 (1.0%)],

  [Other/Unknown],
  [14924 (23.5%)],
  [897 (27.0%)],
  [14027 (23.3%)],
  
  table.cell(colspan: 4)[*Admission Type*],

  [Medical],
  [45009 (71.0%)],
  [2817 (84.8%)],
  [42192 (70.2%)],

  [Surgical],
  [2239 (3.5%)],
  [45 (1.4%)],
  [2194 (3.7%)],

  [Other/Unknown],
  [15200 (24.0%)],
  [458 (13.8%)],
  [14742 (24.5%)],
),
caption: flex-caption(short: [], long: []))

#TODO[
  Features/Labels and preprocessing
  Metrics
  Training
  Hyper in Appendix
  Results
]

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

