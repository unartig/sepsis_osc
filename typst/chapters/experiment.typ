#import "../thesis_env.typ": *
#import "../figures/cohort.typ": cohort_fig

= Experiment <sec:experiment>
To assess the potential benefits from embedding the #acl("DNM") into a short-term sepsis prediction system, the #acl("LDM") (see @sec:ldm) was trained and evaluated using real-world medical data.
This chapter presents the complete experimental setup, including the data basis (data source, cohort selection, preprocessing), the prediction task, and provide details on the implementation and training routine.

== Data <sec:data>
This study relies exclusively on the #acl("MIMIC")-IV database (version 2.3) @johnson2023mimic.
The #acr("MIMIC") database series contains #acr("EHR") information capturing day-to-day clinical routines, including patient measurements, orders, diagnoses, procedures, treatments, and free-text clinical notes.
All included #acr("EHR")s were recorded at Beth Israel Deaconess Medical Center in Boston, America between 2008 and 2022.
Every part of the data has been de-identified and is publicly available to support research in electronic healthcare applications, with special focus on intensive care.
While applications trained on #acr("MIMIC") databases are known to have limited generalization to other data-sources and real-world settings, they remain the default open-data resource for developing sepsis prediction systems @Bomrah2024Review @Rockenschaub2023review.

=== Cohort Definition, Feature Choice and Preprocessing
To derive the cohort from raw data and preprocess clinical features, the #acr("YAIB") framework is used @yaib.
#acr("YAIB") standardizes cohort definition, feature derivation and data preprocessing for retrospective #acr("ICU") studies across different publicly available databases.
It additionally provides benchmark results for common #acr("ICU") prediction task, including the online prediction of sepsis.
For this work, every step from the sepsis and cohort definition, feature choices to the data preprocessing, is adopted from their methodology @yaib to enable direct comparison of prediction results.

Their definition closely follows the Sepsis-3 criteria @Sepsis3:
#quote(block:true)[" The onset of sepsis was defined using the Sepsis-3 criteria (Singer et al., 2016), which defines sepsis
as organ dysfunction due to infection. Following guidance from the original authors of Sepsis-3
(Seymour et al., 2016), organ dysfunction was defined as an increase in SOFA score $>=$2 points
compared to the lowest value over the last 24 hours. Suspicion of infection was defined as the
simultaneous use of antibiotics and culture of body fluids. The time of sepsis onset was defined as
the first time of organ dysfunction within 48 hours before and 24 hours after suspicion of infection.
Time of suspicion was defined as the earlier antibiotic initiation or culture request. Antibiotics
and culture were considered concomitant if the culture was requested $<=$24 hours after antibiotic
initiation or if antibiotics were started $<=$72 hours after the culture was sent to the lab. Where available,
antibiotic treatment was inferred from administration records; otherwise, we used prescription data.
To exclude prophylactic antibiotics, we required that antibiotics were administered continuously for
$>=$3 days (Reyna et al., 2019). Antibiotic treatment was considered continuous if an antibiotic was
administered once every 24 hours for 3 days (or until death) or was prescribed for the entire time
spent in the ICU "]

=== Cohort Selection
The cohort includes all adult patients (age at admission $>=$18, $N=73,181$).
To ensure data volume and quality, patients meeting any of the following criteria were excluded:
#list(
// [1) invalid admission or discharge time defined as a missing value or negative calculated #acr("LOS").],
[Less than six hours spent in the #acr("ICU").],
[Less than four separate hours across the entire stay where at least one feature was measured.],
[Any time interval of $>=$12 consecutive hours throughout the stay during which no feature was measured.],
[Sepsis onset before the 6th hour in the #acr("ICU").])
Applying these criteria resulted in a final cohort size of $N=63,425$ patients.
The selection process with corresponding exclusion numbers is shown in @fig:cohort.


#figure(
  scale(cohort_fig, 75%),
  caption: [Cohort selection and exclusion process],
)<fig:cohort>

=== Cohort Characteristics
Table @tab:cohort presents the demographic and clinical characteristics of the final cohort, stratified by sepsis status according to the Sepsis-3 criteria.
Of the 63,425 patients included, 3,320 (5.2%) met criteria for sepsis.
Sepsis-positive patients exhibited notably higher disease severity, with a median maximum SOFA score of 5.0 compared to 4.0 in sepsis-negative patients, and substantially higher hospital mortality (26.5% vs 6.6%).
The median time to sepsis onset was 13.0 hours (#acr("IQR") (25%-75%): 8.0–34.0).

Both groups were similar in terms of demographic characteristics, including age (median 65 years), sex distribution (approximately 55% male), and weight at admission (median 77.6 kg).
The majority of patients in both groups were white (63.6% overall) and had medical admissions (71.0% overall), though sepsis-positive patients had a higher proportion of medical admissions (84.8% vs 70.2%).


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
caption: flex-caption(short: [Characteristics and demographics of the cohort], long: [Characteristics and demographics of the cohort. Numerical variables are summarized by _median [IQR 25 - 75]_ and numerical variables by incidence (%)])
) <tab:cohort>

==== Feature Choice
To enable direct result comparisons with @yaib benchmark, their feature set is adopted, which has been derived in collaboration with clinical experts.
Each patient in the final cohort has 52 input-features, with four static (age, height, and weight at admission as well as sex) and 48 dynamic time-series variables.
The dynamic variables combine seven vital signs and 39 laboratory tests, and two additional measurements (fraction of inspired oxygen and urine output).
A complete listing of all features with their value ranges, units of measurement and clinical description is provided in @tab:concepts.

The target variables include the #acr("SOFA")-score, a #acr("SI") label and the Sepsis-3 label (see @sec:formal).

==== Preprocessing
The data preprocessing involves three main steps: scaling, sampling, and imputation of features, which again were adopted from @yaib.
All numerical feature were standardized to zero mean and unit variance, while categorical and binary features remained left unchanged.
To prevent data leakage, used statistics from the training split for all data partitions (training, validation, and testing) were used.

All features were uniformly resampled to an hourly basis.
Missing data points for dynamic variables were forward-filled using the last known value of the same #acr("ICU") stay.
For missing values without any prior measurement, the training cohort mean is used as fill value instead.
Lastly the data is augmented by a binary indicator that distinguishes between actual measurements and imputed values.

== Implementation Details <sec:impl>
The #acr("LDM") was implemented in the JAX @jax2018 based Equinox framework @kidger2021equinox and trained on a consumer laptop #acr("GPU").
The cohort was partitioned at the patient level using a stratified split with a 75/12.5/12.5 ratio for training, validation, and test sets respectively.
The split was stratified by sepsis status to maintain the 5.2% prevalence ratio across all sets.
To address the strong imbalance between septic and non-septic samples, each training batch has been randomly oversampled to contain 10% positive samples.

All modules were jointly optimized using AdamW (learning rate $ = 3 times 10^(-3)$, weight decay $ = 1 times 10^(-4)$) with early stopping (patience=#todo[...] epochs on validation #acr("AUPRC")).
Batch size was 512 with [sequence handling details].
The #acr("DNM") latent space was quantized to a $60 times 100$ grid over $beta in [0.4pi, 0.7pi]$ and $sigma in [0.0,1.5]$, with differentiable lookup using 3x3 neighborhood softmax interpolation.

#TODO[padding]

Architecture specifications:
#list(
[The infection indicator module is a single single #acr("GRU") cell with a hidden dimension of 16, followed by a linear down projection to the single target dimension.],
[The #acr("SOFA") pre-encoder uses #todo[...], and the recurrent module a single #acr("GRU") cell #todo[in methods?] with hidden dimension 4 followed by a linear down-projection to the 2 dimensional latent space.],
[The decoder #todo[...]]
)
Loss weights were:
#TODO[ask]

== Evaluation Metrics <sec:metrics>
The prediction of a patient developing sepsis vs. no sepsis is a binary decision problem based off of the continuous estimated heuristic sepsis risk $S_t$.
Given an estimated risk $S_t$ and a decision threshold $tau in [0, 1]$, the decision if a prediction value counts as septic is given by the rule:
$
  delta(tilde(S)_t) = II (tilde(S)_t > tau)
$
where $II(dot)$ is 1 when the condition is met and 0 otherwise.
For different choices of $tau$ the decision rule can be applied and yield different ratios of:
#align(center, list(
    align(left, [*#acr("TP")* where truth and estimation are 1]),
    align(left, [*#acr("FP")* where truth is 0 and estimation 1]),
    align(left, [*#acr("TN")* where truth and estimation are 0]),
    align(left, [*#acr("FN")* where truth is 1 and estimation 0,]),
  )
)
from these one can calculate the #acr("TPR") (also called sensitivity):
$
  "TPR" = "TP"/("TP" + "FN")
$
and the #acr("FPR"):
$
  "FPR" = "FP"/("TP" + "FN")
$
Sweeping the decision boundary $tau$ from 0 to 1 and plotting the corresponding implicit function #acr("TPR") vs #acr("FPR") creates the _receiver operating characteristic_ or _ROC_ curve.
A prediction system operating at chance will have exhibit a diagonal line from (0 #acr("FPR"), 0 #acr("TPR")) to (1 #acr("FPR"), 1 #acr("TPR")).
Everything above that diagonal indicates better predictions than chance, with an optimal predictor "hugging" the left axis until the point (0 #acr("FPR"), 1 #acr("TPR")) followed by hugging the top axis.
The quality of the whole curve can be summarized to a single number, the area under the curve, called #acr("AUROC"), where larger values $<=1$ indicate better prediction performance.

When trying to predict rare events, meaning sparse positive against lots of negative events the #acr("FPR") can become small and thus little informative.
In these cases one commonly plots the _precision_:
$
  P = "TP" / ("TP" + "FP")
$
against the _recall_:
$
  R = "TP" / ("TP" + "FN")
$

creating the _precision recall curve_ or PRC, where an optimal predictor hugs the top right.
Also this curve can be summarized by its area to the #acr("AUPRC") metric, where larger values indicate better performance @murphy2012machine.

Traditionally, the #acr("AUPRC") is referred to as the more appropriate metric for imbalanced prediction tasks.
Though, recent research suggests the #acr("AUROC") as a more reliable metric for use cases with elevated #acr("FN") costs, such as the increased mortality risk in false or delayed sepsis diagnoses @mcdermott2025metrics.
Together the #acr("AUROC") and #acr("AUPRC") are the commonly reported performance metrics used in the sepsis prediction literature @Moor2021Review and will be used to compare the performance between the #acr("LDM") and the #acr("YAIB") baseline.




For binary decision problems such as the prediction of 
Model performance was evaluated using threshold-independent metrics appropriate for imbalanced binary classification:
- #acr("AUROC") Measures discrimination ability across all classification thresholds
- #acr("AUPRC") More informative than AUROC for imbalanced data, emphasizes performance on positive (sepsis) cases
All metrics were computed on the held-out test set. Statistical significance of performance differences was assessed using [bootstrapping / DeLong test / McNemar test].


== Results and Benchmark Comparisons
