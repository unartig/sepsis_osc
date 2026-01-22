#import "../thesis_env.typ": *
#import "../figures/cohort.typ": cohort_fig
#import "../figures/auto_encoder.typ": ae_fig

= Experiment <sec:experiment>
To evaluate whether embedding the #acl("DNM") improves short-term sepsis prediction, the #acl("LDM") (see @sec:ldm) was trained and evaluated using real-world medical data.
This chapter presents the complete experimental setup, including the data basis (data source, cohort selection, preprocessing), the prediction task, and provide details on the implementation and training routine.

== Data <sec:data>
This study relies exclusively on the #acl("MIMIC")-IV database (version 2.3) @johnson2023mimic.
The #acr("MIMIC") database series contains #acr("EHR") information capturing day-to-day clinical routines, including patient measurements, orders, diagnoses, procedures, treatments, and free-text clinical notes.
All included #acr("EHR")s were recorded at Beth Israel Deaconess Medical Center in Boston, America between 2008 and 2022.
Every part of the data has been de-identified and is publicly available to support research in electronic healthcare applications, with special focus on intensive care.
While applications trained on #acr("MIMIC") databases are known to have limited generalization to other data-sources and real-world settings, they remain the default open-data resource for developing sepsis prediction systems @Bomrah2024Review@Rockenschaub2023review.

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
Sepsis-positive patients exhibited notably higher disease severity, with a median maximum #acr("SOFA") score of 5.0 compared to 4.0 in sepsis-negative patients, and substantially higher hospital mortality (26.5% vs 6.6%).
Additionally, the #acr("LOS") of septic patients was significantly longer than for non-septic patients (median 335.1 hours vs 150.3 hours).
The median time to sepsis onset was 13 hours (#acr("IQR") (25%-75%): 8–34).

Both groups were similar in terms of demographic characteristics, including age (median 65 years), sex distribution (approximately 55% male), and weight at admission (median 77.6 kg).
The majority of patients in both groups were white (63.6% overall) and had medical admissions (71.0% overall), though sepsis-positive patients had a higher proportion of medical admissions (84.8% vs 70.2%).


#figure(table(
  columns: 4,
  align: (left, center, center, center),
  
  table.header(
    [*Characteristic*],
    [*All patients*],
    [*SEP-3 positive*],
    [*SEP-3 negative*]
  ),
  table.cell(colspan: 4)[*Demographics*],
  table.hline(stroke:.5pt),
  
  [N],
  [63425 (100.0)],
  [3320 (5.2)],
  [60105 (94.8)],
  
  [Male],
  [35170 (55.5)],
  [1881 (56.7)],
  [33289 (55.4)],  

  [Age at admission],
  [65.0 (53.0–76.0)],
  [65.0 (54.0–76.0)],
  [65.0 (53.0–76.0)],

  [Weight at admission],
  [77.6 (65.1–92.3)],
  [77.6 (65.6–94.0)],
  [77.6 (65.0–92.2)],
  
  table.hline(stroke:.5pt),
  table.cell(colspan: 4)[*Clinical Outcomes*],
  table.hline(stroke:.5pt),

  [#acr("SOFA") median],
  [3.0 (1.0–5.0)],
  [3.0 (1.0–5.0)],
  [3.0 (1.0–5.0)],

  [#acr("SOFA") max],
  [4.0 (2.0–6.0)],
  [5.0 (4.0–8.0)],
  [4.0 (2.0–6.0)],

  [Hospital #acr("LOS") hours],
  [157.7 (92.8–268.9)],
  [335.1 (194.2–548.6)],
  [150.3 (90.9–256.0)],

  [Hospital Mortality],
  [4828 (7.6)],
  [879 (26.5)],
  [3949 (6.6)],

  [Sepsis-3 onset time],
  [-],
  [13.0 (8.0–34.0)],
  [-],
  
  table.hline(stroke:.5pt),
  table.cell(colspan: 4)[*Ethnicity*],
  table.hline(stroke:.5pt),

  [White],
  [40364 (63.6)],
  [2087 (62.9)],
  [38277 (63.7)],

  [Black],
  [5809 (9.2)],
  [262 (7.9)],
  [5547 (9.2)],

  [Asian],
  [721 (1.1)],
  [42 (1.3)],
  [679 (1.1)],

  [Hispanic],
  [630 (1.0)],
  [32 (1.0)],
  [598 (1.0)],

  [Other/Unknown],
  [14924 (23.5)],
  [897 (27.0)],
  [14027 (23.3)],
  
  table.hline(stroke:.5pt),
  table.cell(colspan: 4)[*Admission Type*],
  table.hline(stroke:.5pt),

  [Medical],
  [45009 (71.0)],
  [2817 (84.8)],
  [42192 (70.2)],

  [Surgical],
  [2239 (3.5)],
  [45 (1.4)],
  [2194 (3.7)],

  [Other/Unknown],
  [15200 (24.0)],
  [458 (13.8)],
  [14742 (24.5)],
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
To prevent data leakage, all normalization statistics were computed exclusively from the training split and applied to all partitions.

All features were uniformly resampled to an hourly basis with every trajectory padded to the maximum length of 169 hours, ensuring uniform processing lengths.
Missing data points for dynamic variables were forward-filled using the last known value of the same #acr("ICU") stay.
For missing values without any prior measurement, the training cohort mean is used as fill value instead.
Lastly the input data is augmented by a binary indicator that distinguishes between actual measurements and imputed values.

== Implementation Details <sec:impl>
The #acr("LDM") was implemented in the JAX @jax2018 based Equinox framework @kidger2021equinox.
The following sections present the implementation details for each module along with their respective parameter counts.

The infection indicator module $f_theta_f$ is a single #acr("GRU")-cell with a hidden size of $H_f = 16$, followed by the down-projection layer,.
In total this adds up to $5777$ parameters.


The latent encoder architecture $g_(theta^e_g)$ implements a gated attention mechanism with residual processing.
The input is split into the actual features and the imputation indicator.
A sigmoid-gate weights for the features, effectively learning which features to emphasize.
After gating, both halves are recombined and processed through a three-layer residual network, where each block applies layer-normalization @ba2016layer, #acr("GELU") activation @hendrycks2023gelu, and a linear transformation with residual connections.
With #acr("GELU"):
$
  "GELU"(x) = x Phi(x)
$
where $Phi(x)$ is the cumulative distribution function for gaussian distribution.
The final hidden state passes through another #acr("GELU")-activated layer before being projected into the two outputs $bold(h)_0$ and $bold(z)_0^"raw"$.
The architecture is illustrated in panel *A)* of @fig:ae, in total the encoder has $9213$ parameter.
The rollout module $g_(theta^r_g)$ performs latent space dynamics using a single #acr("GRU")-cell, without a bias term and a hidden size of $H_g=8$, followed by the down projecting layer.
This adds to $2752$ parameter.

The decoder $d_theta_d$ is implemented as a four-layer feed-forward network that progressively up-samples from the latent representation back to the feature dimension, reconstructing only the features, not the imputation indicator.
It uses #acr("GELU") activations between layers to introduce non-linearity.
The architecture is illustrated in panel *B)* of @fig:ae, in total the decoder has $3370$ parameter.

#figure(
  ae_fig,
  caption: [*A)* Shows the initial latent encoder $g_(theta^e_g)$ architecture with feature gating and residual connections. Dashed arrows indicate residual skip connections. *B)* Shows the decoder $d_theta^d$ architecture, reconstructing only the features but not the imputation indicators.]
) <fig:ae>

=== Training Details <sec:train>
The cohort was partitioned at the patient level using a stratified split with a 75/12.5/12.5 ($N=$6342/6342) ratio for training, validation, and test sets respectively, yielding 57082, 6343 and 6343 samples each.
The split was stratified by sepsis status to maintain the 5.2% prevalence ratio across all sets.
To address the strong imbalance between septic and non-septic samples, each training batch has been randomly over-sampled to contain 10% positive samples.

All modules were jointly optimized using the AdamW-optimizer @loshchilov2019adamw.
The first epoch serves as a warm-up phase where the learning rate increases linearly from 0 to $1.75 times 10^(-4)$.
Subsequently, a constant learning rate is maintained for all remaining epochs.
The weight decay $ $lambda$ = 1 times 10^(-4)$ is chosen, with momentum parameters $beta_1=0.9$, $beta_2=0.999$.
The #acr("DNM") latent space was quantized to a $60 times 100$ grid over $beta in [0.4pi, 0.7pi]$ and $sigma in [0.0,1.5]$, with differentiable lookup using a $3times 3$ neighborhood softmax interpolation ($K=9$).


Hyperparameters were manually tuned rather than automatically optimized for two key reasons.
First, the loss function jointly optimizes prediction accuracy and latent space interpretability.
Automated hyperparameter optimization would require reducing these to a single scalar metric, which risks producing models with high predictive performance but degraded latent space interpretability, defeating a core purpose of the #acr("DNM") integration.
Second, exhaustive search over eight loss components and 14 tunable parameters would require hundreds of training runs, which is prohibitive for a proof-of-concept study.
The primary goal is demonstrating feasibility and interpretability rather than achieving state-of-the-art performance.
Full hyperparameter specifications and initial values for learnable parameters are listed in @tab:tparams.
The test set was reserved for final evaluation only and was not involved in hyperparameter tuning.

Training was carried out on a consumer laptop #acr("GPU") (NVIDIA RTX 500 Ada Generation with 4 GB of VRAM) on 32-bit floating-point precision.
The model trained on a batch-size of $256$ and a maximum of $1000$ epochs.
To prevent overfitting, early stopping was employed, where training stops after $30$ consecutive epochs without improvement in validation #acr("AUPRC").
A typical training run to convergence required approximately 40 minutes, with early stopping occurring around epoch 140 to 200.
The final model is selected by the geometric mean between the peaks of #acr("AUROC") and #acr("AUPRC").

#figure(table(
  columns: 4,
  align: (left, left, left, center),
  
  table.header(
    [*Parameter*],
    [*Value*],
    [*Description*],
    [*Reference*],
  ),

  table.cell(colspan: 4)[*Hyperparameter*],
  table.hline(stroke:.5pt),
    [$lambda_"sepsis"$],
    [$100.0$],
    [Weight of $cal(L)_"sepsis"$],
    [@eq:loss],
    
    [$lambda_"inf"$],
    [$1.0$],
    [Weight of $cal(L)_"inf"$],
    [@eq:loss],
    
    [$lambda_"sofa"$],
    [$1 times 10^3$],
    [Weight of $cal(L)_"sofa"$],
    [@eq:loss],
    
    [$lambda_"focal"$],
    [$2.0$],
    [Weight of $cal(L)_"focal"$],
    [@eq:loss],
    
    [$lambda_"diff"$],
    [$10.0$],
    [Weight of $cal(L)_"diff"$],
    [@eq:loss],
    
    [$lambda_"dec"$],
    [$5.0$],
    [Weight of $cal(L)_"dec"$],
    [@eq:loss],
    
    [$lambda_"spread"$],
    [$6 times 10^(-3)$],
    [Weight of $cal(L)_"spread"$],
    [@eq:loss],
    
    [$lambda_"boundary"$],
    [$30.0$],
    [Weight of $cal(L)_"boundary"$],
    [@eq:loss],

    [$tau$],
    [$12$],
    [Radius of causal smoothing],
    [@eq:cs],

    [$k$],
    [$3$],
    [Side length of the latent-lookup kernel],
    [@eq:ll\ @eq:llk],

  table.hline(stroke:.5pt),
  table.cell(colspan: 4)[*Learnable Parameter*],
  table.hline(stroke:.5pt),

    [$d$],
    [0.04],
    [#acr("SOFA") increase detection threshold ],
    [@eq:otoa],

    [$s$],
    [50],
    [#acr("SOFA") increase detection sharpness],
    [@eq:otoa],

    [$T_d$],
    [0.05],
    [Lookup interpolation temperature],
    [@eq:ll],

    [$alpha$],
    [0.7],
    [Causal smoothing decay],
    [@eq:cs],
),
caption: flex-caption(
  short: [Hyperparameters and learnable parameters],
  long: [Hyperparameters and initial values for learnable parameters.
  Hyperparameters control training dynamics and loss weighting.
  Learnable parameters are initialized to the listed values and updated during training.]
  )
) <tab:tparams>

== Results and Benchmark Comparisons <sec:results>

// Quantitatively
// Learning curves? Maybe Total loss?
// Lookup Temp
// 
// AUROC & AUPRC  - with plots
// Benchmark comparison, also with params and training?
// Heatmap of which latent areas are mostly used :^)
//
//
// Qualitatively
// How one healthy one unhealthy move over time
// In space and in Values
//
//
//
//
// Discussion
// +++++++
// Competitive performance - DNM seems beneficial
// Already more informative and interpretable predictions
// Small and straight forward
// Modular - change infection module - or add more
// -------
// Sensitivity regarding initializations (random seed)
// Pseudo-interpretability of latent space (we still guess semantics) - problem of projection
//
//
// Outlook
// Latent Space analysis, does the decoder reg really work? Is the gating informative?
// Actually backprop through ODE solver
// More parameters to tune
// Validate on external data
// Incorporate treatment?
// Offline Predictions
