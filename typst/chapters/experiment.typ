#import "../thesis_env.typ": *
#import "../figures/cohort.typ": cohort_fig
#import "../figures/auto_encoder.typ": ae_fig

= Experiment <sec:experiment>
To evaluate whether embedding the #acl("DNM") improves short-term sepsis prediction, the #acl("LDM") (see @sec:ldm) was trained and evaluated using real-world medical data.
This chapter presents the complete experimental setup and results, including data basis (data source, cohort selection, preprocessing in @sec:data), the prediction task, and implementation and training details in @sec:impl and @sec:train.
Finally, the chapter will end with the discussion of experimental results using the previously established metrics #acr("AUROC") and #acr("AUPRC"), along with qualitative analyses of individual patient trajectories through the #acr("DNM") parameter space.

== Data <sec:data>
This study relies exclusively on the #acl("MIMIC")-IV database (version 2.3) @johnson2023mimic.
#acr("MIMIC") database series contains #acr("EHR") information capturing day-to-day clinical routines, including patient measurements, orders, diagnoses, procedures, treatments, and free-text clinical notes.
All included #acr("EHR")s were recorded at Beth Israel Deaconess Medical Center in Boston, America between 2008 and 2022.
Every part of the data has been de-identified and is publicly available to support research in electronic healthcare applications, with special focus on intensive care.
While applications trained on #acr("MIMIC") databases are known to have limited generalization to other data-sources and real-world settings, they remain the default open-data resource for developing sepsis prediction systems @Bomrah2024Review@Rockenschaub2023review.

=== Cohort Definition, Feature Choice and Preprocessing
To derive a cohort from raw data and preprocess clinical features, the #acr("YAIB") framework is employed @yaib.
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
  caption: [Cohort selection and exclusion process.],
)<fig:cohort>

=== Cohort Characteristics
@tab:cohort presents the demographic and clinical characteristics of the final cohort, stratified by sepsis status according to the Sepsis-3 criteria.
Of the $63,425$ patients included, $3,320$ (5.2%) met the criteria for sepsis.
Sepsis-positive patients exhibited notably higher disease severity, with a median maximum #acr("SOFA") score of 5.0 compared to 4.0 in sepsis-negative patients, and substantially higher hospital mortality (26.5% vs 6.6%).
Additionally, septic patients had significantly longer #acr("LOS") than non-septic patients (median 335.1 hours vs 150.3 hours).
Median time to sepsis onset was 13 hours (#acr("IQR") (25%-75%): 8–34).

Both groups were similar in terms of demographic characteristics, including age (median 65 years), sex distribution (approximately 55% male), and weight at admission (median 77.6 kg).
Most patients in both groups were white (63.6% overall) and had medical admissions (71.0% overall), though sepsis-positive patients had a higher proportion of medical admissions (84.8% vs 70.2%).


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
To enable direct result comparisons with @yaib benchmark, their feature set is adopted, which has been derived in collaboration with clinical experts and includes only widely available clinical markers.
Each patient in the final cohort has 52 input-features, with four static variables (age, height, and weight at admission as well as sex) and 48 dynamic time-series variables.
Dynamic variables combine seven vital signs and 39 laboratory tests, and two additional measurements (fraction of inspired oxygen and urine output).
@tab:concepts provides a complete listing of all features with their value ranges, units of measurement and clinical descriptions.
Target variables additionally include #acr("SOFA")-score, a #acr("SI") label, in contrast to #acr("YAIB") where only the Sepsis-3 label is used (see @sec:formal).

==== Preprocessing
Data preprocessing involves three main steps: scaling, sampling, and imputation of features.
Theses steps were again adopted from #acr("YAIB").
All numerical features were standardized to zero mean and unit variance, while categorical and binary features remained left unchanged.
To prevent data leakage, all normalization statistics were computed exclusively from the training split and applied to all partitions.

All features were uniformly resampled to an hourly basis with every trajectory padded to the maximum length of 169 hours, ensuring uniform processing lengths.
Padded time-points were masked out for training and evaluation.
Missing data points for dynamic variables were forward-filled using the last known value of the same #acr("ICU") stay.
For missing values without any prior measurement, the training cohort mean is used as fill value instead.
Lastly the input data is augmented by a binary indicator that distinguishes between actual measurements and imputed values.

== Implementation Details <sec:impl>
Implementation of the #acr("LDM") was done in the JAX @jax2018 based Equinox framework @kidger2021equinox.
Following sections present the implementation details for each module along with their respective parameter counts.

The infection indicator module $f_theta_f$ is a single #acr("GRU")-cell with a hidden size of $H_f = 32$, followed by the down-projection layer, in total this adds up to $13,249$ parameters (component-specific parameter counts are listed in @tab:paramcount of the @a:paramcount).

The latent encoder architecture $g_(theta^e_g)$ implements a gated attention mechanism with residual processing.
Incoming samples of $bold(mu)_t$ are split into the actual features and the imputation indicator.
A sigmoid-gate weights the features, effectively learning which features to emphasize.
After gating, both halves are recombined and processed through a three-layer residual network, where each block applies layer-normalization @ba2016layer:
$
  x' = (x-EE[x])/("Var"[x]) dot gamma + beta
$
where $gamma$ and $beta$ are learnable parameters, with the same dimensionality as $x$.
After normalization, the input is processed by the #acr("GELU") activation @hendrycks2023gelu, and a linear transformation with residual connections.
With #acr("GELU"):
$
  "GELU"(x) = x Phi(x)
$
where $Phi(x)$ is the cumulative distribution function for Gaussian distribution.
The final hidden state passes through another #acr("GELU")-activated layer before being projected into the two outputs $bold(h)_0$ and $bold(z)_0^"raw"$.
The architecture is illustrated in panel *A* of @fig:ae, in total the encoder has $19,350$ parameter.
The rollout module $g_(theta^r_g)$ performs latent space dynamics using a single #acr("GRU")-cell, with a hidden size of $H_g=8$, followed by the down projecting layer.
This adds to $1,344$ parameter.

#figure(
  scale(ae_fig, 80%),
  caption: flex-caption(
  short: [Implementation of the latent encoder and decoder module.],
  long: [*A* Shows the initial latent encoder $g_(theta^e_g)$ architecture with feature gating and residual connections. Dashed arrows indicate residual skip connections. *B* Shows the decoder $d_theta^d$ architecture, reconstructing only the features but not the imputation indicators. #todo[SHAPES]])
) <fig:ae>


The decoder $d_theta_d$ is implemented as a four-layer feed-forward network that progressively up-samples from the latent representation back to the feature dimension, reconstructing only the features, not the imputation indicator.
It uses #acr("GELU") activations between layers to introduce non-linearity.
The architecture is illustrated in panel *B* of @fig:ae, in total the decoder has $3,524$ parameter.

=== Training Details <sec:train>
The cohort was partitioned at the patient level using a stratified split with a 80/10/10 ratio for training, validation, and test sets respectively, yielding $N=$50,740/6,343/6,342 samples.
Splitting was stratified by sepsis status to maintain the 5.2% prevalence ratio across all sets.
The test set was reserved for final evaluation only and was not involved in hyperparameter tuning.
To address the strong imbalance between septic and non-septic samples, each training mini-batch of size $128$ was randomly over-sampled to contain 10% positive samples.

Hyperparameters were manually tuned rather than automatically optimized for two key reasons.
First, the loss function jointly optimizes prediction accuracy and latent space interpretability.
Automated hyperparameter optimization would require reducing these to a single scalar metric, which risks producing models with high predictive performance but degraded latent space interpretability, defeating a core purpose of the #acr("DNM") integration.
Second, exhaustive search over six loss components and 12 total tunable parameters would require hundreds of training runs, which is prohibitive for a proof-of-concept study.
The primary goal is to demonstrate feasibility and interpretability rather than achieving state-of-the-art performance.
@tab:tparams lists full hyperparameter specifications and initial values for learnable parameters.

All modules were jointly optimized using the AdamW-optimizer @loshchilov2019adamw.
The first epoch serves as a warm-up phase where the learning rate increases linearly from $0.0$ to $5 times 10^(-5)$.
Subsequently, a constant learning rate is maintained for all remaining epochs.
For the optimizer configuration, a weight decay of $lambda = 1 times 10^(-4)$ is chosen, with momentum parameters $beta_1=0.9$, $beta_2=0.999$.
#acr("DNM") latent space was quantized to a $60 times 100$ grid over $beta in [0.4pi, 0.7pi]$ and $sigma in [0.0,1.5]$, with differentiable lookup using a $3times 3$ neighborhood softmax interpolation ($K=9$).

Training was carried out on a consumer laptop #acr("GPU") (NVIDIA RTX 500 Ada Generation with 4 GB of VRAM) on 32-bit floating-point precision.
Training lasted a maximum of $1000$ epochs, to prevent overfitting, early stopping was employed, where training stops after $30$ consecutive epochs where both #acr("AUROC") and #acr("AUPRC") have not been improved.
A typical training run to convergence required approximately 40 minutes, with early stopping occurring between epoch 100 to 400.
To evaluate model performance against the held-out test set, the parameter configuration at the geometric mean between the epochs where #acr("AUROC") and #acr("AUPRC") peak is selected as the final model.

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
    
    [$lambda_"sofa"$],
    [$1 times 10^3$],
    [Weight of $cal(L)_"sofa"$],
    [@eq:loss],
    
    [$lambda_"inf"$],
    [$1.0$],
    [Weight of $cal(L)_"inf"$],
    [@eq:loss],
    
    [$lambda_"spread"$],
    [$6 times 10^(-3)$],
    [Weight of $cal(L)_"spread"$],
    [@eq:loss],
    
    [$lambda_"boundary"$],
    [$30.0$],
    [Weight of $cal(L)_"boundary"$],
    [@eq:loss],
    
    [$lambda_"dec"$],
    [$5.0$],
    [Weight of $cal(L)_"dec"$],
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

#todo[justify large auxil weight, why H_g so small]

== Results <sec:results>
Following sections present the experimental results generated by training the #acr("LDM") on the #acr("MIMIC")-IV dataset.
Details on the implementation and hyperparameters are specified in the previous two sections.
Here, the training progression is discussed in @sec:train_pro, quantitative results presented in @sec:quant followed by qualitative analyses in @sec:qual.

=== Training Progression <sec:train_pro>
@fig:losses shows the progression of total loss $cal(L)_"total"$ and each loss component on both the training and validation set, as well as the metrics #acr("AUROC") and #acr("AUPRC") on the validation set.
The total loss $cal(L)_"total"$ rapidly decreases in the first $10$ epochs in both the training and validation set, thereafter, declining more gradually.
For the whole training time, the training and validation curves closely align, indicating stable multitask, showing no signs of overfitting.
Total training loss decreases from $474$ to $79$, while the total validation loss reaches a slightly lower value of $78$.

#TODO[Analyzing component losses $cal(L)_"sofa"$ and $cal(L)_"inf"$ provides further insight into the learning dynamics.
Both validation curves decrease rapidly and stabilize afterwards, indicating the respective modules quickly learn robust representation, namely the alignment of the latent space with organ dysfunction, as well as #acr("SI").

The primary sepsis loss $cal(L)_"sepsis"$ decreases more gradually, reflecting the higher complexity of predicting the conjunction of infection and acute organ failure, but follows a stable downward trend on validation data, in contrast.]


#figure(
  image("../images/losses.svg"),
  caption: flex-caption(
  short: [Progression of training and validation losses.],
  long: [Training and validation curves of the #acr("LDM") showing the evolution of the total loss, task metrics (#acr("AUROC"), #acr("AUPRC")), and all individual loss components.
  The plots illustrate stable multi-objective convergence, early alignment of infection $f_theta_f$ and #acr("SOFA") $g_theta_g$ submodules, and gradual refinement of the final sepsis risk prediction without signs of overfitting.])
) <fig:losses>

Decoder reconstruction loss $cal(L)_"dec"$ exhibits an overall healthy learning progression, again decreasing early and followed by a complete stabilization.
Here, a modest gap between training and validation, i.e. weak generalization, does not negatively affect predictive metrics and the loss can still fulfill main role as a structural regularizer of the latent space.

Spread loss $cal(L)_"spread"$ decreases gradually initially, preventing latent collapse, and later rises slightly, indicating a balanced trade-off between latent diversity and task alignment.
Starting from $0.0$, since no latent points are close to the edges of the parameter space, $cal(L)_"boundary"$ starts increasing gradually over training, as the model carefully learns to approach the space boundaries.

By optimizing these losses, the model improves at predicting the sepsis label from #acr("EHR") histories.
Quantitatively, #acr("AUROC") rises from near random performance (0.5) to $0.84$, with most gains occurring in the first third of training, followed by smaller but consistent improvements.
Similarly, #acr("AUPRC") increases monotonically, which is particularly relevant given the strong class imbalance.
Absence of any late-stage decline in these metrics suggests that the model continues refining clinically meaningful discrimination rather than memorizing training data.

=== Quantitative Results <sec:quant>

This training run was early-stopped after $217$ epochs, with $174$ being the selected epoch, from validation #acr("AUROC") peak at epoch $160$ with value $0.838$ and #acr("AUPRC") peak at epoch $187$ with value $0.096$  .
Evaluating the model on the held-out test set, complete curves of the ROC and precision-recall curves are shown in @fig:areas by sweeping the decision threshold $tau$ from @eq:detect.
Here, the ROC curve exhibits an expected shape, bending toward the top-left corner and away from the chance level curve shown as the dashed line.
An #acr("AUROC") of $auroc$ tells us that the model will assign ground truth sepsis patients a higher risk $tilde(S)_t$ than non-septic patients in $aurocp%$ of the time.

In contrast, the precision-recall curve shows higher precision at lower recall values (corresponding to lower threshold values $tau$), indicating the model struggles to produce confident positive predictions without predicting many false positives.
On average, when model predicts a positive label, it is correct only $auprcp%$ of the time, compared to a chance level of 1% determined by the prevalence of sepsis positive time window (~1% of hours in the cohort with ~5% septic patients).

#figure(
  image("../images/areas.svg"),
  caption: flex-caption(
  short: [Receiver Operating Characteristic and Precision-Recall Curve.],
  long: [Receiver Operating Characteristic and precision-recall curves generated by sweeping the detection threshold $tau$ from @eq:detect.
  ])
) <fig:areas>

To put these results into perspective, @tab:comp compares the #acr("LDM") performance to baseline models trained in #acr("YAIB") @yaib on the same task, data-source, cohort definition and train-test split.
The baseline models include include classical #acr("ML") and #acr("DL") approaches, with hyperparameters tuned via Bayesian optimization, 30 iterations for #acr("ML") models, 50 for #acr("DL") models.
The #acr("LDM") slightly outperforms all baselines in both metrics.

#figure(
  table(
  columns: 3,
  align: (left, center, center),

  table.header(
    [*Model*],
    [*AUROC*],
    [*AUPRC*],
  ),

  table.cell(colspan: 3)[*YAIB*],
  table.hline(stroke: .5pt),

  [Regularized Logistic Regression],
  [77.1],
  [4.6],

  [Light Gradient Boosting Machine],
  [77.5],
  [5.9],

  table.cell(colspan: 3)[],

  [#acl("GRU")],
  [*83.6*],
  [*9.1*],

  [#acl("LSTM")],
  [82.0],
  [8.0],

  [Temporal Convolutional Network],
  [82.7],
  [8.8],

  [Transformer],
  [80.0],
  [6.6],

  table.hline(stroke: .5pt),
  table.cell(colspan: 3)[*This work*],
  table.hline(stroke: .5pt),

  [#acl("DNM")],
  [*$aurocp$*],
  [*$auprcp$*],
  ),
  caption:
  flex-caption(short:[Performance Comparison to Baseline Model],
  long: [Performance Comparison against the mean performance of the baseline models trained in #acr("YAIB") @yaib, in terms of #acr("AUROC") $times 100$ ($arrow.t$, higher is better) and #acr("AUPRC") $times 100$ ($arrow.t$).
  Best performances in *bold*.])
) <tab:comp>

In @fig:heat sample-density plots compare predicted and ground-truth values, demonstrating that the model captures both the magnitude and temporal dynamics of organ dysfunction and infection.
For each variable a diagonal line corresponds to optimal prediction performance.
Predictions for the magnitude in #acr("SOFA")-score follow the diagonal trend across the full severity range.
This is indicating that the model preserves the ordinal structure of organ failure rather than collapsing toward the mean, though here it distributes most of the mass.
While extreme #acr("SOFA") values are slightly smoothed, the overall distribution is well reproduced, suggesting that the latent representation inside the #acr("DNM") parameter space is able to capture clinically meaningful severity information.

For the change in #acr("SOFA") score ($Delta$#acr("SOFA"), of consecutive time-steps), the distribution is strongly centered around zero for both ground truth and predictions, reflecting that most time-steps do not involve large acute changes.
The model captures this concentration as well as the spread toward positive and negative deviations, indicating that it learns not only absolute severity but also the direction and magnitude of temporal deterioration or recovery.
Yet there remains some weight where the ground truth increases but the model predicts a decrease, and vice versa.

Density values of prediction $tilde(I)_t$ vs. ground truth $I_t$ show a mild separation between low and high infection probabilities.
It is highly concentrated around non-infectious ground-truth values, but one can see a staircase pattern starting from correct non-infectious, leading up to infectious predictions.
Together, these results indicate that the models internal representations align well with the clinical variables that define sepsis.
Prediction densities are not shown for the sepsis label $S_t$ since the strong imbalance renders this visualization uninformative even with log scaling.
 

#figure(
  image("../images/heat.svg"),
  caption: flex-caption(
  short: [Receiver Operating Characteristic and Precision Recall Curve.],
  long: [Density plots comparing ground truth and predicted values for #acr("SOFA") score, immediate #acr("SOFA") change ($Delta$#acr("SOFA")), and #acr("SI").
  The model captures the overall #acr("SOFA") severity distribution and its temporal changes, while infection predictions show reasonable separation between low- and high-probability states.
  Color indicates log sample density.])
) <fig:heat>

=== Qualitative Results <sec:qual>

This section investigates, whether the models is able to create plausible patient trajectories inside #acr("DNM") parameter space.
First of all, in panel *A* of @fig:heat_space shows predicted sample density of $bold(z)$ layered over the latent space, where both the smooth low-desynchronized between $beta$ values $0.4$ and $0.5$ and the more dynamic highly-desynchronized area between $beta$ values $0.4$ and $1.0$ are occupied.
The lower half of the latent space is completely unused.
The distribution strongly centers around the point $bold(z)_"center" approx (beta=0.58, sigma=1.00)$, from there, individual trajectories spread out into all directions, mostly ending up in the low-desynchronized.

In panel *B*, the predicted latent positions colored by the ground truth #acr("SOFA")-scores are layered over the latent space.
Since brighter colors for predicted $s^1(bold(z))$ and ground truth correspond to higher desynchronization values, and therefore a more pathologic state of the organ system, ideally the gradients perfectly match.
For the investigated model, this is not the case, but the general systematics align, where the majority of darker shaded trajectories overlap darker areas, and the same for brighter shades.
This is indicating that the model is able to systematically use the #acr("DNM") parameter space to express a patients physiological organ system state.

#figure(
  image("../images/heat_space.png"),
  caption: flex-caption(
  short: [Distribution of latent predictions.],
  long: [*A* shows the distribution of predicted latent points $bold(z)=(z_beta, z_sigma)$ over the latent space.
  The latent space is colored with the values of the normalized desynchronization metric $s^1_(bold(z))$, where brighter values indicate larger desynchronization.
  The point distribution is colored by density, with brighter values having greater density.
  *B* shows the same latent space, but the overlay points are colored by the ground truth #acr("SOFA")-score, here it is desired that the color gradients align.])
) <fig:heat_space>

Moving from the system wide behavior to three individual patient trajectories shown in @fig:traj.
These hand-picked examples illustrate possible physiological progressions and how each is represented inside the #acr("DNM") parameter space.
The figure includes Patient 1 indicated by cyan, Patient 2 by purple and Patient 3 by pink colors.
Their predicted and ground truth #acr("SOFA")-score evolution is shown on the right side of the figure, with the left side showing corresponding latent trajectories colored by the ground truth values.

Patient 1 has at #acr("ICU") admission $t=0$ a relatively low #acr("SOFA")-score of 2 but the organ system quickly deteriorates and the score increases to a final value of 17.
In the beginning, the prediction strongly overestimates the severity of Patient 1's condition, yet it picks up the trend of deterioration and arrives at the same final value.
This mirrors in the latent space, where the points clearly move from darker background-colors (more synchronization, better organ functionality) to brighter ones (less synchronization, worse organ functionality).

Patient 2 arrives in the #acr("ICU") with a similarly low #acr("SOFA") score of 2 followed by a steep increase and a subsequent recovery after roughly 36 hours.
Again the model is able to detect the trend, even though not the full magnitude, and is able to match increase and decrease timings better.
This is reflected visually by the trajectory in the latent space, where it starts by moving towards the brighter region to the right but starts to rotate back into the darker region.

Similarly, Patient 3 arrives with a initial #acr("SOFA")-score of 3, which increases to 4 over one day and returns to the baseline value, the prediction does not detect the slight increase and stays at 3 for the whole time.
The latent trajectory of Patient 3 shortly moves towards the brighter region for the first few steps, but the corresponding prediction value does not increase.
After taking a sharp turn orthogonal to the bright region the trajectory keeps slowly bending away into the darker part.
Even though the prediction value does not indicate the small increase, the latent trajectory provides a visual hint.

Overall these three exemplary patients show that the model can capture diverse physiological trajectories within the #acr("DNM") parameter space.
While the magnitude of predicted #acr("SOFA")-scores sometimes deviates from ground truth, the model consistently captures directional trends, like deterioration, recovery, and stability, through meaningful movement in latent space.
Correspondence between trajectory direction and organ dysfunction severity suggests the model has learned clinically relevant representations that map physiological states to interpretable #acr("DNM") parameters.
However, it should be noted that these hand-picked examples demonstrate typical model behavior, though performance varies across patients.
Some trajectories show better alignment with clinical progression, while others exhibit larger deviations from ground truth, this is somewhat reflected by the deviations in @fig:heat and heterogeneity of trajectories in panel *B* in @fig:heat_space. 


#figure(
  image("../images/trajectory.svg"),
  caption: flex-caption(
  short: [Individual patient trajectories demonstrate representations of clinical trends.],
  long: [
  Left: Three patient trajectories plotted in the latent #acr("DNM") parameter space $(beta, sigma)$, colored by ground truth #acr("SOFA") scores with timestamps marking trajectory progression.
  The background heatmap shows $s^1$ values (right colorbar), with darker regions indicating lower desynchronization (better organ function) and brighter regions higher desynchronization (worse organ function).
  Right: Time series comparing predicted (orange) versus ground truth (blue) SOFA scores for each patient. Time annotations on trajectories indicate beginning and end of temporal progression.
  While prediction magnitudes occasionally diverge from ground truth, the model consistently captures directional trends through systematic latent space navigation.])
) <fig:traj>
