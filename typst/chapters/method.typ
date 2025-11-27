#import "../thesis_env.typ": *
#import "../figures/high_level.typ": high_fig
#import "../figures/fsq.typ": fsq_fig

= Method (Latent Dynamics Model) <sec:ldm>

This chapter introduces the methodological framework used to address the first research question stated in @sec:problemdef:
#align(
  center,
  [*Usability of the #acr("DNM")*: How and to what extent can the #acr("ML")-determined trajectories of the #acr("DNM") be used for detection and prediction, especially of critical infection states and mortality.#todo[format]],
)

To investigate this, a deep learning pipeline has been developed, in which the #acr("DNM") is embedded as central part.
Instead of predicting the sepsis directly, the two components, #acl("SI") and increase in #acr("SOFA") scores are predicted as direct proxies creating more nuanced and therefore more interpretable results.
For the increase in #acr("SOFA") component, the main idea is to utilize the parameter level synchronization dynamics, particularly $cmbeta(beta)$ and $cmsigma(sigma)$, of the functional #acr("DNM") which are expected to describe organ failure on a systemic level.
The complete architecture, consisting of the #acr("DNM") and additional auxiliary modules, which will be referred to as the #acr("LDM") from now on.

This chapter proceeds with the prediction task to be reiterated formally and the introduction of desired prediction properties, and justification of modeling choices.
Afterwards, the individual modules of the #acr("LDM") will be discussed and focusing on what purpose each serves and how it is integrated into the broader system.
#todo[ref sections]
#todo[Notation table]

== Formalizing the Prediction Task <sec:formal>
In automated clinical prediction systems, a patient is typically represented through their #acr("EHR").
Where the #acr("EHR") aggregates multiple clinical variables, such as laboratory biomarker, for example from blood or urine tests, or physiological scores and, further demographic information, e.g. age and gender.
Using the information in the #acr("EHR"), the objective is to estimate the patient's risk of developing sepsis in the near future.

=== Patient Representation
Let $t$ be an arbitrary chosen time-point of a patients #acr("ICU")-stay and the available #acr("EHR") at that time consisting of $n$ variables.
After imputation of missing values, normalization, and encoding of non-numerical quantities, each variable $mu_j$ is mapped to a numerical value:
$
  mu_(t,j) in RR, " " j = 1,...,n
$
These values are collected into a column-vector:
$
  bold(mu)_t = (mu_(t,1),..., mu_(t,n))^T in RR^n
$
which is fully describing the current physiological state of the #acr("ICU")-patient.

=== Modeling the Sepsis-3 Target
The goal is calculate the risk of patient developing a septic condition given an initial observation $bold(mu)_(t=0)$ in the next $T$ future time-steps.
Following the Sepsis-3 definition, the risk requires both suspected infection and multi-organ failure.
Defining the _sepsis onset event_ $S$ as the occurrence of the Sepsis-3 criteria at any time point within the window $t=0,...,T$:
$
  S_(0:T) := union.big_(t=0)^T (A_t inter I_t)
$

Here $A_t={Delta O_t >= 2}$, is denoting an acute change in organ function, more specifically a worsening of the organ system, i.e. multi organ failure.
With $O_t$ being the #acr("SOFA")-score and $Delta O_t=O_t-O_(t-1)$ the change in #acr("SOFA")-score with respect to the previous time-step.
$I_t$ is an event indicator for a #acl("SI") at time $t$.
The target probability given the current #acr("EHR") is then:
$
  Pr(S_(0:T)|bold(mu)_0) = Pr(union.big^T_(t=0)(A_t inter I_t) | bold(mu)_0)
$
=== Heuristic Scoring and Risk Estimation <sec:heu>
The direct estimation of the conditional probability $Pr(S_(0:T)|bold(mu))$ is computationally and statistically challenging due to the temporal dependency between the binary Sepsis-3 criteria.
To make the prediction of this probability more tractable but still connect the statistical model to the clinical definition the following assumptions and modeling choices are made resulting in a _heuristic risk score_ $tilde(S)$:

#list(
[*Independence between infection and organ failure*\
The strongest assumption is the independence of infection $I_t$ and multi-organ failure $A_t$.
Clinically it is known that a majority of situations with multi-organ failure stem from an underlying infection, meaning they exhibits strong partial correlations.
Yet this assumptions allows to treat both components separately for the prediction and enhances the overall interpretability.],

[*Short-horizon infection stability*\
For small $T$, relative to the total #acr("SI")-window of 72 h, $I_t$ is approximated as constant over the sequence: $tilde(I) approx I_t$ for $t=0,...,T$.
This binary variable serves as a time-invariant proxy for the presence of a #acl("SI") and can be estimated from $bold(mu)_0$.
Purely clinically the presence of an infection is difficult to define generally, and exact starting and ending times complicated to measure.
Once a infection is detected and antibiotic treatment initiated, the patient likely to be infectious for the next couple of hours, but not for the next couple of days.],

[*Temporal independence of organ worsening events*\
The events $A_t$ are statistically independent across time steps.
This is necessary to aggregate the risk across time-points:
$ Pr(A_(0:T)) = 1 - product^T_(t=0)Pr(A_t) $
Instead of predicting $Pr(A_(0:T))$ directly, first the #acr("SOFA")-score for each time-step $hat(O)_t$ is estimated from $bold(mu)$.
These estimated scores are then used to create a non-linear summary statistic $tilde(A)$ that relates to the formula of the probability of a union of events:

$
  tilde(A) = o_(s,d)(hat(O)_(0:T)) =  1 - product^T_(t=1) "sigmoid"(s(hat(O)_t - hat(O)_(t-1) - d))
$
where the learnable parameters $d$ and $s$ of the function $o_(s,d) (dot)$ being a calibration threshold and scale respectively.
In the original Sepsis-3 definition $d$ is chosen as two.
The choice of the $"sigmoid"$ function in the product-sequence ensures monotonicity (larger increase $->$ more likely organ failure) and the aggregation of temporal risks into a single measure.
This risk function is used as a summary statistic for the overall risk of #acr("SOFA")-score increase within the window but is not a strict probability, rather a smoothed approximation.])

The high-dimensional $bold(mu)_0$ has now been condensed into two clinically motivated summary statistics $tilde(A)$ and $tilde(I)$.
The final sepsis risk is then estimated by combining these features treated as independent events:
$
 tilde(S) = S_(0:T) approx tilde(A) tilde(I)
$
The interaction term $tilde(A) tilde(I)$ is essential as the formal Sepsis-3 definition is based of the conjunction of the two events.

It is important to note that $tilde(S)$ is *not a calibrated probability* but a heuristically derived and empirical risk score based on the Sepsis-3 definition, serving as proxy to the real event probability $P(S_(0:T)|bold(mu)_0)$.

== Architecture

To estimate the components $tilde(A)$ and $tilde(I)$ from $bold(mu)_0$ two #acl("DL") modules have been designed.
A flow-chart overview  in @fig:flow summarizes same information provided in @sec:formal but also integrates the different learnable neural modules, indicated by the parameters $theta$.


#figure(
  scale(high_fig, 65%),
  caption: [
    Flow chart of the different steps taken to produce the heuristic sepsis risk measure $tilde(S)$ from an observed #acl("EHR") $bold(mu)_0$.
    Learnable neural function parameters are indicated by a $theta$ subscript.
  ]
) <fig:flow>

After explaining how the #acr("DNM") is used to model the severity of organ failure, which is not directly shown in the flow chart, each #acr("DL")-module is introduced in the following subsections.

=== Infection Indicator Module
The first module of the #acr("LDM") estimates the presence of a #acr("SI"), represented by the binary indicator $tilde(I)$.
Given $N$ pairs of #acr("EHR") vectors and ground truth #acr("SI")-indicator
$
(bold(mu)_i, tilde(I)_i), i = 1...N
$
a parameterized non-linear function
$
f_theta: RR^n -> [0,1]
$
is trained to map the patients physiological state to an estimated probability of suspected infection:

$
  hat(I)=f_theta (bold(mu)_0)
$

The model is implemented as a supervised neural network optimized with gradient descent.
To fit model, the #acr("BCE")-loss which measures the distance between true label $y_i$ and the predicted label $hat(y)_i$:
$
  L_"inf" = B C E (y, hat(y)) = -1/N sum^N_(i=1) [y_i log(hat(y)_i) + (1-y_i)log(1-hat(y)_i)]
$
is minimized during training.
The resulting estimator provides a stable time-invariant proxy for suspected infection over the short prediction horizon.

=== SOFA Predictor Module
The complete #acr("SOFA") predictor module is composed of multiple smaller submodules, described below.
 Most notably it connects the #acr("EHR") with #acr("SOFA") estimations through the #acr("DNM") parameters $cmbeta(beta)$ and $cmsigma(sigma)$, which is described in @sec:theory_fsq.
@sec:theory_enc presents how #acr("EHR") information is embedded into the #acr("DNM") parameter space, and @sec:theory_gru how the evolution of the patient state is modeled.

==== DNM Surrogate <sec:theory_fsq>
As discussed in @sec:heu the summarized organ-condition statistic $tilde(A)$ depends on estimates #acr("SOFA")-scores at future time-steps $hat(O)_(0:T)$.

These scores are obtained through a multi-stage process that integrates the dynamics of the #acr("DNM") with learned patient-specific latent representations.

Recalling that the pathological organ conditions within the #acr("DNM") are characterized by frequency clustering in the parenchymal layer.
The amount of frequency clustering is measured by the ensemble average standard deviation of the mean phase velocity $s^1$ (see @eq:std).
Naturally this measure can be used as a proxy for a patients #acr("SOFA")-score.
Increasing values of $s^1$ indicate a higher #acr("SOFA")-score and a worse condition of the patients organ system.

Numerical integration of the DNM equations for a given parameter pair $(cmbeta(beta), cmsigma(sigma))$ yields the corresponding #acr("SOFA") estimate $hat(O)_t$:
$
  hat(O) = s^1 (cmbeta(beta), cmsigma(sigma))
$
these two parameters where identified as the most influential quantities in the original #acr("DNM") publications @osc2.
Every other parameter is assumed constant and chosen as listed in @tab:init.

In order to massively reduce the computational burden of computing the $s^1$-metric for the continuous space in $cmbeta(beta) in [0.4pi, 0.7pi]$ and $cmsigma(sigma) in [0.0, 1.5]$ the space has been quantized and pre-computed.

For an estimated coordinate pair $hat(z)=(hat(z)_cmbeta(beta), hat(z)_cmsigma(sigma))$ in the continuous $(cmbeta(beta), cmsigma(sigma))$-space the quantized metrics are interpolated by smoothing nearby quantization points with a gaussian kernel, which is illustrated in @fig:fsq.
This allows for a continuous space approximation from the quantized space, while allowing for the pre-computation of the quantized space and therefore drastically reducing the computational expenses.

#figure(
fsq_fig,
caption: [Quantized latent lookup of precomputed synchronization metrics.
Point colors represent the amount of desynchronization.
Neighboring points (the $3times 3$ sub-grid indicated by the red outlines) are used smoothed using a gaussian kernel, represented by the color gradient around estimation point $hat(z)$.
This allows to continuously interpolate the parameter space.
],
) <fig:fsq>

This quantization strategy, called _latent lookup_ is closely related to #acr("FSQ") @Placeholder, used in Dreamer V3 @Placeholder for example.
Details on the latent lookup implementation, including grid-resolution and kernel size, can be found in @sec:impl_fsq.

#todo[explain STE (straight through estimation)]

==== Latent Parameter Encoder <sec:theory_enc>
To connect the high-dimensional #acr("EHR") to the dynamical regime of the #acr("DNM"), a neural encoder:

$
  g_theta: RR^n -> RR^2 times RR^h = RR^(2+h)
$
maps the patient state to a two-dimensional latent vector

$
  hat(z)_0 = (hat(z)_(0,cmbeta(beta)), hat(z)_(0,cmsigma(sigma))) = g_theta (bold(mu)_0)
$
This embedding locates the patient within a physiologically meaningful region of the #acr("DNM") parameter space, which in context of the #acr("LDM") is called the latent space.
The latent coordinate $hat(z)_0$ provides the initial condition for short-term dynamical organ condition forecasting.

Here the latent-lookup introduced in the previous @sec:theory_fsq comes in to play.
Unlike classical #acr("PINN") @Placeholder where the gradients of the #acr("ODE") integration provide useful information to the encoder module directly, here the gradient information is provided by the nearby quantized points which contribute to estimated synchronicity measure through the gaussian smoothing.

In addition to the estimated system parameter $bold(z)_0$, the encoder outputs another vector with dimension $h<<n$ that is a compressed representation of patient physiology relevant for short-term evolution of $hat(bold(z))$.
This vector $bold(h) in RR^h$ is referred to as the hidden state.

Since both output of the function $g_theta$ mark the initial step of the prediction horizon, they receive a $0$ as subscript $hat(bold(z))_0$ and $bold(h)_0$.

The placement of $N$ latent points $bold(z)$ is driven by a supervision signal between the known $O_0$ and the predicted #acr("SOFA")-score $hat(O)_0$
$
  L_"enc" = M S E(bold(O)_0, bold(hat(O))_0) = 1/N sum^N_(i=1) (O_(0,i) - hat(O)_(0,i))^2
$
with #acr("MSE") as the loss function.

==== Recurrent Parameter Dynamics <sec:theory_gru>
Since the heuristic #acr("SOFA") risk $tilde(A)$ depends on the evolution of organ function, it is necessary to estimate not only the initial state $hat(z)_0$ but also its evolution.
For this purpose a neural recurrent function:

$
  hat(bold(z))_t, bold(h)_t = r_theta (bold(z)_(t-1), bold(h)_(t-1)), "  " t = 1,...,T
$
is trained to propagate the latent #acr("DNM") parameters forward in time.
This recurrent mechanism captures how the underlying physiology influences the drift of the DNM parameters, and therefore how the level of synchrony changes across the prediction horizon, which translates to the pathological evolution of patients.

Here again the placement of latent points $bold(z)$ is guided by a supervision signal:
$
  L_"recurrent" = M S E(bold(O)_t, hat(O)_t) = 1/T sum^T_(i=1) (O_t - hat(O)_t)^2
$
and through the #acr("MSE")-loss.

=== Decoder <sec:theory_dec>
As shown in the visualization of the #acr("DNM") phase space in @fig:phase multiple latent coordinates $bold(z)$ result in the same amount of desynchronization, which is not surprising, since different physiological states share the same #acr("SOFA") level.
But even though different physiological states have a common #acr("SOFA")-score, their latent representations should be unique.
This should enable to distinguish different triggers of the organ failure inside the latent space, similarly to how it is possible to distinguish the different triggers from the #acr("EHR").

In a classical Auto-Encoder @Placeholder setting, to encourage a semantically structured latent space, a decoder module is added as an auxiliary regularization component.
A neural decoder network:

$
  d_theta: RR^2 times [0,1] times [0,1] -> RR^n
$
 attempts to reconstruct the original #acr("EHR") features from the latent representation, the resulting desynchronicity of that latent coordinate and the heuristic risk measures:

$
 hat(bold(mu))_t  = d_theta (hat(bold(z))_(t), tilde(s^1_t), tilde(I))
$
where the gradients only flow through $bold(z)$, the amount of desynchronicity $s^1_t$ and the risk measure $tilde(I)$ are provided as additional information.
This way the decoder only learns to disentangle the latent coordinates in $bold(z)_t$ based on the ground truth $bold(mu)_t$, through a supervised loss:
$
  L_"dec" = M S E(bold(mu)_t, bold(hat(mu))_t) = 1/T sum^T_(i=0) (bold(mu)_t - bold(hat(mu))_t)^2 
$

The formulation is based on the assumption:
$
  bold(mu)_t = hat(bold(mu))_t + epsilon
$
with $epsilon$ the measurement noise, to hold.

This latent regularization is motivated by _Representation Learning_ @Bengio2012Representation and ensures that nearby points in the latent $(cmbeta(beta), cmsigma(sigma))$-space correspond to physiologically similar patient states.
It helps the encoder $f_theta$ to learn a meaningful alignment between #acr("EHR")-derived latent-embeddings and the dynamical #acr("DNM") landscape.

Using this regularization the recurrent predictor $g_theta$ is encouraged to map temporally consecutive to spatially near latent coordinates, since it is expected that consecutive #acr("EHR")s do not exhibit drastic changes.
Leading smooth patient trajectories through the latent space.

== Overall Training Objective


