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
Instead of predicting the sepsis directly, the two components, #acl("SI") and increase in #acr("SOFA") scores are predicted as direct proxies creating more nuanced and more interpretable prediction results.

To predict the increase in #acr("SOFA"), namely the worsening of organ functionality, the main idea is to utilize parameter level synchronization dynamics inside the parenchymal layer of the functional #acr("DNM"), which is expected to model systemic organ failure.
Particularly the parameters $cmbeta(beta)$ and $cmsigma(sigma)$, interpreted as biological age and amount of cellular interaction between immune cells and functional organ cells, are of great interest.
To achieve this, the #acr("DNM") is embedded into a learnable latent dynamical system, where patients are placed into the two-parameter phase space and a recurrent module predicts physiological drift in that space.
Pre-computed #acr("DNM") dynamics give rise to differentiable #acr("SOFA") and #acr("SI") estimates.
The complete architecture, consisting of the #acr("DNM") and additional auxiliary modules, which will be referred to as the #acr("LDM") from now on.

This chapter proceeds in @sec:formal with the prediction task to be reiterated formally and the introduction of desired prediction properties, together with the justification of modeling choices.
Afterwards, in @sec:arch, the individual modules of the #acr("LDM") will be discussed, focusing on what purpose each serves and how it is integrated into the broader system.
#todo[Notation table]

== Formalizing the Prediction Task <sec:formal>
In automated clinical prediction systems, a patient is typically represented through their #acr("EHR").
Where the #acr("EHR") aggregates multiple clinical variables, such as laboratory biomarker, for example from blood or urine tests, or physiological scores and, further demographic information, like age and gender.
Using the information that is available in the #acr("EHR"), the objective is to estimate the patients risk of developing sepsis in the near future.

=== Patient Representation
Let $t$ denote an observation time during a patients #acr("ICU")-stay and the available #acr("EHR") at that time consisting of $n$ variables.
After imputation of missing values, normalization, and encoding of non-numerical quantities, each variable $mu_j$ is mapped to a numerical value:
$
  mu_(t,j) in RR, " " j = 1,...,n
$
These values are collected into a column-vector:
$
  bold(mu)_t = (mu_(t,1),..., mu_(t,n))^T in RR^n
$
, where the superscript $""^T$ denoting a transpose operation.
The vector $bold(mu)_t$ is fully describing the current physiological state of the #acr("ICU")-patient at observation time point $t$.

=== Modeling the Sepsis-3 Target
The goal is derive the risk of patient developing a septic condition in the next $T$ future time-steps given an initial observation $bold(mu)_(t=0)$.
Following the Sepsis-3 definition, the risk requires both suspected infection and multi-organ failure.
Defining the _sepsis onset event_ $S$ as the occurrence of the Sepsis-3 criteria at any time point within the window $t=0,...,T$:
$
  S_(0:T) := union.big_(t=0)^T (A_t inter I_t)
$

Here $A_t={Delta O_t >= 2}$, is denoting an acute change in organ function, more specifically the worsening of the organ system.
With $O_t$ being the #acr("SOFA")-score and $Delta O_t=O_t-O_("base")$ the change in #acr("SOFA")-score with respect to some patient specific baseline #acr("SOFA")-score $O_("base")$.

The event $I_t$ is an indicator for a #acl("SI") at time $t$, it indicates not only the #acr("SI")-onset-time but includes the 48 h before and 24 h after.
Conditioned on the current #acr("EHR") $bold(mu)_0$ the target probability given is then:
$
  Pr(S_(0:T)|bold(mu)_0) = Pr(union.big^T_(t=0)(A_t inter I_t) | bold(mu)_0)
$
=== Heuristic Scoring and Risk Estimation <sec:heu>
The direct estimation of the conditional probability $Pr(S_(0:T)|bold(mu))$ is computationally and statistically challenging due to the temporal dependency between the binary Sepsis-3 criteria.
To make the prediction of this probability more tractable but still connect the statistical estimator to the clinical definition the following assumptions and modeling choices are made.
Importantly, all assumptions result in differentiable approximations of the real events or probabilities, making it possible to learn estimators with gradient descent methods.
This changes the prediction target from a calibrated probability estimate to a _heuristic risk score_ $tilde(S)$:

#list(
[*Independence between infection and organ failure*\
The strongest assumption is the independence of infection $I_t$ and multi-organ failure $A_t$.
Clinically it is known that a majority of situations with multi-organ failure stem from an underlying infection, meaning they exhibits strong partial correlations.
Yet this assumptions allows treating both components separately for the prediction.
Splitting the target into its two components enhances the overall interpretability since each component can also be analyzed on its own.],

[*Short-horizon infection stability*\
For small $T$, relative to the total #acr("SI")-window of 72 h, $I_t$ is approximated as constant over the sequence: $tilde(I) = union.big^T_(t=0) I_t$.
If the patient is suspected to be infectious for at least one time point in the sequence the approximated indicator is will be positive.
This binary variable serves as a time-invariant proxy for the presence of a #acl("SI") and can be estimated from $bold(mu)_0$.

This approximation is mainly motivated by the fact that if an infection is detected and antibiotic treatment initiated, the patient likely stays infectious for the next couple of hours, but not for the next couple of days.
Generally exact starting and ending time points of an infection are difficult to measure, especially in a hospital context.
],

[*Temporal independence of organ worsening events*\
The events $A_t$ are statistically independent across time steps.
This assumption is necessary to aggregate the risk across time-points:

$
Pr(A_(0:T)) = 1 - product^T_(t=0)Pr(A_t)
$
Instead of predicting $Pr(A_(0:T))$ directly, first the #acr("SOFA")-score for each time-step $hat(O)_t$ is estimated from $bold(mu)_0$.
Often times there is no knowledge about the baseline #acr("SOFA")-score $O_("base")$, for example when a patient enters the #acr("ICU") for the first time.
To unify the treatment of sequences where the baseline is either known or unknown the worsening of organ functionality is approximated by instantaneous change between two time steps $hat(O)_t - hat(O)_(t-1)$ instead of relying an actual baseline value.
These $Delta hat(O)_t$ approximations are used to create a non-linear summary statistic $tilde(A)$ that relates to the formula of the probability of a union of events:

$
  tilde(A) = o_(s,d)(hat(O)_(0:T)) =  1 - product^T_(t=1) "sigmoid"(s(hat(O)_t - hat(O)_(t-1) - d))
$
where the learnable parameters $d$ and $s$ of the function $o_(s,d) (dot)$ being a calibration threshold and scale respectively.

// In the original Sepsis-3 definition $d$ is chosen as two.
The choice of the function $"sigmoid"(x)=1/(1+e^-x)$ in the product-sequence ensures monotonicity (larger increase $->$ more likely organ failure) and the aggregation of temporal risks into a single measure, while still being differentiable.
This risk function is used as a summary statistic for the overall risk of #acr("SOFA")-score increase within the window but is not a strict probability, rather a smoothed approximation.])

The high-dimensional $bold(mu)_0$ has now been condensed into two clinically motivated time-invariant summary statistics $tilde(A)$ and $tilde(I)$.
The final sepsis risk is then estimated by combining these features treated as independent events:
$
 S_(0:T) approx tilde(S) = tilde(A) tilde(I)
$
The interaction term $tilde(A) tilde(I)$ is essential as the formal Sepsis-3 definition is based of the conjunction of the two events.

It is important to note that $tilde(S)$ is *not a calibrated probability* but a heuristically derived and empirical risk score based on the Sepsis-3 definition, serving as proxy to the real event probability $P(S_(0:T)|bold(mu)_0)$.

== Architecture <sec:arch>

To estimate the components $tilde(A)$ and $tilde(I)$ from $bold(mu)_0$ multiple #acl("DL") modules have been designed.
Each module is a differentiable function with parameters $theta$ and will be optimized via gradient descent.
A flow-chart overview in @fig:flow summarizes same information provided in @sec:formal but also integrates the learnable neural modules $f_theta, e_theta $ and $r_theta$.


#figure(
  scale(high_fig, 65%),
  caption: [
    Flow chart of the different steps taken to produce the heuristic sepsis risk measure $tilde(S)$ from an observed #acl("EHR") $bold(mu)_0$.
    Learnable neural function parameters are indicated by a $theta$ subscript.
  ]
) <fig:flow>

After explaining how the #acr("DNM") is used to model the severity of organ failure, which is not directly shown in the flow-chart (only a textual hint), each #acr("DL")-module is introduced in the following subsections.

=== Infection Indicator Module
The first module of the #acr("LDM") estimates the presence of a #acr("SI"), represented by the binary indicator $tilde(I)$, the estimation is a continuous value $hat(I) in [0, 1]$.
Given $N$ pairs of #acr("EHR") vectors and ground truth #acr("SI")-indicator
$
(bold(mu)_i, tilde(I)_i), "  " i = 1...N
$
a parameterized non-linear function
$
f_theta: RR^n -> [0,1]
$
is trained to map the patients physiological state to an estimated probability of suspected infection:

$
  hat(I)=f_theta (bold(mu)_0)
$

The model is implemented as a supervised neural network optimized with stochastic gradient descent, throughout training minibatches of size $B$ are sampled for all modules.
To fit the model, #acr("BCE")-loss which measures the distance between true label $tilde(I)$ and the predicted label $hat(I)$:
$
  cal(L)_"inf" = B C E _B (tilde(bold(I)), hat(bold(I))) = -1/B sum^B_(i=1) [I^((i)) log(hat(I)^((i))) + (1-tilde(I)^((i)))log(1-hat(I)^((i)))]
$
is minimized.
The resulting estimator provides a stable time-invariant proxy for suspected infection over the short prediction horizon.

=== SOFA Predictor Module
The complete #acr("SOFA") predictor module is composed two multiple submodules, an encoder $e_theta$ and a recurrent auto-regressive latent predictor $r_theta$, each described below.
Most notably both #acr("DNM") parameters $cmbeta(beta)$ and $cmsigma(sigma)$ to an #acr("SOFA")-score prediction.
@sec:theory_enc presents how the encoder embeds the #acr("EHR") information into the #acr("DNM") parameter space, and @sec:theory_gru how the evolution of the patient state is modeled.
Lastly in @sec:theory_fsq is described how computational cost are significantly reduced by pre-computing the #acr("DNM") parameter space.

==== The DNM as SOFA Surrogate <sec:theory_surro>

// As discussed in @sec:heu the summarized organ-condition statistic $tilde(A)$ depends on estimated #acr("SOFA")-scores at future time-steps $hat(O)_(0:T)$.
Recalling that the pathological organ conditions within the #acr("DNM") are characterized by frequency clustering in the parenchymal layer.
The amount of frequency clustering is quantified by the ensemble average standard deviation of the mean phase velocity $s^1$ (see @eq:std).
Naturally this measure can be used as a proxy for a patients #acr("SOFA")-score.
Since $s^1$ monotonically increases with loss synchrony, it serves as an interpretable and natural surrogate for the #acr("SOFA")-score.
Increasing values of $s^1$ indicate a higher #acr("SOFA")-score and a worse condition of the patients organ system.

Numerical integration of the DNM equations for a given parameter pair $(cmbeta(beta), cmsigma(sigma))$ yields the corresponding #acr("SOFA") estimate $hat(O)$:
$
  hat(O) = s^1 (cmbeta(beta), cmsigma(sigma))
$
these two parameters were identified as highly influential and interpretable quantities in the original #acr("DNM") publications @osc2.
Every other parameter is assumed constant and chosen as listed in @tab:init.
The space spanned by the two parameters is called the _latent space_, coordinate-pairs of that latent space are denoted $bold(z) = (z_cmbeta(beta),cmsigma(sigma))$.
In the following $s^1$ and $hat(O)$ are used synonymously, depending on the context the notation emphasizes the connection to either the #acr("DNM") or the interpretation as #acr("SOFA")-score.

The prediction strategy involves the mapping of individual #acr("EHR") to the latent space, so that the ground truth #acr("SOFA")-aligns with the desynchronization measure of the latent coordinate.
Based off of this initial location (and additional information), the patient will perform a trajectory through the latent space yielding step by step #acr("SOFA")-score $hat(O)_(0:T)$ estimates needed to calculate the summarized organ-condition statistic $tilde(A)$.

==== Latent Parameter Encoder <sec:theory_enc>
To connect the high-dimensional #acr("EHR") to the dynamical regime of the #acr("DNM"), a neural encoder:

$
  e_theta: RR^n -> RR^2 times RR^h = RR^(2+h)
$
maps the patient state to a two-dimensional latent vector

$
  hat(z)_0 = (hat(z)_(0,cmbeta(beta)), hat(z)_(0,cmsigma(sigma))) = e_theta (bold(mu)_0)
$
This embedding locates the patient within a physiologically meaningful region of the #acr("DNM") parameter space, which in context of the #acr("LDM") is called the latent space.
The latent coordinate $hat(z)_0$ provides the initial condition for short-term dynamical organ condition forecasting.

In addition to the estimated system parameter $bold(z)_0$, the encoder outputs another vector with dimension $h<<n$ that is a compressed representation of patient physiology relevant for short-term evolution of $hat(bold(z))$.
This vector $bold(h) in RR^h$ is referred to as the hidden state.

Since both output of the function $e_theta$ mark the initial step of the prediction horizon, they receive a $0$ as subscript $hat(bold(z))_0$ and $bold(h)_0$.

For a minibatch of size $B$, the placement of latent points $bold(z)$ is driven by a supervision signal between the observed $O_0$ and the predicted #acr("SOFA")-score $hat(O)_0$
$
  cal(L)_"enc" = M S E_B (bold(O)_0, bold(hat(O))_0) = 1/B sum^B_(i=1) (O^((i))_0 - hat(O)^((i))_0)^2
$
with a batched #acr("MSE") as the loss function.

==== Recurrent Parameter Dynamics <sec:theory_gru>
Since the heuristic #acr("SOFA") risk $tilde(A)$ depends on the evolution of organ function, it is necessary to estimate not only the initial state $hat(z)_0$ but also its evolution.
For this purpose a neural recurrent function:

$
  hat(bold(z))_t, bold(h)_t = r_theta (bold(z)_(t-1), bold(h)_(t-1)), "  " t = 1,...,T
$
is trained to propagate the latent #acr("DNM") parameters forward in time.
This recurrent mechanism, primed by the hidden stat $bold(h)_0$ and initial latent location $bold(z)_t$, captures how the underlying physiology influences the drift of the DNM parameters.
Depending on the movement in the latent space the level of synchrony changes across the prediction horizon, which translates to the pathological evolution of patients.

Here again the placement of latent points $bold(z)$ is guided by a supervision signal:
$
  cal(L)_"recurrent" = M S E_(B,T) (bold(O)_t, hat(O)_t) = 1/(B T) sum^B_(i=1) sum^T_(t=1) (O^((i))_t - hat(O)^((i))_t)^2
$
and through the batched #acr("MSE")-loss.

==== Latent Lookup <sec:theory_fsq>

Intuitively one would numerically integrate the #acr("DNM") every estimate $hat(bold(z))$ to receive the $s^1$-metric for the continuous space in $cmbeta(beta) in [0.4pi, 0.7pi]$ and $cmsigma(sigma) in [0.0, 1.5]$.
But to massively reduce the computational burden the space has been quantized to a discrete grid and the metric pre-computed for each cell.
Differentiable approximation values are retrieved by using localized soft interpolation 

For an estimated coordinate pair $hat(bold(z))=(hat(z)_cmbeta(beta), hat(z)_cmsigma(sigma))$ in the continuous $(cmbeta(beta), cmsigma(sigma))$-space the quantized metrics are interpolated by smoothing nearby quantization points with a Gaussian-like kernel, which is illustrated in @fig:fsq.

The smoothing is performed by weighting the amount of desynchronicity $s^1_bold(z)'$ of quantized nearby latent points $bold(z)'$ by the euclidean distance to the estimation $hat(bold(z))$.
The nearby points are selected by a quadratic slice around the closest quantized point $tilde(bold(z))$, with $k$ being the sub-grid size:

$
tilde(s)^1_hat(bold(z))=sum_(bold(z)' in cal(N)_(k times k)(tilde(bold(z)))) "softmax"(-(||hat(bold(z))-bold(z)'||^2)/T_d)s^1_bold(z)'
$
with $"softmax"$ for $K=k dot k$ neighboring points being:
$
 "softmax"(bold(x))_j = (e^(x_j))/(sum^K_(k=1) e^(x_k)), "  for " j=1,...,K
$
and:
$
 cal(N)_(k times k)(tilde(bold(z))) = {(tilde(z)_cmbeta(beta) +i dot beta_"step size"),(tilde(z)_cmsigma(sigma) +j dot sigma_"step size") | i,j in {-1, 0, 1} }
$
There has been introduced a learnable temperature parameter $T_d in RR_(>0)$ which controls the sharpness of the smoothing, with larger values producing stronger smoothing and smaller values converging to the value of the closest point $tilde(bold(z))$ exclusively.

This allows for a continuous space approximation from the quantized space, while also making it possible to pre-compute the quantized space and therefore drastically reducing the computational expenses.

#figure(
fsq_fig,
caption: [Quantized latent lookup of precomputed synchronization metrics.
Point colors represent the amount of desynchronization $s^1$ in the parenchymal layer.
Neighboring points, the $cal(N)_(3times 3)$ sub-grid indicated by the red outlines around $tilde(bold(z))$, are used smoothed using a Gaussian-like kernel, represented by the color gradient around estimation point $hat(z)$.
This allows continuous interpolation the parameter space.
],
) <fig:fsq>

This quantization strategy, called _latent lookup_ is closely related to #acr("FSQ") @Placeholder, which used in Dreamer V3 @Placeholder for example.
Both allow for differentiable quantization, with details on the latent lookup implementation, including grid-resolution and kernel size, can be found in @sec:impl_fsq.

Unlike classical #acr("PINN") @Placeholder where the integration of the #acr("ODE") itself provides gradients directly, here the gradient information is provided by the nearby quantized points which contribute to estimated synchronicity measure through the smoothing.

// #footnote[Dropping the softmax-normalization and defining the $2T_d=sigma^2$, where $sigma^2$ is the variance, the smoothing resembles a Gaussian or Radial-Basis Kernel]
=== Decoder <sec:theory_dec>
As shown in the visualization of the #acr("DNM") phase space in @fig:phase multiple latent coordinates $bold(z)$ result in the same amount of desynchronization, which is not surprising, since different physiological states share the same #acr("SOFA") level.
But when different physiological states have a common #acr("SOFA")-score but from different physiological reasons, their latent representations should be different and unique to that exact physiological state.
This should enable to distinguish different triggers of the organ failure inside the latent space, similarly to how it is possible to distinguish the different triggers from the #acr("EHR").

In a classical Auto-Encoder @Placeholder setting, to encourage a semantically structured latent space, a decoder module is added as an auxiliary regularization component.
A neural decoder network:

$
  d_theta: RR^2 times [0,1] times [0,1] -> RR^n
$
 attempts to reconstruct the original #acr("EHR") features from the latent representation, the resulting desynchronicity of that latent coordinate and the heuristic risk measures:

$
 hat(bold(mu))_t  = d_theta (hat(bold(z))_(t), tilde(s)^1_(hat(bold(z))_(t)), tilde(I))
$
where the gradients only flow through $hat(bold(z))_(t)$, the flow is stopped for the amount of desynchronicity $tilde(s)^1_(hat(bold(z))_(t))$ and the infection risk $tilde(I)$, but they are provided as additional information.
This way the decoder only learns to disentangle the latent coordinates in $hat(bold(z))_(t)$ based on ground future #acr("EHR")s $bold(mu)_t$, through a supervised loss:
$
  cal(L)_"dec" = M S E_(B,T) (bold(Mu)_t, bold(hat(Mu))_t) = 1/(B T) sum^B_(i=1) sum^T_(t=0) (bold(mu)^((i))_t - bold(hat(mu))^((i))_t)^2 
$

The formulation is based on the assumption:
$
  bold(mu)_t = hat(bold(mu))_t + epsilon
$
with $epsilon$ the measurement noise, to hold.

This latent regularization is motivated by _Representation Learning_ @Bengio2012Representation and ensures that nearby points in the latent $(cmbeta(beta), cmsigma(sigma))$-space correspond to physiologically similar patient states.
It helps the encoder $e_theta$ to learn a meaningful alignment between #acr("EHR")-derived latent-embeddings and the dynamical #acr("DNM") landscape.

Using this regularization the recurrent predictor $e_theta$ and the auto-regressive predictor $r_theta$ are encouraged to map temporally consecutive to spatially near latent coordinates, since it is expected that consecutive #acr("EHR")s do not exhibit drastic changes.
Leading smooth patient trajectories through the latent space.

== Overall Training Objective and Metrics
#todo[should summarize objectives again, maybe table?]
