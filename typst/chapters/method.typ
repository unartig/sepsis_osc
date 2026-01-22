#import "../thesis_env.typ": *
#import "../figures/online_method.typ": high_fig
#import "../figures/fsq.typ": fsq_fig
#import "../figures/modules.typ": sofa_fig, inf_fig, ldm_fig, dec_fig

= Method \ (Latent Dynamics Model) <sec:ldm>
This chapter introduces the methodological framework used to address the first research question as proposed in @sec:problemdef:
#quote(
  [_*Usability of the #acr("DNM")*: How and to what extent can the #acr("ML")-determined trajectories of the #acr("DNM") be used for detection and prediction, especially of critical infection states and mortality._],
  block: true
)#todo[format]
As established in the previous chapter, the #acr("DNM") provides a theoretical framework for understanding sepsis dynamics through coupled oscillator networks, but it has not yet been validated against real patient data.
This chapter describes how the #acr("DNM") can be embedded within a machine learning pipeline to enable parameter inference from clinical observations and probabilistic risk prediction.

Rather than predicting sepsis directly as a single binary outcome, proposed architecture decomposes the prediction task into two clinically meaningful quantities aligned with the Sepsis-3 definition.
The #acl("SI") and increase in #acr("SOFA") scores are predicted as proxies creating more nuanced and more interpretable prediction results.

The core ideas is to embed the #acr("DNM") into a learnable latent dynamical system.
Here, a neural network learns to position patients into the $(beta, sigma)$-parameter phase space of the #acr("DNM") and a recurrent neural network learns to predict the drift through that space based on observed clinical time series.
The complete architecture, consisting of the #acr("DNM") and additional auxiliary modules, will be referred to as the #acr("LDM") from now on.

This chapter proceeds with @sec:formal, where the prediction task will be reiterated and the prediction strategy formalized.
Desired prediction properties, together with the justification of modeling choices are also introduced here.
Afterwards, in @sec:arch, the architecture will be discussed, focusing on what purpose each part serves and how it is integrated into the broader system.

// To predict the increase in #acr("SOFA"), namely the worsening of organ functionality, the main idea is to utilize parameter level synchronization dynamics inside the parenchymal layer of the functional #acr("DNM"), which is expected to model systemic organ failure.
// Particularly the parameters $cmbeta(beta)$ and $cmsigma(sigma)$, interpreted as biological age and amount of cellular interaction between immune cells and functional organ cells, are of great interest.

// Pre-computed #acr("DNM") dynamics give rise to differentiable #acr("SOFA") and #acr("SI") estimates.


== Formalizing the Prediction Strategy <sec:formal>
In automated clinical prediction systems, a patient is typically represented through their #acl("EHR") (#acr("EHR")).
The #acr("EHR") aggregates multiple clinical variables, such as laboratory biomarker, for example from blood or urine tests, or physiological scores and, further demographic information, like age and gender.
Using the information that is available in the #acr("EHR") until the prediction time-point $t$, the objective is to estimate the patients risk of developing sepsis at that time $t$.
The following methodology will formalize the online-prediction, where newly arriving observations are continuously integrated into updated risk estimates.
To use this prediction system in a clinical setting causality is important.
Causality requires that for every prediction at time $t$ only the information available up to that time-point can be used, and no future observations.

=== Patient Representation
Let $t$ denote an observation time during a patients #acr("ICU")-stay and the available #acr("EHR") at that time consisting of $D$ variables.
After imputation of missing values, normalization, and encoding of non-numerical quantities, each variable $mu_j$ is mapped to a numerical value:
$
  mu_(t,j) in RR, " " j = 1,...,D
$
These values are collected into a column-vector:
$
  bold(mu)_t = (mu_(t,1),..., mu_(t,D))^T in RR^D
$
where the superscript $dot^T$ denoting a transpose operation.
The vector $bold(mu)_t$ is fully describing the current physiological state of the #acr("ICU")-patient at observation time point $t$. 
It is used as the feature vector, meaning it does not carry information that directly translate to the sepsis definition.

=== Modeling the Sepsis-3 Target
The goal is derive continuously updated estimates of sepsis risk based on newly arriving observations $bold(mu)_(t)$ over time, with equally spaced and discrete time-steps $t in { 0, ... ,T }$.
Following the Sepsis-3 definition, the onset of sepsis requires both suspected infection and acute multi-organ failure.

Defining the instantaneous _sepsis onset event_ $S_t in {0, 1}$ as the occurrence of the Sepsis-3 criteria at time point $t$ within the patients #acr("ICU") stay as:
$
  S_t := A_t and I_t
$

Here $A_t={Delta O_t >= 2}$, indicates an acute worsening in organ function, measured via a change in #acr("SOFA")-score $Delta O_t=O_t-O_("base")$ with respect to some patient specific baseline #acr("SOFA")-score $O_("base")$.
The choice of $O_"base"$ has to align with the Sepsis-3 definition, for example a 24 hours running minimum, the or $O_"base" = O_(t-1)$.
The event $I_t$ is an indicator for a #acl("SI") at time $t$ defined according to the Sepsis-3 definition, spanning the 48 hours before and 24 hours after the documented #acr("SI")-onset-time (see @sec:sep3def).
Although the label $I_t$ is defined retrospectively using a time window around the infection onset, this does not violate causality.
The predictive model only conditions on $bold(mu)_(0:t)$, i.e., information available up to time $t$. Future observations are used exclusively for label construction during training and are not available at inference time.

Conditioned on the history of observations $bold(mu)_(0:t)$ the target probability is given by:
$
  Pr(S_t|bold(mu)_(0:t)) = Pr(A_t inter I_t | bold(mu)_(0:t))
$

=== Heuristic Scoring and Risk Estimation <sec:heu>
The direct estimation of the true conditional probability $Pr(S_t|bold(mu)_(0:t))$ is computationally and statistically challenging due to the temporal dependency between the binary Sepsis-3 criteria.
To make the prediction of the target probability more tractable but still connect the statistical estimation to the clinical definition several assumptions and modeling choices are introduced.

Importantly, all assumptions result in differentiable approximations of the real events or probabilities, enabling end-to-end learning of estimators through gradient-based methods.

The central assumption is that infection $I_t$ and multi-organ failure $A_t$ are conditionally independent:

$
  Pr(A_t inter I_t|bold(mu)_(0:t)) = Pr(I_t|bold(mu)_(0:t))Pr(A_t|bold(mu)_(0:t))
$
Clinically this assumption does not hold, since the majority multi-organ failures stem from an underlying infection, meaning they exhibit strong partial correlations.
Yet this assumption is necessary because the #acr("DNM"), which is an essential building block to the #acr("LDM"), only captures organ failure risk irrespective of infection states and the independence allows treating both components separately for the prediction.
Additionally, this separation improves interpretability, since each component can be analyzed individually.

As a second assumption, although the indication $I_t$ is binary, the prediction target is a temporally smoothed version.
The surrogate label $overline(I)_t in [0, 1]$ increases linearly in the 48 hours preceding the infection onset, it reaches maximum at onset, and it decays exponentially afterwards (24 hour window).
This is mimicking temporal uncertainty of the diagnosis, for example due to delayed documentation and treatment effects such as antibiotic half-life.

Thus, the overall prediction requires two separate risk estimators:
$
  tilde(A)_t approx Pr(A_t|bold(mu)_(0:t)) " and " tilde(I)_t approx overline(I)_t
$
Both $tilde(A)_t in (0, 1)$ and $tilde(I)_t in (0, 1)$ are heuristic risk scores serving as approximations for the real event probabilities and the surrogate infection risk score.
The original prediction target has been converted from a calibrated probability to a _heuristic risk score_ $tilde(S)_t$:

$
 Pr(S_(t)|bold(mu)_(0:t)) approx tilde(S)_t := tilde(A)_t tilde(I)_t
$
The interaction term $tilde(A)_t tilde(I)_t$ mirrors their logical conjunction in the Sepsis-3 definition.
It is important to note that $tilde(S)$ is *not a calibrated probability* but a heuristically derived and empirical risk score based on the Sepsis-3 definition, serving as a differentiable surrogate for the Sepsis-3 sepsis onset criterion $P(S_t|bold(mu)_(0:t))$.
Larger values of $tilde(S)_t$ correspond to higher expected risk of sepsis outbreak.

=== From EHR to Risk Scores
The high-dimensional #acr("EHR") history $bold(mu)_(0:t)$ must now be condensed into these two clinically motivated statistics $tilde(A)_t$ and $tilde(I)_t$. 
The #acr("LDM") architecture implements two learned mappings:

*Infection risk estimation*: A data-driven module directly estimates infection risk from the #acr("EHR") history:
$
  tilde(I_t) = f (bold(mu)_(0:t); theta_f)
$
where $f(dot; theta_f)$ represents a neural network with learnable parameters $theta_f$ that will be specified in @sec:arch.

*Organ dysfunction estimation*:
Rather than directly predicting $tilde(A)_t$ from the #acr("EHR"), the #acr("LDM") uses an intermediate representation, a latent #acr("SOFA")-score estimate:
$
  hat(O)_t := g (bold(mu)_(0:t); theta_g)
$
where $hat(O)_t$ denotes a latent, differentiable estimation for the true #acr("SOFA")-score $O_t$.
The function $g(dot; theta_g)$ represents a combined #acr("DNM") pipeline, where $theta_g$ combines all learnable parameters of that pipeline.
Again the function is further specified in @sec:arch.

Given two consecutive estimated #acr("SOFA")-scores $hat(O)_(t-1)$ and $hat(O)_(t)$ a differentiable increase indicator $tilde(A)_t$ is calculated to indicate the event of organ failure:
$
  tilde(A)_t = o_(s,d)(hat(O)_(t-1:t)) =  "sigmoid"(s(hat(O)_t - hat(O)_(t-1) - d))
$ <eq:otoa>
The function $o_(s,d) (dot)$ contains two globally learnable parameters, $d$ a threshold and $s$ a sharpness parameter.
While the Sepsis-3 definition corresponds to a fixed threshold of #box($d = 2$), here $d$ is treated as learnable to obtain a smooth, fully differentiable approximation of the discrete #acr("SOFA") increase criterion and to account for uncertainty in baseline estimation.
The choice of the function
$
"sigmoid"(x)=1/(1+e^(-x))
$
yields a monotonic indicator (larger #acr("SOFA") increase $->$ more likely organ failure) while still being differentiable.

== Architecture <sec:arch>
The previous subsection explained how the sepsis onset target even $S_t$ can be decomposed into the conjunction of suspected infection indication $I_t$ and organ failure event $A_t$ that itself can be calculated from two consecutive #acr("SOFA")-scores $O_(t-1:t)$.
The presented #acl("LDM") is designed to estimate the fundamental components $tilde(O)_t$ and $tilde(I)_t$ from a history of #acr("EHR") $bold(mu)_(0:t)$ to derive the heuristic sepsis risk score $tilde(S)_t approx S_t$ for individual patients.
Each component is estimated by #acr("RNN") module, e.g. #acr("GRU") or #acr("LSTM"), enabling continuous estimation updates based on newly arriving measurements.

The following subsection will introduce the individual modules which are fully differentiable functions with learnable parameters allowing for optimization via first order gradient descent methods.
Starting with the estimator module for the suspected infection indication module in @sec:minf, followed the organ failure estimation module in @sec:msofa which includes the #acr("DNM") to derive #acr("SOFA") estimates and lastly an auxiliary regularization module in @sec:mdec.


#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    
    [*Symbol*], [*Description*],
    
    // Core variables
    [$i, N$], [Patient index and total patients],
    [$t, T_i$], [Time point and trajectory length],
    [$bold(mu)_t in RR^D$], [#acr("EHR") vector with $D$ variables at time $t$],
    [$bold(mu)_(0:t)$], [#acr("EHR") history from time 0 to $t$],

    [$S_t, A_t, I_t in {0, 1}$], [Binary sepsis onset, organ failure, and infection events],
    [$O_t, Delta O_t in [0, ... 24]$], [#acr("SOFA") score and change from baseline],
    [$overline(I)_t in [0,1]$], [Continuous surrogate infection indicator],
    
    [$hat(O)_t in {0, ... 24}$], [Estimated #acr("SOFA") score],
    [$o_(s,d)(hat(O)_(t-1:t))$], [Differentiable increase detection function],
    [$tilde(S)_t, tilde(A)_t, tilde(I)_t in (0, 1)$], [Predicted sepsis, organ failure, and infection risks],
    [$bold(z) = (z_beta, z_sigma)$], [Latent coordinates in #acr("DNM") parameter space],
    [$hat(bold(z))_t, Delta hat(bold(z))_t$], [Predicted latent position and change],
    [$s^1 (bold(z))$], [Synchronization measure (desynchronicity) in #acr("DNM")],
    
    [$bold(h)_t in RR^h$], [Hidden state vector],
    [$f_theta, g^e_theta, g^r_theta, d_theta$], [Infection indicator, encoder, recurrent, decoder modules],
    [$theta$], [Learnable parameters],
    
    table.hline(),
    [$cal(L)_"sepsis", cal(L)_"inf", cal(L)_"sofa"$], [Primary sepsis, infection, and #acr("SOFA") losses],
    [$cal(L)_({"focal","diff","dec","spread","boundary"})$], [Auxiliary focal, directional, decoder, diversity, and boundary losses],
    [$lambda_i$], [Loss weight for component $i$],
    [$B$], [Mini-batch size],
    
  ),
  caption: flex-caption(
    long: [Summarization of notation used in the Latent Dynamics Model methodology.],
    short: [LDM Notation Table]
  ),
  kind: table
) <tab:ldnmnot>


=== Infection Indicator Module <sec:minf>
The first module of the #acr("LDM") estimates the presence of a #acr("SI"), represented by the continuous surrogate indicator $overline(I)_t$, the module predicts a continuous surrogate infection risk $tilde(I)_t in (0, 1)$.
Given $N$ patient trajectories with $T_i$ pairs of #acr("EHR") vectors and ground truth #acr("SI")-indicator each:
$
(bold(mu)_(i,t), overline(I)_(i,t)), "  " i = 1...N, " " t = 1...T_i
$
a parameterized nonlinear recurrent function
$
f_theta_f: RR^D times RR^(H_f) -> (0,1) times RR^(H_f)
$
is trained to map the patients physiological state represented by the #acr("EHR") to an estimated risk of suspected infection:

$
  (tilde(I)_t, bold(h)^f_t)=f_theta_f (bold(mu)_(t),bold(h)^f_(t-1))
$
The hidden state $bold(h)_t$ propagates temporal information through time.
For the first time-step $t=0$ a learnable initial hidden state $bold(h)^f_0$ is used.

The model is implemented as a #acr("RNN"), illustrated in @fig:inf.
At each timestep, a recurrent cell updates the hidden state, and a learned linear projection $bold(W)_f bold(h) ^f_t$, with $bold(W)_f in RR^(1times H_f)$, followed by sigmoid activation produces the infection risk estimate:
$
bold(h)_t &= "RNN-Cell"_(theta_f^"rnn") (bold(mu)_t, bold(h)^f_(t-1)) \
tilde(I)_t &= "sigmoid"(bold(W)_f bold(h)^f_t + b_f)
$
where $theta_f={theta^"rnn"_f, bold(W)_f, b_f}$ combines all learnable parameters, the bias $b_f in RR$ is a single scalar.

To fit the model, given a mini-batch if size $B$, #acr("BCE") loss which measures the distance between true label $overline(I)_t$ and the predicted label $tilde(I)_t$:
$
  cal(L)_"inf" = - 1/(B) sum^B_(i=1) 1/(T_i) sum^(T_i)_(t=1) lr([overline(I)_(i,t) log(tilde(I)_(i,t)) + (1-overline(I)_(i,t))log(1-tilde(I)_(i,t))], size:#150%)
$
is minimized and thus the estimator provides a differentiable estimate of the surrogate infection activity.

#figure(inf_fig,
caption: flex-caption(
long: [Schematic of the Infection Indicator Module architecture and rollout. The #acr("RNN") process the #acr("EHR") sequence $bold(mu)_(0:T)$ step-by-step, maintaining $bold(h)_t$ to capture temporal dependencies, and outputs infection risk estimates $tilde(I)_t$ at each timestep.],
short: [Infection Indicator Module Architecture])
) <fig:inf>

=== SOFA Predictor Module <sec:msofa>
The complete #acr("SOFA") predictor module $g_theta$ is composed two submodules, an initial-encoder $g^e_theta$ and a recurrent latent predictor $g^r_theta$, each described below.
The idea is to translate the physiological patient trajectory to a sequence of the #acr("DNM") parameters $beta$ and $sigma$, where the desynchronization of given parameter pairs should match the physiological organ failure.
To begin with, @sec:theory_surro once more tries to strengthen the connection between organ failure and the #acr("DNM"), followed by @sec:md presents how the #acr("EHR") information is embedded evolved inside the #acr("DNM") parameter space.
Lastly @sec:theory_fsq describes how computational cost can be significantly reduced by precomputing the #acr("DNM") parameter space.

==== The DNM as SOFA Surrogate <sec:theory_surro>
Recalling that the pathological organ conditions within the #acr("DNM") are characterized by frequency clustering in the parenchymal layer.
The amount of frequency clustering is quantified by the ensemble average standard deviation of the mean phase velocity $s^1$ (see @eq:std).
Since $s^1$ monotonically increases with loss of frequency synchronization, it serves as an interpretable and natural surrogate for the #acr("SOFA")-score.
Increasing values of $s^1$ indicate a higher #acr("SOFA")-score and a worse condition of the patients organ system.

Numerical integration of the DNM equations for a given parameter pair #box($bold(z) = (z_beta, z_sigma) = (beta, sigma)$) yields the corresponding #acr("SOFA") approximation $hat(O) (bold(z))$.
By taking the amount of desynchronicity at the end of the integration time $s^1 (T_"sim")$ at the coordinates of $bold(z)$, for readability the time argument is replaced by the parameter arguments $bold(z)$.
Given a desynchronization measure $s^(bold(z))$, the #acr("SOFA") approximate is calculated using:
$
  hat(O) (bold(z)) =round((24 dot  s^1 (bold(z)))/s^1_"max") = round( (24 dot s^1 (beta, sigma)) / s^1_"max") 
$
// These two parameters $(beta, sigma)$ were identified as highly influential and interpretable quantities in the original #acr("DNM") publications @osc2.
Every other system parameter is assumed constant and chosen as listed in @tab:init.
The space spanned by the two parameters is called the _latent space_, coordinate-pairs of that latent space are denoted $bold(z) = (z_beta,z_sigma)$.
In this work only a predefined  subspace of the entire $(beta, sigma)$ plane is used.
To normalize $s^1$ to a $[0, 1]$ range, and by this making it able to retrieve all 24 #acr("SOFA") levels, the values of $s^1$ are divided by the maximum value of the subspace $s^1_max$.
The rounding operation is used only for interpretability and evaluation; during training the normalized continuous $s^1$ value is used.
// In the following $s^1 (bold(z))$ and $hat(O) (bold(z))$ are used synonymously, depending on the context the notation emphasizes the connection to either the #acr("DNM") or the interpretation as #acr("SOFA")-score.

The prediction strategy involves the mapping of individual #acr("EHR") to the latent space, so that the ground truth #acr("SOFA") aligns with the desynchronization measure of the latent coordinate.
Based off of this initial location (and additional information), the patient will perform a trajectory through the latent space yielding step-by-step #acr("SOFA")-score $hat(O)_(0:T) (bold(z))$ estimates needed to calculate the heuristic organ failure statistic $tilde(A)_t$.

==== Latent Parameter Dynamics <sec:md>
Focusing on a single patient, but omitting the $i$ subscript for readability, with its first observation at time $t=1$, an encoder connects the high-dimensional #acr("EHR") to the dynamical regime of the #acr("DNM"), a neural encoder:

$
  g^e_(theta_g^e): RR^D -> RR^2 times RR^(H_g) = RR^(2+H_g)
$
where the high dimensional patient state is mapped to a two-dimensional latent vector, and a $H_g$-dimensional hidden state.

$
  (hat(bold(z))^"raw"_0 , bold(h)^g_0) = ((hat(z)^"raw"_(0,beta), hat(z)^"raw"_(0,sigma)), bold(h)^g_0) = g^e_(theta_g^e) (bold(mu)_0)
$
This encoding locates the patient within a physiologically meaningful region of the #acr("DNM") parameter space, which in context of the #acr("LDM") is called the latent space.
To keep latent coordinates in the predefined area they are ultimately transformed by:
$
  hat(bold(z)) = "sigmoid"(hat(bold(z))^"raw") dot vec(beta_max-beta_min, sigma_max - sigma_min)^T + vec(beta_min, sigma_min)^T
$
Where $dot$ is the element wise matrix multiplication.
The latent coordinate $hat(bold(z))_0$ provides the initial condition for short-term dynamical organ condition forecasting.
As described in @sec:theory_surro the latent coordinates correspond to a #acr("DNM") synchronization behavior and can therefore be directly interpreted as #acr("SOFA")-score estimates (#box($bold(hat(z)) -> s^1 (bold(hat(z))) -> hat(O) (hat(bold(z)))$)).

In addition to the estimated parameter pair $hat(bold(z))^"raw"_0$, the encoder outputs another vector with dimension $H_g<<D$ that is a compressed representation of patient physiology relevant for short-term evolution of $hat(bold(z))$.
This vector $bold(h)^g_0 in RR^(H_g)$ is the initial hidden space.

Since the heuristic #acr("SOFA") risk $tilde(A)$ depends on the evolution of organ function $hat(O)^g_(0:t)$, it is necessary to estimate not only the initial state $hat(bold(z))_0$ but also its evolution.
For this purpose a neural recurrent function:
$
  g^r_theta: RR^(D+2) times RR^(H_g) -> RR^2 times RR^(H_g)
$
is trained to propagate the latent #acr("DNM") parameters forward in time.

This recurrent mechanism, conditioned on the hidden state $bold(h)^g_t$ and previous latent location $bold(z)^"raw"_(t-1)$, captures how the underlying physiology influences the drift of the #acr("DNM") parameters.
From the previous hidden state and latent-position a recurrent cells updates the hidden state, followed by a linear down-projection $bold(W)_g bold(h)^g_t$, with $bold(W)_g in RR^(2times H_g)$, to receive the updated latent-position.
$
  bold(h)_t &= "RNN-Cell"_(theta_g^"rnn") ((bold(mu)_(t), bold(hat(z))^"raw"_(t-1)), bold(h)^g_(t-1)), "  " t = 2,...,T \ 
  Delta hat(bold(z))^"raw"_t &= (bold(W)_g bold(h)^g_t) \
  hat(bold(z))^"raw"_t &= bold(hat(z))^"raw"_(t-1) + Delta hat(bold(z))^"raw"_t
$
where $theta_g^r={theta^"rnn"_g,bold(W))_g in RR }$ combines all the learnable parameters.
The down-projection does not have a bias-term so that no direction is inherently preferred.

Depending on the movement in the latent space the level of synchrony changes across the prediction horizon, which translates to the pathological evolution of patients.
The online-prediction rollout is shown in figure @fig:sofa.

By predicting the movement in the latent space $Delta bold(z)_t$ instead of the raw parameters, smooth trajectories can be learned.
For the latent sequence this is more desirable compared to the infection indicator, where jumps in predicted values do not matter as much. 

To fit the functions, here the placement of latent points $bold(z)$ is guided by a supervision signal through a #acr("MSE") loss:
$
cal(L)_"sofa" = 1/B sum^B_(i=1) 1/(T_i) sum^(T_i)_(t=1) w_(O_(i,t)) dot (O_(i,t)/24 - (s^1_(i,t)(bold(hat(z))))/s^1_"max")^2
$
where the class-balancing weight:
$
  w_O = log(1 + f_O^(-1))
$
with $f_O$ being the relative frequency of #acr("SOFA")-scores $O$.
This inverse-frequency weighting up-weights rare high #acr("SOFA")-scores that are clinically critical but statistically underrepresented.
Also notice that both parts, i.e. the continuous approximation (given by the desynchronicity) and ground truth are scaled to the interval $[0, 1]$.

Because gradients can flow backwards through the whole sequence, minimizing the loss can jointly  train the encoder $g^e_theta$ and recurrent function $g^r_theta$.

#figure(sofa_fig,
  caption: flex-caption(
  long: [Schematic of the online-prediction rollout by the #acr("SOFA") Predictor Module.
  The Encoder $g^e_theta_g^e$, generates the initial latent position $hat(bold(z))^"raw"_0$ based on the first observed #acr("EHR").
  Afterwards, the #acr("RNN") processes the following #acr("EHR") sequence $bold(mu)_(0:T)$ step-by-step, maintaining $bold(h)_t$ to capture temporal dependencies, and outputs the change in latent position $Delta bold(hat(z))^"raw"_t$ at each timestep.
  The new position is the sum of the previous position and its update $bold(hat(z))^"raw"_(t-1) + Delta bold(hat(z))^"raw"_t$],
  short: [#acs("SOFA") Predictor Module Architecture])
  ) <fig:sofa>

#TODO[Raws]
==== Latent Lookup <sec:theory_fsq>
Intuitively one would numerically integrate the #acr("DNM") every estimate $hat(bold(z))$ to receive the $s^1 (bold(hat(z)))$-metric for the continuous space in $(beta, sigma)$.
This approach is taken in  Neural Differential Equations @kidger2022neuraldifferentialequations and Physics Informed Neural Networks @sophiya2025pinn where gradients are typically backpropagated through the #acr("ODE") integration to their input parameters ($beta, sigma$ in this case).
Practically, in case of the #acr("DNM") this is hardly tractable, since the integration is computationally intensive and gradients are prone to vanish over the large integration time and ensemble setup of the #acr("DNM").

To address these challenges, the #acr("LDM") uses a fully differentiable precomputing and caching methodology that still provides meaningful gradients and simultaneously reduces the computational burden.
For that, the continuous latent space has been quantized to a discrete and regular grid, with the metric precomputed for each coordinate pair in the predefined subspace.
The space is limited to the intervals $beta in [0.4pi, 0.7pi]$ and $sigma in [0.0, 1.5]$ (the phase space of the original publication @osc2).
To retrieve values that do not lie exactly on a grid-points, localized soft interpolation is used to derive differentiable synchronicity approximation values.

For an estimated coordinate pair $hat(bold(z))=(hat(z)_beta, hat(z)_sigma)$ in the continuous $(beta, sigma)$-space the quantized metrics are interpolated by smoothing nearby quantization points with a Gaussian-like kernel, which is illustrated in @fig:fsq.

// The smoothing is performed by weighting the amount of desynchronicity $s^1_bold(z)'$ of quantized nearby latent points  by the euclidean distance to the estimation $hat(bold(z))$.
To enable gradient-based optimization, i.e. being differentiable, the lookup of nearby points $bold(z)'$ combines two mechanisms.
Firstly, a straight-through estimator @bengio2013ste for the discrete voxel indexing operation, allowing gradients to flow as if the rounding were identity.
$
tilde(bold(z)) = hat(bold(z)) + "stop_grad"(round(hat(bold(z))) - hat(bold(z))) \
$
In the forward pass this equals the rounded value for lookup, in the backwards pass the $"stop_grad"$ operation blocks gradients from the rounding, so the gradient flows as if no rounding occurred.
Secondly, a differentiable $"softmax"$ interpolation over neighboring grid points.
The nearby points are selected by a rectangular kernel around the closest quantized point $tilde(bold(z))$.
Given a kernel-size $k$ the approximated values is calculated by:
$
tilde(s)^1 (hat(bold(z)))=sum_(bold(z)' in cal(N)_(k times k)(tilde(bold(z)))) "softmax"(-(||hat(bold(z))-bold(z)'||^2)/T_d)s^1 (bold(z)')
$ <eq:ll>
with $"softmax"$ for $K=k dot k$ neighboring points, where $k$ is an odd number $>1$.
Here, #box($T_d in RR_(>0)$) is a learnable temperature parameter which controls the sharpness of the smoothing, with larger values producing stronger smoothing and smaller values converging to the value of the closest point $tilde(bold(z))$ exclusively.
This allows the model to adjust the interpolation sharpness during training, potentially using broad smoothing early on for exploration and sharpening later for precision.

While the squared distances $(||hat(bold(z))-bold(z)'||^2)$ receive exponentially more weight, the  $"softmax"$ operation normalizes the weights to 1, creating a proper convex combination of weights.
$
 "softmax"(bold(x))_j = (e^(x_j))/(sum^K_(k=1) e^(x_k)), "  for " j=1,...,K
$
The $K$ neighboring points can be calculated via:

#box(
[$
 cal(N)_(k times k)(tilde(bold(z))) = {(tilde(z)_beta +i dot beta_"step size"),(tilde(z)_sigma +j dot sigma_"step size") &|\ i,j in -((k-1)/2)...,-1, 0, 1, ...,((k-1)/2)&} 
$<eq:llk>]
)

#figure(
fsq_fig,
caption: flex-caption(short: [Latent Lookup],
long:[Quantized latent lookup of precomputed synchronization metrics.
Point colors represent the amount of desynchronization $s^1$ in the parenchymal layer.
Neighboring points, the $bold(z)' in cal(N)_(3times 3)(tilde(bold(z)))$ sub-grid, indicated by the red outlines and the red rectangle around $tilde(bold(z))$, are used smoothed using a Gaussian-like kernel, represented by the color gradient around estimation point $hat(z)$.
This allows continuous interpolation the parameter space.
]),
) <fig:fsq>

// This quantization strategy allows for continuous space approximation from the quantized space, while also making it possible to pre-compute the quantized space and therefore drastically reducing the computational expenses.
This quantization strategy, called _latent lookup_ #footnote[Implementation is available at https://github.com/unartig/sepsis_osc/blob/main/src/sepsis_osc/ldm/lookup.py] is closely related to #acr("FSQ") @mentzer2023fsq, used in Dreamer-V3 @hafner2024dream for example.
In contrast to this presented latent lookup, the latent coordinates in Dreamer-V3 do not have prior semantic meaning associated with them.
Both allow for differentiable quantization, with details on the latent lookup implementation, including grid-resolution and kernel size, can be found in @sec:impl.


// #footnote[Dropping the softmax-normalization and defining the $2T_d=sigma^2$, where $sigma^2$ is the variance, the smoothing resembles a Gaussian or Radial-Basis Kernel]
=== Decoder <sec:mdec>
As shown in the visualization of the #acr("DNM") phase space in @fig:phase multiple latent coordinates $bold(z)$ result in the same amount of desynchronization, since different physiological states share the same #acr("SOFA") level.
But when different physiological states have a common #acr("SOFA")-score but from different physiological reasons, their latent representations should be different and unique to that exact physiological state.
This should enable to distinguish different triggers of the organ failure inside the latent space, similarly to how it is possible to distinguish the different triggers from the #acr("EHR").

In a classical Auto-Encoder @Bengio2012Representation setting, to encourage a semantically structured latent space, a decoder module is added as an auxiliary regularization component.
A neural decoder network:

$
  d_theta_d: RR^2 -> RR^D
$
 attempts to reconstruct the original #acr("EHR") features from the latent representation, the resulting desynchronicity of that latent coordinate and the heuristic risk measures:

$
 hat(bold(mu))_t  = d_theta_d (hat(bold(z))_(t))
$
This way the decoder only learns to disentangle the latent coordinates in $hat(bold(z))_(t)$ based on ground future #acr("EHR")s $bold(mu)_t$.
The module is trained using a supervised loss:
$
  cal(L)_"dec" = 1/(B) sum^B_(i=1) 1/(T_i) sum^(T_i -1)_(t=0) (bold(mu)_(i,t) - bold(hat(mu))_(i,t))^2 
$
This serves as regularization because the reconstruction objective forces the latent space to maintain a structured organization where physiologically distinct states are positioned into different regions, rather than allowing arbitrary latent encodings.

This latent regularization is motivated by _Representation Learning_ @Bengio2012Representation and ensures that nearby points in the latent $(beta, sigma)$-space correspond to physiologically similar patient states.
It should help the encoder $g^e_theta^e_g$ to learn a meaningful alignment between #acr("EHR")-derived latent-embeddings and the dynamical #acr("DNM") landscape.
Using this regularization the latent encoder $g^e_theta^e_g$ and the recurrent predictor $g^r_theta^r_g$ are encouraged to map temporally consecutive to spatially near latent coordinates, since it is expected that consecutive #acr("EHR")s do not exhibit drastic changes.
Leading smooth patient trajectories through the latent space.

#figure(dec_fig,
  caption: flex-caption(
  long: [Schematic of the data flow in the decoder module.
  The decoder network $d_theta_d$ tries to reconstruct every latent coordinate pair $bold(z)_t$ to the original #acr("EHR") features $bold(mu)_t$.
  This auxiliary component encourages semantically structured latent representations: physiologically similar patient states map to nearby points in the $(beta, sigma)$-space, while different triggers of organ failure occupy distinct regions.
  ],
  short: [Decoder Architecture]))

=== Combining Infection and Acute Change Signals
The complete #acr("LDM"), shown in @fig:ldm, is trained jointly by combining the previously introduced Infection Indicator Module $f_theta_f$ and the #acr("SOFA") prediction module $g_theta_g$.
The output of these modules yield the components $hat(O)_t$, from which $tilde(A)_t$ can be derived (@eq:otoa) and $tilde(I)_t$.

Because positive labels may be temporally windowed around the true onset of sepsis $S_t$, the estimated sepsis risk score is computed via causal smoothing:
$
  tilde(S)_t = "CS"(tilde(A)_t) dot tilde(I)_t
$
where $"CS"(dot)$ denotes a causal smoothing operator that maintains elevated predictions in the time-steps preceding sepsis onset.
The causal smoothing operation is defined as:
$
  "CS"(x_t) = sum_(tau=0)^r w_tau dot x_(t-tau), quad w_tau = (e^(-alpha tau))/(sum_(k=0)^r e^(-alpha k))
$ <eq:cs>
with radius $r$ is a hyper-parameter controlling the temporal radius of the smoothing window, and $alpha$ a learnable decay parameter controlling the length and shape of the smoothing kernel.
To handle the sequence boundaries $x_(t-tau)=0$ for $t - tau < 0$.

This smoothing ensures that organ failure predictions remain elevated during the causal window preceding sepsis onset, matching the clinical reality that organ dysfunction typically precedes documented sepsis.

#figure(ldm_fig,
  caption: flex-caption(
  long: [
  Complete #acr("LDM") architecture with three main components.
  The Infection Module $f_theta_f$ and #acr("SOFA") Module $g_theta_g$ process #acr("EHR") data $bold(mu)_t$ through recurrent networks to estimate infection level $tilde(I)_t$ and latent coordinates $bold(hat(z))_t$ respectively.
  The latent coordinates map to organ failure $hat(O)_t$, from which acute changes $tilde(A)_t$ are computed using consecutive predictions.
  The heuristic organ failure risk is assumed to be 0 for the initial time step.
  The Decoder $d_theta_d$ reconstructs EHR features $bold(mu)_t$ from latent coordinates, regularizing the latent space to maintain clinically meaningful structure.
  Final sepsis risk $S_t$ combines infection and acute change signals.
  ],
  short: [Complete #acs("LDM") Architecture])
  ) <fig:ldm>


== Training Objective and Auxiliary Losses <sec:training_objective>
Besides the losses already presented, to guide the training process multiple auxiliary losses are used and introduced in the following.

*Primary Sepsis Prediction Loss*\
The main training signal aligns the heuristic sepsis score $tilde(S)_t$ with ground truth sepsis labels:

$
cal(L)_"sepsis" = -1/B sum^B_(i=1) 1/T_i sum^(T_i)_(t=1) [S_(i,t)log(tilde(S)_(i,t)) + (1-S_(i,t))log(1-tilde(S)_(i,t))]
$
using the #acr("BCE").

*Organ Failure Alignment*\
To address the problem of high class imbalance, the focal loss @lin2018focal is used to penalize the misclassification of the rare discrete organ failure events:
$
cal(L)_"focal" = -1/B sum^B_(i=1) sum^(T_i)_(t=1) alpha(1-p_(i,t))^gamma log(p_(i,t))
$
with $p_(i,t) = A_(i,t)dot tilde(A)_(i,t) + (1 - A_(i,t)dot (1 - tilde(A)_(i,t)))$, the hyper-parameter $gamma$ controlling the focus on hard examples and $alpha$ emphasizing positive vs. negative samples.
With this loss the model is encouraged to align the timing of predicted #acr("SOFA") increase with the ground truth.

*Difference Alignment*\
This loss encourages temporally coherent latent dynamics that align with ground truth #acr("SOFA") progression:
$
  cal(L)_"diff" = 1/B sum^B_(i=1) sum_(t=1)^(T_i) sum_(t'=t+1)^(T_i) w_(t,t') dot "ReLU"(-a_(t,t'))
$

where the alignment term $a_(t,t')$ measures directional consistency between predicted and true #acr("SOFA") changes:
$
  a_(t,t') = (hat(O)_(t') - hat(O)_t) dot (O_(t') - O_t)
$

and the weight $w_(t,t')$ emphasizes larger ground truth changes:
$
  w_(t,t') = |O_(t') - O_t| + 1
$
The #acr("ReLU") activation:
$
 "ReLU"(x) = max(x, 0)
$
penalizes only misaligned directions (when $a_(t,t') < 0$), meaning the predicted change points in the opposite direction to the true change.
This loss ensures that if a patient's ground truth #acr("SOFA")-score increases between time $t$ and $t'$, the predicted score also increases (and vice versa for decreases), without strictly enforcing the magnitude of change.

*Latent Space Regularization*\
To prevent collapse and ensure diverse latent representations the following loss is introduced:
$
cal(L)_"spread" = -log(det("Cov"(bold(hat(Z)))))
$
where $bold(hat(Z)) in RR^(2 times B dot T)$ collects all predicted latent coordinates of a batch.
$"Cov"(dot)$ computes the sample covariance matrix.

The loss is minimized when the _generalized variance_ @Carroll1997 of the latent dimensions $beta$ and $sigma$.
The generalized variance roughly measures the density of distributions and increases they become more dense, the loss $cal(L)_"spread"$ therefore encourages a larger spread inside the latent space.

*Latent Space Regularization*\
In order to keep the predicted latent inside the predefined area, they will be discouraged to move too close to the edges:
$
  cal(L)_"boundary" = "ReLU"(f - "sigmoid"(bold(z)^"raw"_t)) + "ReLU"("sigmoid"(bold(z)^"raw"_t - (1 - f))
$
with $f in (0,0.5)$ sets a boundary threshold as a fraction of the space, creating a "penalty buffer" that discourages latent variables from entering the outer $f$-percent of the space near the edges..

=== Combined Objective
The complete #acr("LDM") #footnote[Implementation of the #acr("LDM") components is available at https://github.com/unartig/sepsis_osc/tree/main/src/sepsis_osc/ldm] is trained jointly by optimizing all components with the weighted total loss:
$
  cal(L)_"total" = lambda_"inf" cal(L)_"inf" &+&
                   lambda_"sofa" cal(L)_"sofa" &+&
                   lambda_"dec" cal(L)_"dec" &+&
                   lambda_"sepsis" cal(L)_"sepsis" + \
                   lambda_"diff" cal(L)_"diff"  &+&
                   lambda_"focal" cal(L)_"focal" &+&
                   lambda_"spread" cal(L)_"spread" &+&
                   lambda_"boundary" cal(L)_"boundary" 
$ <eq:loss>

The loss weights $lambda$ balance the contribution of each objective during training.
The primary sepsis prediction loss $cal(L)_"sepsis"$ provides the main learning objective aligned with the clinical task, while component losses $cal(L)_"inf"$ and $cal(L)_"sofa"$ ensure accurate estimation of the underlying infection and organ failure indicators.
The auxiliary losses ($cal(L)_"focal"$, $cal(L)_"diff"$, $cal(L)_"dec"$, $cal(L)_"spread"$) regularize the latent space structure and temporal dynamics to improve generalization and interpretability.
Specific values for the loss weights $lambda$ and other hyperparameters are reported in @sec:impl.

@tab:losses provides an overview of all loss components, their purpose, and the modules they supervise.

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    [*Loss*], [*Type*], [*Purpose*], [*Supervises*],
    [$cal(L)_"sepsis"$], [#acr("BCE")], [Primary sepsis prediction], [$f_theta_f, g^e_theta^e_g, g^r_theta^r_g$],
    [$cal(L)_"inf"$], [#acr("BCE")], [Infection indicator], [$f_theta_f$],
    [$cal(L)_"sofa"$], [Weighted #acr("MSE")], [#acr("SOFA") estimation], [$g^e_theta^e_g, g^r_theta^r_g$],
    [$cal(L)_"focal"$], [Focal Loss], [Organ failure timing], [$g^e_theta^e_g, g^r_theta^r_g$],
    [$cal(L)_"diff"$], [Directional], [Difference timing], [$g^r_theta^r_g$],
    [$cal(L)_"dec"$], [#acr("MSE")], [Latent semantics], [$d_theta_d$, ($g^e_theta^e_g, g^r_theta^r_g$)],
    [$cal(L)_"spread"$], [Covariance], [Latent diversity], [$g^e_theta^e_g, g^r_theta^r_g$],
    [$cal(L)_"boundary"$], [Positional], [Latent Space], [$g^e_theta^e_g, g^r_theta^r_g$],
  ),
  caption: flex-caption(long: [Overview of loss components in the #acr("LDM") training objective.], short: [Training Objectives])
) <tab:losses>

== LDM Inference
At inference time, the #acr("LDM") operates as a continuous monitoring system for #acr("ICU") patients, providing real-time risk assessment from admission through the entire ICU stay.
Upon patient admission to the #acr("ICU"), and once initial laboratory measurements are available, the first #acr("EHR") observation $bold(mu)_0$ is processed by both the Infection Indicator module and the latent encoder.
The infection indicator $f_theta_f$ produces an initial infection risk estimate  $tilde(I)_0$ and hidden state $h^f_0$.
The latent encoder $g^e_theta_g^e$ maps the $bold(mu)_0$ to the initial latent coordinates $bold(hat(z))_0$.
Deriving the synchronicity measure $s^1_(0) (bold(hat(z)))$ from the coordinates provides an immediate indication of organ system functionality.

This initialization establishes the patients baseline physiological state within the #acr("DNM") parameter space and provides initial risk indicators.
The hidden states $bold(h)^f_0$ and $bold(h)^g_0$ are saved to enable temporal continuity in subsequent predictions.
Triggered by newly arriving measurements or at regular hourly intervals, the system performs sequential updates.
Updated #acr("EHR")s $bold(mu)_t$ are processed by the recurrent modules $f_theta_f$ and $g^r_theta_g^r$ generating updated estimates on the infection risk and organ system state $tilde(I)_t$ and $bold(hat(z))_t$.
From the history of the latent trajectory $bold(hat(z))_(0:t)$ the acute risk of organ failure $tilde(A)_t$ is calculated and the risk of sepsis estimated $tilde(S)_t$.
This process is run until the patient leaves the #acr("ICU").

Overall, at inference time, the #acr("LDM") provides multiple clinically interpretable indicators at each timestep:
#align(center,
list(
align(left, [*$tilde(I)_t in (0,1)$*: Current infection likelihood]),
align(left, [*$s^1_t (hat(bold(z))) in [0,1]$*: Organ system desynchronization (proxy for SOFA score)]),
align(left, [*$tilde(A)_t in (0,1)$*: Acute organ failure risk (recent worsening)]),
align(left, [*$tilde(S)_t in (0,1)$*: Overall sepsis risk (primary alert signal)]),)
)
These outputs allow clinicians to not only assess overall sepsis risk but also understand the contributing factors, whether the risk stems primarily from suspected infection, acute organ deterioration, or both.
Additionally, the latent trajectory $s^1_(0:t) (hat(bold(z)))$ through the #acr("DNM") parameter space provides interpretable visualization of the patients physiological evolution over time.

== Assessing the Prediction Performance <sec:metrics>
In order to qualitatively assess the prediction performance of sepsis prediction models two theoretically grounded metrics will be introduced. 
The prediction of a patient developing sepsis vs. no sepsis is a binary decision problem based off of the continuous estimated heuristic sepsis risk $tilde(S)_t$.
Given an estimated risk $tilde(S)_t$ and a decision threshold $tau in [0, 1]$, the decision if a prediction value counts as septic is given by the rule:
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
Together the #acr("AUROC") and #acr("AUPRC") are the commonly reported performance metrics used in the sepsis prediction literature @Moor2021Review and will be used to compare the performance between the #acr("LDM") and a baseline approach.

== Summary of Methods
This chapter introduced the proposed model for short-term and interpretable risk prediction of developing sepsis for #acr("ICU") patients, referred to as #acl("LDM").
Starting from the formal task definition, the full processing pipeline and detailed the architecture of the encoder, recurrent latent dynamics module, decoder, and the infection-indicator classifier have been presented.
A key component of the approach is the integration of the functional #acr("DNM") into the latent dynamics, enabling physiologically meaningful and interpretable temporal modeling.

The training losses, including auxiliary losses, used for each component were defined and explained how they contribute to the overall optimization objective.
Additionally, chapter explained how the #acr("LDM") can be used to support clinical monitoring of patients.
Lastly, the two metrics #acr("AUROC") and #acr("AUPRC") were introduced to assess the prediction performance.

The next @sec:experiment presents an experiment where the #acr("LDM") is trained using a widely used data-source #acr("ICU") in order to benchmark sepsis prediction capabilities.
Therefore, it presents the exact task and cohort definitions and #acr("LDM") parameterization as well as the training procedure.
The relevant evaluation metrics #acr("AUROC") and #acr("AUPRC") are used to assess the predictive performance and compare to existing baseline methods.
