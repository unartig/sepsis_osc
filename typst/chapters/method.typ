#import "../thesis_env.typ": *
#import "../figures/online_method.typ": high_fig
#import "../figures/fsq.typ": fsq_fig
#import "../figures/modules.typ": sofa_fig, inf_fig, ldm_fig, dec_fig

= Method \ (Latent Dynamics Model) <sec:ldm>
This chapter introduces the methodological framework used to address the first research question stated in @sec:problemdef:
#align(
  center,
  [*Usability of the #acr("DNM")*: How and to what extent can the #acr("ML")-determined trajectories of the #acr("DNM") be used for detection and prediction, especially of critical infection states and mortality.
  ],
)#todo[format]

To investigate this, a deep learning pipeline targeting the online prediction scheme (see @sec:comp) has been developed, in which the #acr("DNM") is embedded as central part.
Instead of predicting sepsis directly, the two components, #acl("SI") and increase in #acr("SOFA") scores are predicted as proxies creating more nuanced and more interpretable prediction results.

To predict the increase in #acr("SOFA"), namely the worsening of organ functionality, the main idea is to utilize parameter level synchronization dynamics inside the parenchymal layer of the functional #acr("DNM"), which is expected to model systemic organ failure.
Particularly the parameters $cmbeta(beta)$ and $cmsigma(sigma)$, interpreted as biological age and amount of cellular interaction between immune cells and functional organ cells, are of great interest.

In order to achieve this, the #acr("DNM") is embedded into a learnable latent dynamical system, where patients are placed into the two-parameter phase space and a recurrent module predicts physiological drift in that space.
Pre-computed #acr("DNM") dynamics give rise to differentiable #acr("SOFA") and #acr("SI") estimates.
The complete architecture, consisting of the #acr("DNM") and additional auxiliary modules, which will be referred to as the #acr("LDM") from now on.

This chapter proceeds in @sec:formal with the prediction task to be reiterated and the strategy formalized and the introduction of desired prediction properties, together with the justification of modeling choices.
Afterwards, in @sec:arch, the individual modules of the #acr("LDM") will be discussed, focusing on what purpose each serves and how it is integrated into the broader system, especially the #acr("DNM").
#todo[Notation table?]

== Formalizing the Prediction Strategy <sec:formal>
In automated clinical prediction systems, a patient is typically represented through their #acl("EHR") (#acr("EHR")).
The #acr("EHR") aggregates multiple clinical variables, such as laboratory biomarker, for example from blood or urine tests, or physiological scores and, further demographic information, like age and gender.
Using the information that is available in the #acr("EHR") until the prediction time-point $t$, the objective is to estimate the patients risk of developing sepsis at that time $t$.
The following methodology will formalize the online-prediction, where newly arriving observations are continuously integrated into updated risk estimates.
To use this prediction system in a clinical setting it is causality is important, this requires that for every prediction at time $t$ only the information available up to that time-point can be used, and no future observations.

=== Patient Representation
Let $t$ denote an observation time during a patients #acr("ICU")-stay and the available #acr("EHR") at that time consisting of $n$ variables.
It is assumed that $bold(mu)$ does not carry the quantities that directly translate to the sepsis definition.
After imputation of missing values, normalization, and encoding of non-numerical quantities, each variable $mu_j$ is mapped to a numerical value:
$
  mu_(t,j) in RR, " " j = 1,...,n
$
These values are collected into a column-vector:
$
  bold(mu)_t = (mu_(t,1),..., mu_(t,n))^T in RR^n
$
, where the superscript $dot^T$ denoting a transpose operation.
The vector $bold(mu)_t$ is fully describing the current physiological state of the #acr("ICU")-patient at observation time point $t$.

=== Modeling the Sepsis-3 Target
The goal is derive continuously updated estimates of sepsis risk based on newly arriving observations $bold(mu)_(t)$ over time, with equally spaced and discrete time-steps $t in { 0, ... ,T }$.
Following the Sepsis-3 definition, the onset of sepsis requires both suspected infection and acute multi-organ failure.

Defining the instantaneous _sepsis onset event_ $S_t$ as the occurrence of the Sepsis-3 criteria at time point $t$ within the patients #acr("ICU") stay as:
$
  S_t := A_t inter I_t
$

Here $A_t={Delta O_t >= 2}$, indicates an acute worsening in organ function, measured via a change in #acr("SOFA")-score $Delta O_t=O_t-O_("base")$ with respect to some patient specific baseline #acr("SOFA")-score $O_("base")$.
The choice of $O_"base"$ has to align with the Sepsis-3 definition, for example a 24 h running minimum, the or $O_"base" = O_(t-1)$.
The event $I_t$ is an indicator for a #acl("SI") at time $t$ defined according to the Sepsis-3 definition, spanning the 48 h #footnote[Although the label $I_t$ is defined retrospectively using a time window around the infection onset, this does not violate causality. The predictive model only conditions on $bold(mu)_(0:t)$, i.e., information available up to time $t$. Future observations are used exclusively for label construction during training and available at inference time.] before and 24 h after the documented #acr("SI")-onset-time (see @sec:sep3def).

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

As a second assumption, although the indication $I_t$ is binary, the target is a temporally smoothed version.
The surrogate label $overline(I)_t in [0, 1]$ increases linearly in the (48 h) hours preceding the infection onset, it reaches maximum at onset, and it decays exponentially afterwards (24 h), mimicking temporal uncertainty of establishment and resolution, for example due to delayed documentation and treatment effects such as antibiotic half-life.

Thus the overall prediction requires two separate risk estimators:
$
  tilde(A)_t approx Pr(A_t|bold(mu)_(0:t)) " and " tilde(I)_t approx overline(I)_t
$
Both $tilde(A)_t in (0, 1)$ and $tilde(I)_t in (0, 1)$ are heuristic risk scores serving as approximations for the real event probabilities and the surrogate infection risk score.
The original prediction target has been converted from a calibrated probability to a _heuristic risk score_ $tilde(S)_t$:

$
 Pr(S_(t)|bold(mu)_(0:t)) approx tilde(S)_t := tilde(A)_t tilde(I)_t
$
Where the infection surrogate infection risk score can be directly derived from the #acr("EHR") history:
$
  tilde(I_t) := f_theta (bold(mu)_(0:t))
$

In contrast, $tilde(A)_t$ is not derived directly from the #acr("EHR"), instead it relies on estimated #acr("SOFA")-score dynamics:
$
  hat(O)_t := g_theta (bold(mu)_(0:t))
$
where $hat(O)_t$ denotes a latent, differentiable estimation for the true #acr("SOFA")-score $O_t$ which cannot be derived directly from the available #acr("EHR").
Given two consecutive estimated #acr("SOFA")-scores $hat(O)_(t-1)$ and $hat(O)_(t)$ a differentiable increase indicator $tilde(A)_t$ is calculated to indicate the event of organ failure:
$
  tilde(A)_t = o_(s,d)(hat(O)_(t-1:t)) =  "sigmoid"(s(hat(O)_t - hat(O)_(t-1) - d))
$
The function $o_(s,d) (dot)$ contains two globally learnable parameters, $d$ a threshold and $s$ a sharpness parameter.
While the Sepsis-3 definition corresponds to a fixed threshold of $d = 2$, here $d$ is treated as learnable to obtain a smooth, fully differentiable approximation of the discrete #acr("SOFA") increase criterion and to account for uncertainty in baseline estimation.
The choice of the function $"sigmoid"(x)=1/(1+e^-x)$ yields a monotonic indicator (larger #acr("SOFA") increase $->$ more likely organ failure) while still being differentiable.

The high-dimensional $bold(mu)$'s have now been condensed into two clinically motivated statistics $tilde(A)_t$ and $tilde(I)_t$.
The interaction term $tilde(A)_t tilde(I)_t$ mirrors their logical conjunction in the Sepsis-3 definition.
It is important to note that $tilde(S)$ is *not a calibrated probability* but a heuristically derived and empirical risk score based on the Sepsis-3 definition, serving as a differentiable surrogate for the Sepsis-3 sepsis onset criterion $P(S_t|bold(mu)_(0:t))$.
Larger values of $tilde(S)_t$ correspond to higher expected risk of sepsis outbreak.

== Architecture <sec:arch>
The previous subsection explained how the sepsis onset target even $S_t$ can be decomposed into the conjunction of suspected infection indication $I_t$ and organ failure event $A_t$ that itself can be calculated from two consecutive #acr("SOFA")-scores $O_(t-1:t)$.
The presented #acl("LDM") is designed to estimate the fundamental components $tilde(O)_t$ and $tilde(I)_t$ from a history of #acr("EHR") $bold(mu)_(0:t)$ to derive the heuristic sepsis risk score $tilde(S)_t approx S_t$ for individual patients.
Each component is estimated by a #acr("RNN") module enabling continuous estimation updates based on newly arriving measurements.

The following subsection will introduce the individual modules which are fully differentiable functions with learnable parameters allowing for optimization via first order gradient descent methods.
Starting with the estimator module for the suspected infection indication module in @sec:minf, followed the organ failure estimation module in @sec:msofa which includes the #acr("DNM") to derive #acr("SOFA") estimates and lastly an auxiliary regularization module in @sec:mdec.

// #figure(
//   scale(high_fig, 65%),
//   caption: flex-caption(
//   short:[Latent Dynamic Model],
//   long:[
//     Flow chart of the different steps taken to produce the heuristic sepsis risk measure $tilde(S)$ from an observed #acl("EHR") $bold(mu)_0$.
//     Learnable neural function parameters are indicated by a $theta$ subscript.
//   ])
// ) <fig:flow>

=== Infection Indicator Module <sec:minf>
The first module of the #acr("LDM") estimates the presence of a #acr("SI"), represented by the continuous surrogate indicator $overline(I)_t$, the module predicts a continuous surrogate infection risk $tilde(I)_t in (0, 1)$.
Given $N$ patient trajectories with $T_i$ pairs of #acr("EHR") vectors and ground truth #acr("SI")-indicator each: 
$
(bold(mu)_(i,t), overline(I)_(i,t)), "  " i = 1...N, " " t = 1...T_i
$
a parameterized non-linear recurrent function
$
f_theta: RR^n -> (0,1) times RR^h
$
is trained to map the patients physiological state represented by the #acr("EHR") to an estimated risk of suspected infection:

$
  (tilde(I), bold(h)_t)_t=f_theta (bold(mu)_(t),bold(h)_(t-1))
$
The hidden state $bold(h)_t$ propagates temporal information through time.
For the first time-step $t=0$ a learned initial hidden state $bold(h)_0$ is used.

The model is implemented as a supervised #acr("RNN") optimized with stochastic gradient descent, throughout training minibatches of size $B$ are sampled.
To fit the model, #acr("BCE")-loss which measures the distance between true label $overline(I)_t$ and the predicted label $tilde(I)_t$:
$
  cal(L)_"inf" = - 1/(B) sum^B_(i=1) 1/(T_i) sum^(T_i -1)_(t=0) lr([overline(I)_(i,t) log(tilde(I)_(i,t)) + (1-overline(I)_(i,t))log(1-tilde(I)_(i,t))], size:#150%)
$
is minimized and thus the estimator provides a differentiable estimate of the surrogate infection activity.

#figure(inf_fig,
caption: flex-caption(long: [TODO], short: [TODO]))

=== SOFA Predictor Module <sec:msofa>
The complete #acr("SOFA") predictor module $g_theta$ is composed two submodules, an initial-encoder $e_theta$ and a recurrent latent predictor $r_theta$, each described below.
The high level idea is to translate the physiological patient trajectory to a sequence of the #acr("DNM") parameters $cmbeta(beta)$ and $cmsigma(sigma)$, where the desynchronization of given parameter pairs should match the physiological organ failure. 
To begin with, @sec:theory_surro once more tries to strengthen the connection between organ failure and the #acr("DNM"), followed by @sec:md presents how the #acr("EHR") information is embedded evolved inside the #acr("DNM") parameter space.
Lastly in @sec:theory_fsq is describes how computational cost can be significantly reduced by pre-computing the #acr("DNM") parameter space.

==== The DNM as SOFA Surrogate <sec:theory_surro>
Recalling that the pathological organ conditions within the #acr("DNM") are characterized by frequency clustering in the parenchymal layer.
The amount of frequency clustering is quantified by the ensemble average standard deviation of the mean phase velocity $s^1$ (see @eq:std).
// Naturally this measure can be used as a proxy for a patients #acr("SOFA")-score.
Since $s^1$ monotonically increases with loss synchrony, it serves as an interpretable and natural surrogate for the #acr("SOFA")-score.
Increasing values of $s^1$ indicate a higher #acr("SOFA")-score and a worse condition of the patients organ system.

Numerical integration of the DNM equations for a given parameter pair $(cmbeta(beta), cmsigma(sigma))$ yields the corresponding #acr("SOFA") estimate $hat(O) (bold(z))$:
$
  hat(O) (bold(z)) = s^1 (bold(z))= s^1 (cmbeta(beta), cmsigma(sigma))
$
these two parameters were identified as highly influential and interpretable quantities in the original #acr("DNM") publications @osc2.
Every other system parameter is assumed constant and chosen as listed in @tab:init.

The space spanned by the two parameters is called the _latent space_, coordinate-pairs of that latent space are denoted $bold(z) = (z_cmbeta(beta),z_cmsigma(sigma))$.
In the following $s^1 (bold(z))$ and $hat(O) (bold(z))$ are used synonymously, depending on the context the notation emphasizes the connection to either the #acr("DNM") or the interpretation as #acr("SOFA")-score.

The prediction strategy involves the mapping of individual #acr("EHR") to the latent space, so that the ground truth #acr("SOFA")-aligns with the desynchronization measure of the latent coordinate.
Based off of this initial location (and additional information), the patient will perform a trajectory through the latent space yielding step by step #acr("SOFA")-score $hat(O)_(0:T) (bold(z))$ estimates needed to calculate the heuristic organ-condition statistic $tilde(A)_t$.

==== Latent Parameter Dynamics <sec:md>
Even though this module is also #acr("RNN") based, in contrast to the infection indicator module from @sec:minf, this module follows a different strategy.
Focusing on a single patient, but omitting the $i$ subscript for readability, with its first observation at time $t=0$, an encoder connects the high-dimensional #acr("EHR") to the dynamical regime of the #acr("DNM"), a neural encoder:

$
  e_theta: RR^n -> RR^2 times RR^h = RR^(2+h)
$
where the high dimensional patient state is mapped to a two-dimensional latent vector, and a $h$ dimensional hidden state.

$
  (hat(bold(z))_0, bold(h)_0) = ((hat(z)_(0,cmbeta(beta)), hat(z)_(0,cmsigma(sigma))), bold(h)_0) = e_theta (bold(mu)_0)
$
This encoding locates the patient within a physiologically meaningful region of the #acr("DNM") parameter space, which in context of the #acr("LDM") is called the latent space.
The latent coordinate $hat(bold(z))_0$ provides the initial condition for short-term dynamical organ condition forecasting.
As described in @sec:theory_surro the latent coordinates correspond to a #acr("DNM") synchronization behavior and can therefore be directly interpreted as #acr("SOFA")-score estimates ($bold(hat(z)) -> s^1 (bold(hat(z))) -> hat(O) (hat(bold(z)))$).

In addition to the estimated parameter pair $hat(bold(z))_0$, the encoder outputs another vector with dimension $h<<n$ that is a compressed representation of patient physiology relevant for short-term evolution of $hat(bold(z))$.
This vector $bold(h)_0 in RR^h$ is the initial hidden space.

Since the heuristic #acr("SOFA") risk $tilde(A)$ depends on the evolution of organ function $hat(O)_(0:t)$, it is necessary to estimate not only the initial state $hat(bold(z))_0$ but also its evolution.
For this purpose a neural recurrent function:

$
  Delta hat(bold(z))_t, bold(h)_t &= r_theta (bold(mu)_(t), bold(h)_(t-1)), "  " t = 1,...,T \
  hat(bold(z))_t &= bold(hat(z))_(t-1) + Delta hat(bold(z))_t
$
is trained to propagate the latent #acr("DNM") parameters forward in time.
This recurrent mechanism, primed by the hidden state $bold(h)_t$ and initial latent location $bold(z)_t$, captures how the underlying physiology influences the drift of the DNM parameters.
Depending on the movement in the latent space the level of synchrony changes across the prediction horizon, which translates to the pathological evolution of patients.

By predicting the movement in the latent space $Delta bold(z)_t$ instead of the raw parameters, a more smooth trajectory can be learned.
For the latent sequence this is more desirable compared to the infection indicator, where jumps in predicted values do not matter. 

To fit the functions, here the placement of latent points $bold(z)$ is guided by a supervision signal:
$
  cal(L)_"sofa" = 1/B sum^B_(i=1) 1/(T_i) sum^(T_i-1)_(t=0) (O_(i,t) - hat(O)_(i,t) (bold(hat(z))))^2
$
and through the batched #acr("MSE")-loss.
This jointly fits the encoder $e_theta$ and recurrent function $r_theta$.

#figure(sofa_fig)

==== Latent Lookup <sec:theory_fsq>
Intuitively one would numerically integrate the #acr("DNM") every estimate $hat(bold(z))$ to receive the $s_(bold(hat(z)))^1$-metric for the continuous space in $(cmbeta(beta), cmsigma(sigma))$.
But to massively reduce the computational burden the space has been quantized to a discrete and regular grid, with the metric pre-computed for each coordinate pair.
The space is limited to the intervals $cmbeta(beta) in [0.4pi, 0.7pi]$ and $cmsigma(sigma) in [0.0, 1.5]$ (the phase space of the original publication @osc2).
Instead of integrating the #acr("DNM") over and over, differentiable approximation values are retrieved from the precomputed grid by using localized soft interpolation.

For an estimated coordinate pair $hat(bold(z))=(hat(z)_cmbeta(beta), hat(z)_cmsigma(sigma))$ in the continuous $(cmbeta(beta), cmsigma(sigma))$-space the quantized metrics are interpolated by smoothing nearby quantization points with a Gaussian-like kernel, which is illustrated in @fig:fsq.

The smoothing is performed by weighting the amount of desynchronicity $s^1_bold(z)'$ of quantized nearby latent points $bold(z)'$ by the euclidean distance to the estimation $hat(bold(z))$.
The nearby points are selected by a quadratic slice around the closest quantized point $tilde(bold(z))$, with $k$ being the sub-grid size:

$
tilde(s)^1 (hat(bold(z)))=sum_(bold(z)' in cal(N)_(k times k)(tilde(bold(z)))) "softmax"(-(||hat(bold(z))-bold(z)'||^2)/T_d)s^1 (bold(z)')
$
with $"softmax"$ for $K=k dot k$ neighboring points being:
$
 "softmax"(bold(x))_j = (e^(x_j))/(sum^K_(k=1) e^(x_k)), "  for " j=1,...,K
$
and:
$
 cal(N)_(k times k)(tilde(bold(z))) = {(tilde(z)_cmbeta(beta) +i dot beta_"step size"),(tilde(z)_cmsigma(sigma) +j dot sigma_"step size") | i,j in {-1, 0, 1} }
$
With $T_d in RR_(>0)$ being a learnable temperature parameter which controls the sharpness of the smoothing, with larger values producing stronger smoothing and smaller values converging to the value of the closest point $tilde(bold(z))$ exclusively.


#figure(
fsq_fig,
caption: flex-caption(short: [Latent Lookup], long:[Quantized latent lookup of precomputed synchronization metrics.
Point colors represent the amount of desynchronization $s^1$ in the parenchymal layer.
Neighboring points, the $bold(z)' in cal(N)_(3times 3)(tilde(bold(z)))$ sub-grid indicated by the red outlines around $tilde(bold(z))$, are used smoothed using a Gaussian-like kernel, represented by the color gradient around estimation point $hat(z)$.
This allows continuous interpolation the parameter space.
]),
) <fig:fsq>
#todo[out arrow]

This quantization strategy allows for continuous space approximation from the quantized space, while also making it possible to pre-compute the quantized space and therefore drastically reducing the computational expenses.
This quantization strategy, called _latent lookup_ is closely related to #acr("FSQ") @mentzer2023fsq, used in Dreamer-V3 @hafner2024dream for example.
Unlike in this approach the values of the latent coordinates in Dreamer-V3 do not have prior semantic meaning associated with them.
Both allow for differentiable quantization, with details on the latent lookup implementation, including grid-resolution and kernel size, can be found in @sec:impl_fsq.

Unlike Neural Differential Equations @kidger2022neuraldifferentialequations and Physics Informed Neural Networks @sophiya2025pinn where the integration of the #acr("ODE") itself provides gradients directly by backpropagation through the #acr("ODE"), here the gradient information is provided by the nearby quantized points which contribute to estimated synchronicity measure through the smoothing.

// #footnote[Dropping the softmax-normalization and defining the $2T_d=sigma^2$, where $sigma^2$ is the variance, the smoothing resembles a Gaussian or Radial-Basis Kernel]
=== Decoder <sec:mdec>
As shown in the visualization of the #acr("DNM") phase space in @fig:phase multiple latent coordinates $bold(z)$ result in the same amount of desynchronization, which is not surprising, since different physiological states share the same #acr("SOFA") level.
But when different physiological states have a common #acr("SOFA")-score but from different physiological reasons, their latent representations should be different and unique to that exact physiological state.
This should enable to distinguish different triggers of the organ failure inside the latent space, similarly to how it is possible to distinguish the different triggers from the #acr("EHR").

In a classical Auto-Encoder @Bengio2012Representation setting, to encourage a semantically structured latent space, a decoder module is added as an auxiliary regularization component.
A neural decoder network:

$
  d_theta: RR^2 -> RR^n
$
 attempts to reconstruct the original #acr("EHR") features from the latent representation, the resulting desynchronicity of that latent coordinate and the heuristic risk measures:

$
 hat(bold(mu))_t  = d_theta (hat(bold(z))_(t))
$
This way the decoder only learns to disentangle the latent coordinates in $hat(bold(z))_(t)$ based on ground future #acr("EHR")s $bold(mu)_t$, through a supervised loss:
$
  cal(L)_"dec" = 1/(B) sum^B_(i=1) 1/(T_i) sum^(T_i -1)_(t=0) (bold(mu)_(i,t) - bold(hat(mu))_(i,t))^2 
$

The formulation is based on the assumption:
$
  bold(mu)_t = hat(bold(mu))_t + epsilon
$
with $epsilon$ the measurement noise, to hold.

This latent regularization is motivated by _Representation Learning_ @Bengio2012Representation and ensures that nearby points in the latent $(cmbeta(beta), cmsigma(sigma))$-space correspond to physiologically similar patient states.
It helps the encoder $e_theta$ to learn a meaningful alignment between #acr("EHR")-derived latent-embeddings and the dynamical #acr("DNM") landscape.

Using this regularization the latent encoder $e_theta$ and the recurrent predictor $r_theta$ are encouraged to map temporally consecutive to spatially near latent coordinates, since it is expected that consecutive #acr("EHR")s do not exhibit drastic changes.
Leading smooth patient trajectories through the latent space.

#figure(dec_fig)

== Summary of Methods
This chapter introduced the proposed model for short-term and interpretable risk prediction of developing sepsis for #acr("ICU") patients, referred to as #acl("LDM").
Starting from the formal task definition, the full processing pipeline and detailed the architecture of the encoder, recurrent latent dynamics module, decoder, and the infection-indicator classifier have been presented.
A key component of the approach is the integration of the functional #acr("DNM") into the latent dynamics, enabling physiologically meaningful and interpretable temporal modeling.

#figure(ldm_fig)


The training losses used for each component were defined contributing to the overall optimization objective.
Finally the chapter also introduced the notation and training setup that will used throughout the remainder of the thesis.

The next @sec:experiment presents the #acr("MIMIC")-IV database, a widely used #acr("ICU") used to benchmark sepsis prediction models, and the exact task and cohort definitions used for baseline comparisons.
The relevant evaluation metrics #acr("AUROC") and #acr("AUPRC") used to assess the predictive performance are introduced.

The chapter concludes with implementation details for training the #acr("LDM") on the #acr("MIMIC")-IV data, for example further auxiliary losses to guide the learning and specific architectural choices.
Final quantitative and qualitative results are presented and analyzed in @sec:results.
