#import "../thesis_env.typ": *
#import "../figures/poster_fig.typ": create-poster-figure
#import "../figures/fsq.typ": fsq_fig
#import "../figures/ldm_modules.typ": ldm_fig

= Background <sec:bg>
This chapter introduces the clinical and mathematical foundations underlying the #acr("LDM").
It covers the definition and pathophysiology of sepsis, the physiological network model that provides the interpretable latent structure, and the #acr("LDM") architecture itself.

*Disclaimer*: This chapter omits exhaustive mathematical formulations in favor of a summary of foundational mechanisms necessary to interpret this project.
For more in depth explanations, details and discussions of related works please see the master's thesis and the corresponding paper @backes2026.

== Sepsis
Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection @Sepsis3.
It is not reducible to a single physiological event but rather a multifaceted breakdown of normal regulatory mechanisms, highly heterogeneous in its triggers, progression, and clinical
presentation.

Under healthy conditions, the immune system responds to pathogens with a controlled release of signaling proteins called _cytokines_, which recruit immune cells to the site of infection and the cytokine release fades out once the pathogen is under control.
Under certain circumstances this regulation fails and a positive feedback loop takes hold, cytokine release becomes self-amplifying, and what began as a localized immune response escalates into a systemic inflammatory state known as a cytokine storm @Jarczak2022storm.
The resulting widespread inflammation disrupts normal organ metabolism.
Cells switch from efficient aerobic to less efficient anaerobic glycolysis, lactate accumulates, blood vessel walls become leaky, and blood pressure falls.
As this process extends across multiple organs simultaneously, the clinical state becomes increasingly difficult to reverse @Sepsis3.
Once multiple organs are affected, a septic condition is reached.

Overall, sepsis carries a substantial global burden.
Roughly 11 million deaths annually are caused by it, and it remains the leading cause of in-hospital deaths @rudd2020global@Via2024Burden.
Treatment outcomes are strongly time-dependent, each hour of delayed intervention increases mortality risk, making early detection a clinical priority @seymour2017time.


=== Sepsis-3 definition
The Sepsis-3 consensus @Sepsis3 defines sepsis operationally as the combination of (i) confirmed or suspected infection and (ii) a dysregulated host response quantified by an acute worsening in organ function.
Organ function is measured by the #acr("SOFA") score @SOFAscore, which evaluates six organ systems (respiratory, coagulation, hepatic, cardiovascular, neurological, and renal), each scored from 0 to 4, for a maximum of 24.
An increase in the total SOFA score of at least 2 points in consecutive assessments is taken as evidence of a dysregulated response and constitutes the organ-dysfunction criterion for sepsis onset.

This definition is the most commonly used clinical characterization and forms the labeling basis for the prediction task in this work.

== Physiological network model

The #acl("PNM") (#acr("PNM")) @Sawicki2022DNM @Berner2022Critical is grounded in the framework of network physiology @Ivanov2021Physiolome, which treats the human body as an integrated system where macroscopic health states emerge from the coordinated dynamics of interacting subsystems.
Rather than modeling the biochemical details of sepsis at the molecular level, the #acr("PNM") takes a functional approach.
Collective metabolic activity of organ cells and immune cells are represented as populations of coupled phase oscillators, and models the dysregulation of sepsis as a loss of synchronization between them.

This level of abstraction is deliberate.
Fully mechanistic models of sepsis require high-resolution measurements rarely available in routine clinical care and resist large-scale validation. The #acr("PNM") instead encodes physiological concepts in a small set of interpretable parameters while remaining computationally tractable.

=== Model structure

The model is a duplex network with two layers of $N$ identical phase oscillators, one representing parenchymal organ cells (layer 1) and one representing immune cells in the stroma (layer 2), coupled both within and between layers.
The dynamics are governed by
$
  dot(phi)^1_i =& omega^1 - 1/N sum^N_(j=1) lr({ (a^1_(i j) + kappa^1_(i j))sin(phi^1_i - phi^1_j + alpha^(11)) }) - sigma sin(phi^1_i - phi^2_i + alpha^(12)) #<odep1> \
$
$
  dot(kappa)^1_(i j) &= -epsilon^1 (kappa^1_(i j) + sin(phi^1_i - phi^1_j - beta)) quad quad quad quad quad quad quad quad quad #<odek1> \
$
$
  dot(phi)^2_i =& omega^2 - 1/N sum^N_(j=1) lr({kappa^2_(i j)sin(phi^2_i - phi^2_j + alpha^(22)) }) - sigma sin(phi^2_i - phi^1_i + alpha^(21)) quad quad quad quad #<odep2> \
$
$
  dot(kappa)^2_(i j) &= -epsilon^2 (kappa^2_(i j) + sin(phi^2_i - phi^2_j - beta)). quad quad quad quad quad quad quad quad quad #<odek2>
$ <eq:ode-sys>
Together these equations describe how metabolic coupling evolves in response to a immune disturbance.
$kappa^2_(i j)$, the coupling of the stroma layer, evolves at a faster timescale ($0 < epsilon^1 << epsilon^2 << 1$) to reflect the faster dynamics of immune signaling relative to organ adaptation.
A schematic of the #acr("PNM") can be seen in @fig:dnm-schem.

#figure(
  create-poster-figure(
    show-encoder: false,
    show-predictor: false,
    show-dnm: true,
    show-sofa: false,
    show-labels: false,
  ),
  caption: flex-caption(
    long: [Schematic of the #acr("PNM") model, showing the two fully-connected layers and the cytokine based communication connectivity.
      Each dot represents a single phase oscillator.],
    short: [Schematic of the #acs("PNM").],
  ),
) <fig:dnm-schem>


The instantaneous phase velocities $dot(phi)^(1,2)_i$ model metabolic activity (larger phase velocity, i.e. frequency, correspond to faster metabolism); the adaptive coupling weights $kappa^(1,2)_(i j)$ model cytokine-mediated communication.
Following the findings of @Berner2022Critical, the two relevant parameters are:

- $beta$ *biological age including comorbidities*: controls the plasticity of
  the adaptive coupling. Near $beta = pi/2$ the coupling is maximally
  synchronizing (Hebbian rule); for other values the feedback is phase-shifted,
  progressively reducing the tendency toward synchronization. It summarizes
  age, inflammatory baseline, adiposity, pre-existing conditions, and related
  risk factors.

- $sigma$ *interlayer coupling strength*: governs the interaction between organ
  tissue and immune cells through the basal lamina.

All further system variables and parameters are summarized in @tab:dnm together with their medical interpretation.

The system evolution depends on the choice of initial coupling weights $kappa^(1,2)_(i j)(t=0) in [-1, +1]$ and initial phases $phi^(1, 2)_i(t=0)$, which are typically independent draws from a uniform random distribution over $[0, 2pi)$.
The coupling matrix $kappa^(1)_(i, j)$ is drawn uniformly from its value range, whereas $kappa^2_(i j)$ is split into two clusters, one of size $C N$ with $0 < C < 1$ and one cluster of size $(1-C)N$.
Each cluster is fully interconnected ($kappa^2_(i,j)=1.0$) but without a connection between the clusters.
This cluster structure represents a disturbance to the immune layer, modeling the onset of an infection.
To account for the variability of initial conditions, each parameter set is integrated for an ensemble of $M$ random initializations.

#pagebreak()
#figure(
  table(
    columns: (auto, auto, auto),
    // inset: 10pt,
    align: (center, left, left),
    table.header([*Symbol*], [*Name*], [*Physiological Meaning*]),

    table.cell(colspan: 3, align: left)[*Variables*],
    [$phi_i$], [Phase], [Group of cells],
    [$dot(phi)_i$], [Phase Velocity], [Metabolic activity],
    [$kappa_(i j)$], [Coupling Weight], [Cytokine activity],

    table.cell(colspan: 3, align: left)[*Parameters*],
    [$alpha$], [Phase lag], [Metabolic interaction delay],

    [$beta$],
    [Plasticity rule],
    [Combination of risk factors],
    // [Age, inflammation, pre-existing illness, other risk factor],

    [$omega$],
    [Natural frequency],
    [Natural cellular metabolism],

    [$epsilon$], [Time scale ratios], [Temporal scale of cytokine activity],
    // [$C$], [Initial network perturbation], [-],

    [$a_(i j)$],
    [Connectivity],
    [Fixed intra-organ cell-to-cell interaction],

    [$sigma$],
    [Interlayer coupling],
    [Interaction between parenchymal and \ immune cells through the basal lamina],

    table.cell(colspan: 3, align: left)[*Measures*],
    [$s$],
    [Standard deviation of frequency \ (see @eq:std)],
    [Pathogenicity (Parenchymal Layer)],
  ),
  caption: flex-caption(
    short: [#acl("PNM") Notation],
    long: [Summarization of notation used in the #acr("PNM"). Superscripts indicating the layer are left out for readability.],
  ),
) <tab:dnm>

=== Healthy and pathological states in the PNM
In the model, homeostasis, a healthy balanced physiological state, corresponds to full frequency synchronization of the parenchymal layer.
Pathological states emerge as multifrequency clustering, meaning a subpopulation of oscillators deviates in frequency.
The desychronization reflects the switch of dysfunctional cells to anaerobic metabolism with its altered metabolic rate on the one hand, and which manifests the dysregulated response to a disturbance on the other hand.
The degree of desynchronization is quantified by the ensemble-averaged standard deviation of the mean phase velocity in the parenchymal layer,
$
  s^1(t) = 1/M sum_(m=1)^M sigma_chi (overline(omega)^1_m, t)
$ <eq:std>
where the average is taken over $M = 50$ random initializations.
Larger values of $s^1$ correspond to increasingly severe frequency clustering and a higher risk of organ failure.
Here $overline(omega)^1_m$ is the mean phase velocity of the parenchymal layer of ensemble $m$.

Simulations over the $(beta, sigma)$ parameter plane reveal that smaller values of $beta$ (younger biological age, fewer comorbidities) are associated with stronger synchronization, while larger values progressively approach a pathological regime, in line with the clinical interpretation.
The immune layer exhibits its own distinct dynamics, generally showing stronger activation at low $sigma$ even when the organ layer remains synchronized.

== Latent Dynamics Model <sec:ldm>
The goal of the #acr("LDM"), a physics-informed deep learning architecture, is to combine the predictive power of deep learning with the interpretability of the #acr("PNM").
Given a patient's #acr("EHR"), i.e. the time-resolved sequence of vital signs, laboratory values, and clinical observations recorded during an #acr("ICU") stay, the model not only outputs a sepsis risk score, but also a trajectory through the #acr("PNM")'s $(beta, sigma)$ parameter space.
This grounds the prediction in physiologically meaningful coordinates.
$beta$ can be read as a proxy for biological age and comorbidity burden, and $sigma$ as a proxy for the strength of immune-organ coupling.
Unlike a purely data-driven model, which might achieve comparable accuracy but offer no insight into _why_ a patient is at risk, the #acr("LDM") allows clinicians to track how a patient's estimated physiological state evolves over time.

=== Architecture overview
Following the Sepsis-3 definition, the model factorizes the sepsis probability as
$
  p_theta (S_t | bold(mu)_(1:t)) = p_theta (A_t | I_t inter bold(mu)_(1:t)) dot p_theta (I_t | bold(mu)_(1:t))
$ <sepsis_risk>
where $S_t$ is the sepsis onset indicator, $A_t$ is acute organ dysfunction, corresponding to the instantaneous #acr("SOFA")-score increase $>= 2$, $I_t$ is suspected infection and $bold(mu)_t$ the #acr("EHR") record at time $t$.
This decomposition gives rise to two modules, one for predicting the organ failure and one for the infection.
Here $theta$ represents any set of learnable parameters.

=== Infection module
The infection module $f_theta$ estimates $p_theta (I_t | mu_(1:t))$ using a #acr("GRU") that maintains a hidden state $bold(h)^f in RR^(H_f)$ updated at each timestep.
A linear output layer and sigmoid activation produce the infection probability.
Because infection timing is uncertain in clinical records (documentation is delayed, antibiotic effects are gradual), the binary infection label is replaced during training with a temporally smoothed surrogate that ramps up in the 48 hours before documented onset and decays afterwards.

=== Organ-dysfunction module
The organ-dysfunction module, consisting of $g^e_theta$ and $g^r_theta$, maps #acr("EHR") observations to latent coordinates $bold(z)_t = (z_1, z_2) in RR^2$, which are transformed via sigmoid scaling into the $(beta, sigma)$ parameter space.
To this end, an encoder $g^e_theta$ processes the initial observation $bold(mu)_1$ to produce a starting position $bold(z)_t$ and hidden state $bold(h)^g_1$.
A recurrent function $g^r_theta$ updates both the position and hidden state at each subsequent timestep via residual steps $bold(z)_t = bold(z)_(t-1) + Delta bold(z)_t$.
The resulting trajectory $(beta_t, sigma_t)$ is then evaluated through the precomputed #acr("PNM") grid to produce a time series of desynchronization values $tilde(s)^1_t$, which serves as a continuous proxy for the #acr("SOFA") score $O_t$:

Acute organ dysfunction is detected as a significant increase in consecutive $tilde(s)^1$ values, smoothed over a short causal window to account for the clinical observation that organ failure is a sustained process rather than an instantaneous event:
$
                              hat(O)_t prop & s^1(beta_t, sigma_t), \
  p_theta (A_t| I_t inter bold(mu)_(1:t)) = & sum^r_tau w_t dot "sigmoid"(m(hat(O)_(t-tau)-hat(O)_(t-tau-1) - d)), \
                                    w_tau = & (e^(-alpha tau))/(sum^r_(k=0) e^(-alpha k)).
$
Where $m in RR$ and $d in RR$ are learnable scale and shift parameters, $r in NN$ the temporal window and $alpha in RR_(>0)$ a learnable decay parameter.

=== Latent lookup
Because the #acr("PNM")'s behavior can be modulated by $beta$ and $sigma$ alone, with all other parameters fixed, in principle, it can be fully characterized by simulating a grid of $(beta, sigma)$ values in advance.
To avoid backpropagating through numerical integration of the #acr("PNM")'s ordinary differential equations, the desynchronization values are precomputed on a $60 times 100$ grid over $beta in [0.4pi, 0.7pi]$ and $sigma in [0, 1.5]$.
In the #acr("LDM"), this precomputed grid is used as a lookup table, which enables efficient gradient-based optimization.

At training time, for a given predicted $(beta_t, sigma_t)$, the corresponding $tilde(s)^1$ is obtained by a differentiable softmax interpolation over the $k times k$ neighborhood of the nearest grid point, with a learnable temperature $T_d$ parameter controlling interpolation sharpness.

To achieve smoothing with a kernel-size $k$, the approximated values are calculated by:
$
  tilde(s)^1 (beta_t, sigma_t)=sum_((beta', sigma') in cal(N)_(k times k)(tilde(beta), tilde(sigma))) "softmax"(-(||(beta_t, sigma_t)-(beta', sigma')||^2)/T_d)s^1 (beta', sigma')
$ <eq:ll>
for $K=k^2$ neighboring points, where $k$ is an odd number $>1$.
Here, #box($T_d in RR_(>0)$) the temperature parameter which controls the sharpness of the smoothing, with larger values producing stronger smoothing and smaller values converging to the value of the closest point $tilde(bold(z))$ exclusively.
The $K$ neighboring points can be calculated via:
#box(
  [$
    // i really do love it
    cal(N)_(k times k)(tilde(beta), tilde(sigma)) = quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad quad\ {(tilde(beta) +i dot beta_"step"),(tilde(sigma)+j dot sigma_"step") | i,j in -(k-1)/2,...,-1, 0, 1, ...,(k-1)/2}.
  $<eq:llk>],
)
The interpolation procedure is shown in @fig:fsq.

#figure(
  fsq_fig,
  caption: flex-caption(short: [Latent Lookup], long: [Quantized latent lookup of precomputed synchronization metrics.
    Point colors represent the amount of desynchronization $s^1$ in the parenchymal layer.
    Neighboring points, the $bold(z)' in cal(N)_(3times 3)(tilde(bold(z)))$ sub-grid, indicated by the red outlines and the red rectangle around $tilde(bold(z))$, are used smoothed using a Gaussian-like kernel, represented by the color gradient around estimation point $hat(bold(z))$.
    This allows continuous interpolation the parameter space.
  ]),
) <fig:fsq>

=== Decoder
An auxiliary decoder $d_theta$ attempts to reconstruct #acr("EHR") features from the latent coordinates $hat(bold(mu))_t = d_theta (beta_t, sigma_t)$, encouraging nearby positions in parameter space to correspond to physiologically similar patient states.
This regularization is included to give the latent space additional semantic structure beyond what the prediction loss alone would impose.

=== Training objective
The #acr("LDM"), with all it modules, is trained jointly with a weighted combination of six losses:

*Primary Sepsis Prediction Loss*\
The main training signal aligns the sepsis risk $p_theta (S_t|bold(mu)_(1:t))$ with ground truth sepsis labels $S_t$.
In the following, $i$ indexes patients within a mini-batch of size $B$, and $t$ indexes timestep's recorded stay of length $T_i$.

Given a mini-batch, the #acr("BCE") loss is used, encouraging the model to minimize the distance between the true sepsis label $S^i_t$ and the predicted sepsis risk $p_theta (S^i_t|bold(mu)^i_(1:t))$:
$
  cal(L)_"sepsis"& = \ -&1/B sum^B_(i=1) 1/T_i sum^(T_i)_(t=1) [S^i_(t)log(p_theta (S^i_t|bold(mu)^i_(1:t))) + (1-S^i_(t))log(1-p_theta (S^i_t|bold(mu)^i_(1:t))))].
$
Although the risk $p_theta (S_t|bold(mu)_(1:t))$ is constructed from infection and organ dysfunction estimates, the primary sepsis loss $cal(L)_"sepsis"$ is required to align their interaction with true clinical outcomes.
This should help to compensate for modeling approximations and measuring uncertainties, while enabling the joint end-to-end optimization of all components for the actual prediction objective.
*Infection loss*\
In order to supervise the model to learn infectious patterns from the #acr("EHR") history, again a #acr("BCE") loss is used to align the true suspected infection label $overline(I)_t$ and the predicted infection risk $p_theta (I_t | bold(mu)_(1:t))$:
$
  cal(L)_"inf"& = \ -&1/B sum^B_(i=1) 1/(T_i) sum^(T_i)_(t=1) lr([overline(I)^i_(t) log(p_theta (I^i_t | bold(mu)^i_(1:t))) + (1-overline(I)^i_(t))log(1-p_theta (I^i_t | bold(mu)^i_(1:t)))]., size: #150%)
$

*SOFA loss*\
The encoder $g^e_theta$ and rollout module $g^r_theta$ of the organ-dysfunction branch are learning to position the latent points $(beta, sigma)$ in such a way, that the following #acr("MSE") loss is minimized:
$
  cal(L)_"sofa" = 1/B sum^B_(i=1) 1/(T_i) sum^(T_i)_(t=1) w_(O_(i,t)) dot (O_(i,t)/24 - (s^1_(i,t)(beta_t, sigma_t))/s^1_"max")^2.
$
Aligning the surrogate $s^1$ with the ground truth #acr("SOFA")-score $O$.
The class-balancing weight:
$
  w_O = log(1 + f_O^(-1))
$
with $f_O$ being the relative frequency of #acr("SOFA")-scores $O$ up-weighs rare high #acr("SOFA")-scores, that are clinically critical but statistically underrepresented, by their inverse frequency.
Also notice that both parts, i.e. the continuous approximation (given by the desynchronicity) and ground truth are scaled to the interval $[0, 1]$.

*Latent Space Regularization*\
To prevent collapse and ensure diverse latent representations the following loss is introduced:
$
  cal(L)_"spread" = -log(det("Cov"(bold(hat(Z)))))
$
where $bold(hat(Z)) in RR^(2 times B dot T)$ collects all predicted latent coordinates of a mini-batch.
$"Cov"(dot)$ computes the sample covariance matrix.

The loss is minimized when $det("Cov"(bold(hat(Z))))$ of the latent dimensions $beta$ and $sigma$ is increased.
This quantity is known as the _generalized variance_ @Carroll1997, and roughly measures the volume occupied by the distribution, where it increases when sampled points spread more.
Consequently, $cal(L)_"spread"$ encourages a larger spread in the latent space and prevents collapse to a narrow region.

*Latent Boundary*\
In order to keep the predicted latent points inside the predefined area, they will be discouraged to move too close to the edges:
$
  cal(L)_"boundary" = "ReLU"(f - "sigmoid"(bold(z)_t)) + "ReLU"("sigmoid"(bold(z)_t) - (1 - f))
$ <eq:bound>
with $f in (0,0.5)$ sets a boundary threshold as a fraction of the space, creating a "penalty buffer" that discourages latent variables from entering the outer $f$-percent of the space near the edges.
The #acr("ReLU") activation function is defined as:
$
  "ReLU"(x) = max(x, 0)
$
and therefore nonzero only for positive inputs and zero otherwise.

*The Decoder Loss*\
The decoder module is trained using a #acr("MSE") supervised loss:
$
  cal(L)_"dec" = 1/(B) sum^B_(i=1) 1/(T_i) sum^(T_i)_(t=1) (bold(mu)^i_(t) - bold(hat(mu))^i_(t))^2.
$
In addition to the prediction objectives, this serves as regularization because the reconstruction objective forces the latent space to maintain a structured organization where physiologically distinct states are positioned into different regions, rather than allowing arbitrary latent encodings.


The combined total loss weighs each component by a hyper-parameter:
$
  cal(L)_"total" = lambda_"sepsis"cal(L)_"sepsis"&+
  lambda_"inf"cal(L)_"inf"&+&
  lambda_"sofa"cal(L)_"sofa"+ \
  lambda_"dec"cal(L)_"dec"&+
  lambda_"spread"cal(L)_"spread"&+&
  lambda_"boundary"cal(L)_"boundary"
$ <eq:loss>
Hyperparameters were tuned manually to avoid introducing a bias in the optimization toward predictive accuracy at the expense of latent space interpretability.

The full architecture with all #acr("LDM") modules is shown in @fig:ldm-arch.

#align(center)[#block([
  #set text(size: 8pt)
  #figure(ldm_fig, caption: flex-caption(
    long: [Complete Latent Dynamics Model architecture with three main components.
      The infection module $f_theta$ and the #acr("SOFA") module $g_theta$ process electronic
      health record data $bold(mu)_t$ through recurrent networks to estimate the infection level $p_theta (I_t | bold(mu)_(1: t))$ and latent coordinates $z_t$, respectively.
      The latent coordinates map to organ failure  $O_t$, from which acute changes $p_theta (A_t | I_t inter bold(mu)_(1:t))$ are computed using consecutive predictions.
      The heuristic organ failure risk is assumed to be $0$ for the initial time step.
      The decoder $d_theta$ reconstructs electronic health record features $bold(mu)_t$ from latent coordinates, regularizing the parameter space to maintain clinically meaningful structure.
      Final sepsis risk $p_theta (S_t | bold(mu)_(1:t))$ combines infection and acute change signals.],
      short: [Complete architecture of the #acs("LDM")]
  )) <fig:ldm-arch>
])]
