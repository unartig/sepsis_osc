#import "@preview/acrostiche:0.5.2": *
#import "tuhh_template.typ": thesis
#import "@preview/drafting:0.2.2": margin-note, note-outline, inline-note, set-margin-note-defaults
#import "tree.typ": tree_figure

#show: thesis.with(
  title: "Comprehensive Guidelines and Templates for Thesis Writing",
  summary: [
  ],
  // abstract_de: [
  // ],
  acronyms: (
    "TUHH": ("Hamburg University of Technology"),
    "SOFA": ("Sequential Organ Failure Assessment"),
    "ICU": ("Intensive Care Unit"),
    "EHR": ("Electronic Health Record"),
    "RL": ("Reinforcement Learning"),
  ),
  bibliography: bibliography("bibliography.bib"),
  // acknowledgements: [
  //   This thesis was written with the help of many people.
  //   I would like to thank all of them.
  // ],
)

#let mean(f) =  $angle.l$ +  f + $angle.r$
#note-outline()

#let todo = margin-note
#let caution-rect = rect.with(inset: 1em, radius: 0.5em)
#set-margin-note-defaults(rect: caution-rect, side: right, fill: orange.lighten(80%))
#let TODO = inline-note

= Notes
== Base ODE-System
$
dot(phi)^1_i =& omega^1 - 1/N sum^N_(j=1) lr({ (a^1_(i j) + kappa^1_(i j))sin(phi^1_i - phi^1_j + alpha^(11)) }) - sigma sin(phi^1_i - phi^2_i + alpha^(12)) #<odep1> \
dot(kappa)^1_(i j) &= -epsilon^1 (kappa^1_(i j) + sin(phi^1_i - phi^1_j - beta)) #<odek1> \
dot(phi)^2_i =& omega^2 - 1/N sum^N_(j=1) kappa^2_(i j)sin(phi^2_i - phi^2_j + alpha^(22)) - sigma sin(phi^2_i - phi^1_i + alpha^(21)) #<odep2> \
dot(kappa)^2_(i j) &= -epsilon^2 (kappa^2_(i j) + sin(phi^2_i - phi^2_j - beta)) #<odek2>
$ <eq:ode-sys>
Introduced in @osc1 and slightly adapted in @osc2. #todo[table for parameters]
#figure(tree_figure)

== Computational Saving
For parts in the form of $sin(theta_l-theta_m)$ following @KuramotoComp one can calculate and cache the terms $sin(theta_l), sin(theta_m), cos(theta_l), cos(theta_m)$ in advance:

$
sin(theta_l-theta_m)=sin(theta_l)cos(theta_m) - cos(theta_l)sin(theta_m) "    " forall l,m in {1,...,N}
$
so the computational cost for the left-hand side for $N$ oscillators can be reduced from $N (N-1)$ to $4N$ trigonometric function evaluations, positively impacting the computational efficiency of the whole ODE-system significantly.


== Implementation Differences
#list(
[Actual Mean Phase Velocity instead of averaged difference over time.],
[+They Calculate the difference wrong since the phase difference should be mod[$2pi$]],
[Different solver accuracy, but very similar],
[They are not batching the computation],
)

== Metrics <sec:metrics>
=== Kuramoto Parameter
Kuramoto Order Parameter #todo[cite]
$
R^mu_2 = 1/N abs(sum^N_j e^(i dot phi_j (t))) "   with " 0<=R^mu_2<=1
$
$R^mu_2=0$ splay-state and $R^mu_2=1$ is fully synchronized.

Mean Phase Velocities
$
mean(phi^mu) = 1/N sum^N_j phi^mu_j
$

#todo[Entropy, Splay Ratio, MPV Std, Cluster Ratio]

== Data
MIMIC-3

https://github.com/alistairewj/sepsis3-mimic/tree/v1.0.0

== Model
For a general model setup, the latent space $z=(a^1, sigma, alpha, beta, omega^1, omega^2, C/N,epsilon^1, epsilon^2)$ represents the parameter of the dynamic network model, so we have

$
z in RR^d "  with " d = 9
$
As shown in the supplemental material of @osc2, for example, the parameter $alpha$ exhibits a $pi$-periodicity, allowing to reduce the effective parameter space by constraining certain parameters with upper and lower bounds.
These bounds are omitted here for simplicity but are included in #todo[table].
To further reduce the latent space $z$, the we keep $a^1, omega^1, omega^2, C/N, epsilon^1 "and" epsilon^2$ fixed.
The reduced latent space $z'=(sigma, alpha, beta)$:
$
z' in RR^d' "  with " d'=3  
$
where both alpha and beta exhibit a periodic behavior
Each point in the latent space $z_j$ can be categorized as either of #emph[healthy], #emph[vulnerable] or #emph[pathological].

We relate high-dimensional physiological observations (e.g. samples from the MIMIC-III database) to the latent space via:

$
x_j = f(z_j) + epsilon
$ <eq:decoder>
where $f$ is unknown an unknown function and $epsilon$ the measurement noise.
Note that different observations $x_j$ can be mapped to the same classification, as for the latent space.
We define two the two class mappings $Q$ and $R$:
$
Q(x_j)=c_j=R(z_j) "  where " x_j = f(z_j) + epsilon
$
mapping observations and the latent representation to a shared class label $c$.
To make things more complicated, $R$ does not directly act on $z$, but rather the metrics derived from the solution to a dynamical system (initial value problem) (@eq:ode-sys) parameterized by $z$.
The metrics are detailed in @sec:metrics.

In the setting of structured latent variational learning we want to approximate an encoder $q(z|x)$ to infer the latent variables from observed data $X$ and the class.

#TODO[#text(weight: "bold")[How to structure the latent space?]
Binary classification (sepsis, no sepsis) may not provide enough information to accurately structure the latent space.
The options:
#list(
[Add more classes like resilient/vulnerable... maybe even the full spectrum? #list([need to be modeled by $R$])],
[Introduce the time/action component as additional information (like the #acr("RL") environment?)])
]


== Sepsis Definition and the SOFA score
As the result of an...
The Parenchymal (@odep1 and @odek1) and Immune (@odep2 and @odek2) layer and their respective states of the dynamical system naturally are consistent with the two cornerstones of the Sepsis-3 definition @Sepsis3, i.e. #acr("SOFA") score and suspicion of an infection.
Capturing the #acr("SOFA") score @SOFAscore is regularly used to evaluate the severity of an illness and helps to guide treatment decisions and predict the risk of mortality outside of a sepsis context.
While the magnitude or baseline of a patients initials #acr("SOFA") score captures preexisting organ dysfunction, an increase in SOFA score $>=2$ between measurements indicates an acute organ dysfunction and a drastic worseing in the patients condition.
The SOFA score is calculated at least every 24 hours and assess six different organ systems and assigns a score from 0 (normal function) to 4 (high degree of dysfunction) each, as stated in @tab:sofa.
Even thought an increase in #acr("SOFA") score $>=$ 2 is associated with an increase in mortality of about 10% it does not fully capture a septic condition.
To be classified as septic, the presence of an abnormal host reaction against an infection has to be documented or at least suspected, making the full Sepsis-3 definition less restrictive than the plain #acr("SOFA") score, as shown in @SepsisDefinitionComparison.
Also newly introduced in @SOFAscore a bedside clinical score termed quickSOFA (qSOFA): respiratory rate of 22/min or greater, altered mentation, or systolic blood pressure of 100 mm Hg or less.
If a patient fulfills at least two of these criteria have an increased risk of organ failure, but it is not as accurate as the #acr("SOFA") score and is designed as a fast patient screening tool.

Healthy $->$ sync $mean(dot(phi)^1_j)$ and $mean(dot(phi)^1_j)$

#acr("SOFA") $->$ desync $mean(dot(phi)^1_j)$

Suspected infection $->$ splay/desync $mean(dot(phi)^2_j)$

septic $->$ desync $mean(dot(phi)^1_j)$ and $mean(dot(phi)^1_j)$

#figure(
  table(
    columns: (1fr,  auto, auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header(
      [Category], [Indicator], [1], [2], [3], [4]
    ),
    [Respiration], [$"PaO"_2$/$"FiO"_2$ [mmHg]], [< 400], [< 300], [< 200], [< 100],
    [], [Mechanical Ventilation], [], [], [yes], [yes],
    [Coagulation], [Platelets [$times 10^3/"mm"^3$]], [< 150], [< 100], [< 50], [< 20],
    [Liver], [Bilirubin [$"mg"/"dl"$]], [1.2-1.9], [2.0-5.9], [6.0-11.9], [> 12.0],
    [Cardiovascular #footnote("Adrenergica agents administered for at least 1h (doses given are in [μg/kg · min]")], [MAP [mmHg]], [< 70], [], [], [],
    [], [or Dopamine], [], [$<=$ 5], [> 5], [> 15],
    [], [or Dobutamine], [], [any dose], [], [],
    [], [or Epinephrine], [], [], [$<=$ 0.1], [> 0.1],
    [], [or Noepinephrine], [], [], [$<=$ 0.1], [> 0.1],
    [Central Nervous System], [Glasgow Coma Score], [13-14], [10-12], [6-9], [< 6],
    [Renal], [Creatinine [$"mg"/"dl"$]], [1.2-1.9], [2.0-3.4], [3.5-4.9], [> 5.0],
    [], [or Urine Output [$"ml"/"day"$]], [], [], [< 500], [< 200],
  )
) <tab:sofa>

For the cohort extraction and SOFA calculation I use @ricu and @yaib.
The nice thing is we could interpret larger SOFA scores (> 3) as the vulnerable state introduced by @osc2.
Increases in SOFA score $>=2$ could then be used as definition for sepsis.

#TODO[mapping not really clear, which metrics correspond to sofa/infection]
#TODO[YAIB @yaib and other resources care about the "onset" of infection and sepsis @moor_review.
For sepsis this isn't really problematic since we could use the "state transitions" as indicators.
But for the suspected infection it is problematic, maybe use si_upr and si_lwr provided by @ricu (https://eth-mds.github.io/ricu/reference/label_si.html).
These would be 48h - SI - 24h adapted from @sep3_assessment, maybe a bit too much.]


= Introduction

= State of the Art

= Preliminaries

= Own Contribution

= Experimental Results

= Conclusion
