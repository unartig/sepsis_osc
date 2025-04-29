#import "@preview/acrostiche:0.4.0": *
#import "tuhh_template.typ": thesis
#import "@preview/drafting:0.2.2": margin-note, note-outline


#show: thesis.with(
  title: "Comprehensive Guidelines and Templates for Thesis Writing",
  abstract: [
  ],
  abstract_de: [
  ],
  acronyms: (
    "TUHH": ("Hamburg University of Technology"),
    "SOFA": ("Sequential Organ Failure Assessment"),
    "ICU": ("Intensive Care Unit"),
    "EHR": ("Electronic Health Record"),
  ),
  bibliography: bibliography("bibliography.bib"),
  acknowledgements: [
    This thesis was written with the help of many people.
    I would like to thank all of them.
  ],
)

#let mean(f) =  $angle.l$ +  f + $angle.r$
#note-outline()

#let todo = margin-note


= Notes
== Base ODE-System
$
dot(phi)^1_i =& omega^1 - 1/N sum^N_(j=1) lr({ (a^1_(i j) + kappa^1_(i j))sin(phi^1_i - phi^1_j + alpha^(11)) }) - sigma sin(phi^1_i - phi^2_i + alpha^(12)) #<odep1> \
dot(kappa)^1_(i j) &= -epsilon^1 (kappa^1_(i j) + sin(phi^1_i - phi^1_j - beta)) #<odek1> \
dot(phi)^2_i =& omega^2 - 1/N sum^N_(j=1) kappa^2_(i j)sin(phi^2_i - phi^2_j + alpha^(22)) - sigma sin(phi^2_i - phi^1_i + alpha^(21)) #<odep2> \
dot(kappa)^2_(i j) &= -epsilon^2 (kappa^2_(i j) + sin(phi^2_i - phi^2_j - beta)) #<odek2>
$ <eq:ode-sys>
Introduced in @osc1 and slightly adapted in @osc2. #todo[table for parameters]

== Computational Saving
Using the following reformulation of the parts in the form of $sin(theta_l-theta_m)$ from @KuramotoComp and precalculating the terms $sin(theta_l), sin(theta_m), cos(theta_l), cos(theta_m)$: 
$
sin(theta_l-theta_m)=sin(theta_l)cos(theta_m) - cos(theta_l)sin(theta_m) "    " forall l,m in {1,...,N}
$
the computational cost for the left hand side for $N$ oscillators can be reduced from $N (N-1)$ to $4N$ trigonometric function evaluations, positively impacting the computational efficency of the whole ODE-system significantly.


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

=== Fisher Information
In @fisher-info they mention for a two layer network:
$
mean(phi^mu) = 1/N sum^N_j phi^mu_j \

Omega_12 = R^1_2(1+sin(mean(phi^1)-mean(phi^2))) \
Omega_21 = R^2_2(1+sin(mean(phi^2)-mean(phi^1))) \
Delta Omega = Omega_12 - Omega_21 "and " Delta R_2 = R^1_2 - R^2_1
$
to provide useful measures of the two-layer synchronosation behaviour. #todo[actually implement me please]

#todo[Entropy, Splay Ratio, Mean Phase Velocity, MPV Std, Cluster Ratio]

== Data
MIMIC-3

https://github.com/alistairewj/sepsis3-mimic/tree/v1.0.0

== Model
For a general model setup, the latent space $z=(a^1, sigma, alpha, beta, omega^1, omega^2, C/N,epsilon^1, epsilon^2)$ represents the parameter of the dynamic network model, so we have
$
z in RR^d "  with " d = 9
$
As shown in the supplemental material of @osc2, for example, the parameter $alpha$ exhibits a $pi$-periodicity, allowing to reduce the effective parameter space by constraining certain parameters with upper and lower bounds.
These bounds are omittes here for simplicity but are included in #todo[table].
To further reduce the latent space $z$, the we keep $a^1, omega^1, omega^2, C/N, epsilon^1 "and" epsilon^2$ fixed.
The reduced latent space $z'=(sigma, alpha, beta)$:
$
z' in RR^d' "  with " d'=3  
$
where both alpha and beta exhibit a periodic behaviour
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
To make things more complicated, $R$ does not directly act on $z$, but rather the metrics derived from the solution to a dynamical system (initial value problem) (@eq:ode-sys) by $z$.
The metrics are detailed in @sec:metrics.

In the setting of structured latent variational learning we want to approximate an encoder $q(z|x)$ to infer the latent variables from observed data $X$ and the class.
Further we want to learn the latent mappings $R(z)$, while being provided the ground truth of $Q(x)$.


= Introduction

= State of the Art

= Preliminaries

= Own Contribution

= Experimental Results

= Conclusion
