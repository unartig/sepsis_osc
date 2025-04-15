#import "@preview/acrostiche:0.4.0": *
#import "tuhh_template.typ": thesis


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

= Notes
== Base ODE-System
$
dot(phi)^1_i =& omega^1 - 1/N sum^N_(j=1) lr({ (a^1_(i j) + kappa^1_(i j))sin(phi^1_i - phi^1_j + alpha^(11)) }) - sigma sin(phi^1_i - phi^2_i + alpha^(12)) #<odep1> \
dot(kappa)^1_(i j) &= -epsilon^1 (kappa^1_(i j) + sin(phi^1_i - phi^1_j - beta)) #<odek1> \
dot(phi)^2_i =& omega^2 - 1/N sum^N_(j=1) kappa^2_(i j)sin(phi^2_i - phi^2_j + alpha^(22)) - sigma sin(phi^2_i - phi^1_i + alpha^(21)) #<odep2> \
dot(kappa)^2_(i j) &= -epsilon^2 (kappa^2_(i j) + sin(phi^2_i - phi^2_j - beta)) #<odek2>
$ <ode-sys>
Introduced in @osc1 and adapted in @osc2.

== Computational Saving
Using the following reformulation of the parts in the form of $sin(theta_l-theta_m)$ from @KuramotoComp and precalculating the terms $sin(theta_l), sin(theta_m), cos(theta_l), cos(theta_m)$: 
$
sin(theta_l-theta_m)=sin(theta_l)cos(theta_m) - cos(theta_l)sin(theta_m) "    " forall l,m in {1,...,N}
$
the computational cost for the left hand side for $N$ oscillators can be reduced from $N (N-1)$ to $4N$ trigonometric function calls, positively impacting the computational efficency of the whole ODE-system significantly.


== Implementation Differences
#list(
[Actual Mean Phase Velocity instead of averaged difference over time.],
[+They Calculate the difference wrong since the phase difference should be mod[$2pi$]],
[Different solver accuracy, but very similar],
[They are not batching the computation],
)

== Metrics
=== Kuramoto Parameter
Kuramoto Order Parameter CIT
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
to provide useful measures of the two-layer synchronosation behaviour

== Data
MIMIC-3

https://github.com/alistairewj/sepsis3-mimic/tree/v1.0.0



= Introduction

= State of the Art

= Preliminaries

= Own Contribution

= Experimental Results

= Conclusion
