#import "poster_template.typ": *
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#import "tree.typ": tree_figure
// #import "idea.typ": mapping_figure

#show: tuhh-poster.with(
  title: "Poster Title",
  header_image: placeholder(
    height: 100%,
    width: 100%,
  ),
  institute_name: "Institute Name (Long)",
  footer: tuhh-footer(
    info: info-3-column-qr(),
  ),
)

#show: box.with(inset: 2em)
#set math.equation(numbering: "(1)")
#let mean(f) =  $angle.l$ +  f + $angle.r$

#columns(
  2,
  [
= Introduction
As the most extreme course of an infectious disease, sepsis poses a serious health threat, with a high mortality rate and frequent long-term consequences for survivors.
In 2017, an estimated 48.9 million people worldwide suffered from sepsis and the same year, 11.0 million deaths were associated with sepsis @rudd2020global.
Untreated, the disease is always fatal and even with successful treatment, around 75% of those affected suffer long-term consequences.
Overall, untreated septic diseases in particular represent an enormous burden on the global healthcare system.

The triggers for sepsis are varied, but almost half of all sepsis-related deaths occur as a secondary complication of an underlying injury or non-communicable disease, highlighting the importance of early recognition and treatment of infections in patients with pre-existing health conditions. 
Faster recognition of the septic condition significantly increases the chance of survival @seymour2017time, it urges to develop accurate and robust detection and prediction methods, i.e. reducing the time to receive the appropriate medical attention.

So-called sepsis scores, for example the SOFA-score @SOFAscore (Sequential Organ Failure Assessment), have been established to simplify the early detection of sepsis in people at risk and to support medical personnel in making a diagnosis.
In recent years, machine and deep learning methods have also been increasingly developed to further increase the effectiveness of sepsis predictions @bomrah2024scoping.
Despite good prediction results, these approaches often suffer from a lack of interpretability, making them difficult to accept in clinical practice.

== Dynamic Network Model and Objective
In this project, we aim to lay the necessary foundation to more interpretable sepsis prediction models by integrating deep learning techniques with a specialized dynamical system designed to model septic conditions.
The “Dynamic Network Model” (DNM), introduced in @osc1 and mathematically defined in @eq:ode-sys, represents the interaction between parenchymal cells (functional organ cells, opposed to stroma, the structural cells), denoted $phi^1_i$, and immune cells, denoted $phi^2_i$, using a two-layered, partly adaptive oscillator framework.
$
dot(phi)^1_i =& omega^1 - 1/N sum^N_(j=1) lr({ (a^1_(i j) + kappa^1_(i j))sin(phi^1_i - phi^1_j + alpha^(11)) }) - sigma sin(phi^1_i - phi^2_i + alpha^(12)) \
dot(kappa)^1_(i j) &= -epsilon^1 (kappa^1_(i j) + sin(phi^1_i - phi^1_j - beta)) \
dot(phi)^2_i =& omega^2 - 1/N sum^N_(j=1) {kappa^2_(i j)sin(phi^2_i - phi^2_j + alpha^(22))} - sigma sin(phi^2_i - phi^1_i + alpha^(21)) \
dot(kappa)^2_(i j) &= -epsilon^2 (kappa^2_(i j) + sin(phi^2_i - phi^2_j - beta))
$ <eq:ode-sys>
Communication between the two layers in the biological system via Cytokines is modeled by adaptive coupling weights $kappa^(1"/"2)_(i j)$.
It is hypothesized, that synchronization patterns and other dynamic states within emerging from this system provide valuable insights into a patient's condition @osc1.

Our approach learns patient-specific DNM parameters from medical record data, with these parameters being hinterpretable:
#list(
[$alpha$: represents inter-layer metabolic interaction delay],
[$beta$ a generalized "medical age"],
[$sigma$ controls the interaction strengths between layers],
indent: 2em)


// Patient classification is enabled by mapping the #text([cluster ratio], weight: "bold"), defined in @eq:cluster-ratio, of the patient in parameter space to the SOFA-score.
// $
// f^mu (alpha, beta, sigma) := 1/N_E sum_i bb(1)[exists j: |phi^mu_(j)(t_"max") - 1/N sum_k phi^mu_(k)(t_"max")| > epsilon]
// $ <eq:cluster-ratio>
// Where $N_E$ is the number of ensemble members, $N$ the number of oscillators per layer, and $epsilon$ th
// #figure(image("images/tree.svg", width: 100%), caption:[])

= Methodology
//#figure(image("images/idea.svg", width: 80%), caption:[])
Given preprocessed medical input features $(mu_1, ..., mu_n)$a neural encoder $N_theta$ estimates the model parameters $(alpha, beta, sigma)$:
$
(alpha, beta, sigma) = cal(N)^"Enc"_theta (mu_1, ..., mu_n)
$ <eq:enc>
 
Patient classification is enabled by mapping the #text([ensemble average of the standard deviations of the mean phase velocities], weight: "bold"), defined in @eq:std, of the patient in parameter space to the SOFA-score.
Where each ensemble member represents a stochastic realization of the same patient-specific DNM under varying initial conditions
$
mean(phi^mu) = 1/N sum^N_j phi^mu_j \
s^mu &=  1/N_E sum_E sqrt(1/N sum^N_j (mean(dot(phi)^mu_j) - overline(omega)^mu)^2) \
g&: s^mu (alpha, beta, sigma) -> hat(y)
$ <eq:std>

$
cal(L)_"total" (mu_1, ..., mu_n, y) &= cal(L)_"concept" [f^1(alpha, beta, sigma), y] + cal(L)_"locality" (alpha, beta, sigma, bold(mu))
$

= Future Work
There are two main aspects to be tackled in the future, firstly the introduction of detected/suspected infection, which besides the SOFA-score is other decisive factor for a sepsis diagnosis.
The DNM already includes the notion of infection by clustered or splayed states in its immune layer ($mu=2$).

Secondly, it is proposed that tracking a patient's trajectory through the lower dimensional parameter-space to reflect clinical progression over time.
This potentially allows the extrapolation of pathological developments, treatment responses and prediction of critical conditions.

#[
#set text( size: 20pt)
#bibliography("bibliography.bib")
]
  ],
)

