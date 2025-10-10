#import "@preview/acrostiche:0.5.2": *
#import "thesis_template.typ": thesis
#import "@preview/drafting:0.2.2": (
  inline-note, margin-note, note-outline, set-margin-note-defaults,
)
#import "tree.typ": tree_figure

#show: thesis.with(
  title: "Comprehensive Guidelines and Templates for Thesis Writing",
  summary: [
  ],
  // abstract_de: [
  // ],
  acronyms: (
    "TUHH": "Hamburg University of Technology",
    "SOFA": "Sequential Organ Failure Assessment",
    "qSOFA": "Quick Sequential Organ Failure Assessment",
    "ICU": "Intensive Care Unit",
    "EHR": "Electronic Health Record",
    "YAIB": "Yet Another ICU Benchmark",
    "FSQ": "Finite Scalar Quantization",
    "SI": "Suspected Infection",
    "ABX": "Antibiotics",
  ),
  bibliography: bibliography("bibliography.bib"),
  // acknowledgements: [
  //   This thesis was written with the help of many people.
  //   I would like to thank all of them.
  // ],
)

#let mean(f) = $angle.l$ + f + $angle.r$
#note-outline()

#let todo = margin-note
#let caution-rect = rect.with(inset: 1em, radius: 0.5em)
#set-margin-note-defaults(
  rect: caution-rect,
  side: right,
  fill: orange.lighten(80%),
)
#let TODO = inline-note

= Notes
#TODO[actual functional model
  what is learned
  connecting parts
]
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
    // [Introduce the time/action component as additional information (like the #acr("RL") environment?)],
  )
]

For the cohort extraction and SOFA calculation I use @ricu and @yaib.
The nice thing is we could interpret larger SOFA scores (> 3) as the vulnerable state introduced by @osc2.
Increases in SOFA score $>=2$ could then be used as definition for sepsis.

#TODO[mapping not really clear, which metrics correspond to sofa/infection]
#TODO[YAIB @yaib and other resources care about the "onset" of infection and sepsis @moor_review.
  For sepsis this isn't really problematic since we could use the "state transitions" as indicators.
  But for the suspected infection it is problematic, maybe use si_upr and si_lwr provided by @ricu (https://eth-mds.github.io/ricu/reference/label_si.html).
  These would be 48h - SI - 24h adapted from @sep3_assessment, maybe a bit too much.]


= Introduction

= Medical Background (Sepsis)

As the most extreme course of an infectious disease, sepsis poses a serious health threat, with a high mortality rate and frequent long-term consequences for survivors.
In 2017, an estimated 48.9 million people worldwide suffered from sepsis and the same year, 11.0 million deaths were associated with sepsis @rudd2020global, making it the most common cause of in-hospital deaths.
Untreated, the disease is always fatal and even with successful treatment, around 75\% of those affected suffer long-term consequences.
Highlighting the importance of early recognition and treatment of infections in patients with pre-existing health conditions.
Overall, untreated septic diseases in particular represent an enormous burden on the global healthcare system.

The triggers for sepsis are varied, but almost half of all sepsis-related deaths occur as a secondary complication of an underlying injury or a non-communicable, also known as chronic disease @fleischmann2022sepsis.
Faster recognition of a septic condition significantly increases the chance of survival @seymour2017time, it urges to develop accurate and robust detection and prediction methods, i.e. reducing the time to receive the appropriate medical attention.

Per definition, sepsis is a "life-threatening organ dysfunction caused by a
dysregulated host response to infection" @Sepsis3.
There are multiple (now historic) more specific definitions available and sometimes blurry terminology used when dealing with the sepsis and septic shocks.
The following chapter @sec:sep3def gives a more detailed introduction to the most commonly used sepsis definition, which is referred to as Sepsis-3.
Additionally, the chapter @sec:sepbio provides a short introduction of both the pathology and biology of sepsis and @sec:sepwhy talks about the need for reliable sepsis prediction systems.

== Sepsis-3 definition <sec:sep3def>
Out of the need for an update of an outdated and partly misleading sepsis model a task force led by the "Society of Critical Care Medicine and the European Society of Intensive Care Medicine", was formed in 2016.
Their resolution, named "Third International Consensus Definitions for Sepsis and Septic Shock" @Sepsis3, provides until today the most up to date and most widely used sepsis definition and guidance on sepsis identification.

In general sepsis does not classify as a specific illness, rather a condition of "physiologic, pathologic, and biochemical abnormalities" @Sepsis3, where the original causes are still mostly uncertain.
Most commonly the underlying cause of sepsis is diarrhoeal disease, road traffic injury the most common underlying injury and maternal disorders the most common non-communicable disease causing sepsis @rudd2020global.

According to Sepsis-3, sepsis is defined by two criteria: a documented or #acr("SI") and the presence of a dysregulated host response.
This combination represents an exaggerated immune reaction that results in organ dysfunction and potential harm to the body.
When infection is first suspected, even modest organ dysfunction is linked to a 10% increase of in-hospital mortality.

Regarding the first criterion, in clinical practice any patient prescribed with antibiotics followed by the cultivation of body fluids, or the other way around, is characterized with a suspected infection.
The timings of prescription and fluid samplings play a crucial role.
If the antibiotics were administered first, then the cultivation has to be done in the first 24h after first prescription, if the cultivation happened first, the antibiotics have to be prescribed in the following 72h @Sepsis3.
This can be seen in the lower part of figure @fig:ricu, where antibiotics is abbreviated with ABX.
Regardless which happened first, the earlier of the two times is treated as the time of suspected infection onset time.

Regarding the second criterion, a responses dysregulation is characterized by the worsening of organ functionality over time.
For this purpose, the Sepsis-3 consensus has introduced the #acr("SOFA") score (@SOFAscore@Sepsis3#todo[can we fix please?]) and which is now regularly used to evaluate the functionality of organ systems and helps to predict the risk of mortality, also outside of a sepsis context.
The #acr("SOFA") score is calculated at least every 24 hours and assess six different organ systems by assigning a score from 0 (normal function) to 4 (high degree of dysfunction) to each.
The overall score is calculated as sum of each individual system.
It includes the respiratory system, the coagulation/clotting of blood, i.e. changing from liquid to gel, the liver system, the cardiovascular system, the central nervous system and the renal system/kidney function.
A more detailed listing of corresponding markers for each organ assessment can be found in table @tab:sofa in the @sec:appendix.
While the magnitude or baseline of a patients initials #acr("SOFA") score captures preexisting organ dysfunction, an increase in SOFA score $>=2$ between measurements indicates an acute worsening of organ functionalities and a drastic worsening in the patients condition.

An increase of #acr("SOFA") $>=2$ in the 48h before or 24h after the #acr("SI") time, the so called #acr("SI")-window, is per Sepsis-3 definition the "sepsis onset time".
A schematic of all timings is show in figure @fig:ricu.
With respect to which value the increase in #acr("SOFA") is measured is not clearly stated in the consensus and leaves room for interpretation.
Common approaches are the minimal value inside the #acr("SI")-window before the #acr("SOFA") increase, the first value of the #acr("SI")-window, or the lowest value of the 24h previous to the increase.
Differences in precise definitions greatly influence the (retrospective) detection of sepsis, which are used for prevalence estimates for example #todo[cite].
Using the lowest #acr("SOFA") score as baseline the increase $>=2$ for patients with inspected infection was associated with an 18% higher mortality rate.

#figure(
  image("images/sofa-sep-3-1.png", width: 100%),
  caption: [
    Graphical representation of the timings in the Sepsis-3 definition, taken from @ricufig
  ],
)<fig:ricu>

Up until today, even though #acr("SOFA") was created as a clinical bedside score, some of the markers used in it are not always available to measure or at least not at every 24h @moreno2023sofaupdate.
For a faster bedside assessment @SOFAscore also introduced a clinical score termed #acr("qSOFA"), with highly reduced marker number and complexity, it includes:
#(
  list(
    [Respiratory rate $>=$ 22/min],
    [Altered mentation],
    [Systolic blood pressure $<=$ 100 mm Hg],
  )
)
Patients fulfilling at least two of these criteria have an increased risk of organ failure.
The #acr("qSOFA") is not as accurate as the #acr("SOFA") score, meaning it has less statistical significance i.e. #acr("qSOFA") $P<0.01$ vs. #acr("SOFA") $P<0.001$ in a two sided t-test @SOFAscore #todo[against what?].
#todo[risk increase by classification]

#pagebreak()
== Biology of sepsis and Cytokine Storms <sec:sepbio>
The hosts dysregulated response to an infection connected to the septic condition is driven by the release of an unreasonable amount of certain signaling proteins, so called _cytokines_ @Jarczak2022storm.
Cytokines a general family of cells play a special role in the communication and interaction effects between other, both neighboring and distant, cells @Zhang2007cyto.
They are an integrate part of the innate immune system, i.e. the body's first line of non-specific defense @InnateImmuneSystemWiki, where they act as mediators and are required to regulate the elimination of pathogens and trigger the healing process right after.

There are plenty of different specific cytokines, some acting pro- and others anti-inflammatory, but all having the common characteristic of signal mediation @Jarczak2022storm.
One specialty of these relatively simple cells is that they can be produced by immune cells or non-immune cells, with different cells being able to produce the same cytokine.
Further, cytokines are redundant, meaning targeted cells can show identical responses to different cytokines @House2007cyto, this feature seems to fulfill a kind of safety mechanism to guarantee the vital communication flow.
Together after production both have a short half-life (only a few minutes) but through cascading-effects the cytokines can have substantial impact on their micro-environment.
Out of all cytokines, only a very small subset or secondary markers can be measured blood samples to evaluate increased cytokine activity.
This makes them hard to study and little useful as direct indicators of pathogenesis or prediction purposes.


In certain scenarios a disturbance to the regulatory mechanisms triggers a chain reaction, followed by a massive release of cytokines, coupled with self-reinforcement of other regulatory mechanisms @Jarczak2022storm.
This overreaction, called _cytokine storm_, is often harmful to the hosts body and can lead to multi organ failure, like in sepsis, or even death.
In these cases, the damage done by the immune system's reaction is magnitudes greater than the triggering infection iteself.
Even though the quantity of cytokines roughly correlates with disease severity, concentrations of cytokines vary between patients and even different body-parts making a distinction between an appropriate reaction and a harmful one almost impossible @Jarczak2022storm.
Since the 90s there has been a lot of research focused on cytokines and their role in the innit immune system and overall activation behviour, but to this day no breakthrough has been done and underlying principles have not been uncovered.


== The need for sepsis prediction <sec:sepwhy>
== Maybe Treatment

The Parenchymal (@odep1 and @odek1) and Immune (@odep2 and @odek2) layer and their respective states of the dynamical system naturally are consistent with the two cornerstones of the Sepsis-3 definition @Sepsis3, i.e. #acr("SOFA") score and suspicion of an infection.

Healthy $->$ sync $mean(dot(phi)^1_j)$ and $mean(dot(phi)^1_j)$

#acr("SOFA") $->$ desync $mean(dot(phi)^1_j)$

Suspected infection $->$ splay/desync $mean(dot(phi)^2_j)$

septic $->$ desync $mean(dot(phi)^1_j)$ and $mean(dot(phi)^1_j)$

= The Data and Task problems
= Problem definition
In @EiniPorat2022, a survey among clinicians regarding AI-assistance in healthcare, one participant emphasizes that specific vitals signs might not be to be of less importance, rather the change/trend of a patients trajectory.
Another piece of finding of the same study was the preference of trajectories over plain event predictions.


= State of the Art
== Model Based Methods
== Data Based Methods
=== Selected Works

= Dynamic Network Model (DNM)
== Description
=== Functional Models
=== Parenchymal
=== Immune System
=== Kuramoto
== Implementation
=== Standard
=== Lie

= Latent Dynamics Model
== Task - Definition of Ins and Outs
== Data
=== MIMIC-III/IV
=== YAIB + (Further) Preprocessing
==== ricu-Concepts
== Latent Dynamics Model (LDM)
=== The high level ideas
==== Representation Learning and Latent Spaces
==== Semantics
==== Autoregressive Prediction
=== The Lookup (FSQ)
=== Encoder
=== Decoder
=== Introducing time
=== Combining the building blocks
== Metrics (How to validate performance?)

= Experimental Results
== Metrics
== Further Experiments
=== Custom Latent Space
=== SOFA vs Infection

= Conclusion

= Appendix
<sec:appendix>
#figure(
  table(
    columns: (1fr, auto, auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([Category], [Indicator], [1], [2], [3], [4]),
    [Respiration],
    [$"PaO"_2$/$"FiO"_2$ [mmHg]],
    [< 400],
    [< 300],
    [< 200],
    [< 100],

    [], [Mechanical Ventilation], [], [], [yes], [yes],
    [Coagulation],
    [Platelets [$times 10^3/"mm"^3$]],
    [< 150],
    [< 100],
    [< 50],
    [< 20],

    [Liver],
    [Bilirubin [$"mg"/"dl"$]],
    [1.2-1.9],
    [2.0-5.9],
    [6.0-11.9],
    [> 12.0],

    [Cardiovascular #footnote("Adrenergica agents administered for at least 1h (doses given are in [μg/kg · min]")],
    [MAP [mmHg]],
    [< 70],
    [],
    [],
    [],

    [], [or Dopamine], [], [$<=$ 5], [> 5], [> 15],
    [], [or Dobutamine], [], [any dose], [], [],
    [], [or Epinephrine], [], [], [$<=$ 0.1], [> 0.1],
    [], [or Noepinephrine], [], [], [$<=$ 0.1], [> 0.1],
    [Central Nervous System],
    [Glasgow Coma Score],
    [13-14],
    [10-12],
    [6-9],
    [< 6],

    [Renal],
    [Creatinine [$"mg"/"dl"$]],
    [1.2-1.9],
    [2.0-3.4],
    [3.5-4.9],
    [> 5.0],

    [], [or Urine Output [$"ml"/"day"$]], [], [], [< 500], [< 200],
  ),
) <tab:sofa>
#todo[caption]
