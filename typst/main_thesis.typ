#import "@preview/acrostiche:0.5.2": *
#import "thesis_template.typ": thesis
#import "@preview/drafting:0.2.2": (
  inline-note, margin-note, note-outline, set-margin-note-defaults,
)
#import "figures/tree.typ": tree_fig
#import "figures/fsq.typ": fsq_fig
#import "figures/kuramoto.typ": kuramoto_fig
#import "figures/helper.typ": cmalpha, cmbeta, cmred, cmsigma
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
    "DNM": "Dynamic Network Model",
    "LDM": "Latent Dynamics Model",
    "ML": "Machine Learning",
    "DL": "Deep Learning",
    "ODE": "Ordinary Differential Equation",
    "JIT": "Just In Time Compilation",
    "GPU": "Graphics Processing Unit",
    "PID": "Proportional-Integral-Derivative",
  ),

  bibliography: bibliography("bibliography.bib"),
  // acknowledgements: [
  //   This thesis was written with the help of many people.
  //   I would like to thank all of them.
  // ],
)

#let mean(f) = $chevron.l$ + f + $chevron.r$
#let ot = $1"/"2$
#note-outline()

#let todo = margin-note
#let caution-rect = rect.with(inset: 1em, radius: 0.5em)
#set-margin-note-defaults(
  rect: caution-rect,
  side: right,
  fill: orange.lighten(80%),
)
#let TODO = inline-note


#TODO[
  #list(
    [Sections to Chapters],
    [Styling],
    [Appendix to real Appendix],
    [Fix ACR seperation],
    [Fix newline/lineabreak after Headings],
  )
]
= Notes
#TODO[actual functional model
  what is learned
  connecting parts
]

= Introduction

= Medical Background (Sepsis) <sec:sepsis>

As the most extreme course of an infectious disease, sepsis poses a serious health threat, with a high mortality rate and frequent long-term consequences for survivors.
In 2017, an estimated 48.9 million people worldwide suffered from sepsis and the same year, 11.0 million deaths were associated with sepsis @rudd2020global, which makes up 19.7% of yearly deaths.
Sepsis is also the most common cause of in-hospital deaths.
Untreated, the disease is always fatal and even with successful treatment, around 40\% of those affected suffer long-term consequences, such as cognitive, physical or physiological problems, the so called _post-sepsis syndrome_ @vanderSlikke2020post.
Overall, treated and untreated septic diseases in particular represent an enormous burden on the global healthcare system.

The triggers for sepsis are varied, but almost half of all sepsis-related deaths occur as a secondary complication of an underlying injury or a non-communicable, also known as chronic disease @fleischmann2022sepsis.
A recent study @seymour2017time highlights the importance of early recognition and subsequent treatment of infections in patients, reducing the mortality risk caused from sepsis.
Each hour of earlier detection can significantly increase the chance of survival @seymour2017time, it urges to develop accurate and robust detection and prediction methods, i.e. reducing the time to receive the appropriate medical attention.

Per definition, sepsis is a "life-threatening organ dysfunction caused by a
dysregulated host response to infection" @Sepsis3.
There are multiple (now historic) more specific definitions available and sometimes blurry terminology used when dealing with the sepsis and septic shocks.
The following @sec:sep3def gives a more detailed overview to the most commonly used sepsis definition, which is referred to as Sepsis-3.
Additionally, @sec:sepbio provides a short introduction of both the pathology and biology of sepsis.
Lastly, in @sec:sepwhy the necessity for reliable sepsis prediction systems is discussed.


== The Sepsis-3 Definition <sec:sep3def>
Out of the need for an update of an outdated and partly misleading sepsis model a task force led by the "Society of Critical Care Medicine and the European Society of Intensive Care Medicine", was formed in 2016.
Their resolution, named "Third International Consensus Definitions for Sepsis and Septic Shock" @Sepsis3, provides until today the most widely used sepsis definition and guidance on sepsis identification.

In general, sepsis does not classify as a specific illness, rather a multifaceted condition of "physiologic, pathologic, and biochemical abnormalities" @Sepsis3, and septic patients are largely heterogeneous.
Most commonly the underlying cause of sepsis is diarrhoeal disease, road traffic injury the most common underlying injury and maternal disorders the most common non-communicable disease causing sepsis @rudd2020global.

According to the Sepsis-3 definition, a patient is in a septic condition if the following two criteria are fulfilled:
#(
  align(center, list(
    align(left, [a documented or #acr("SI") and]),
    align(left, [the presence of a dysregulated host response]),
  ))
)
The combination of the two criteria represents an exaggerated immune reaction that results in organ dysfunction, when infection is first suspected, even modest organ dysfunction is linked to a 10% increase of in-hospital mortality.
A more pathobiological explanation of what a "dysregulated host response" means is given in the next @sec:sepbio.

*Confirmed or Suspected Infection* is suggested to characterize any patient prescribed with #acr("ABX") followed by the cultivation of body fluids, or the other way around, with a suspected infection.
The timings of prescription and fluid samplings play a crucial role.
If the antibiotics were administered first, then the cultivation has to be done in the first 24h after first prescription, if the cultivation happened first, the #acr("ABX") have to be prescribed in the following 72h @Sepsis3.
This can be seen in the lower part of figure @fig:ricu, with the abbreviated #acr("ABX").
Regardless which happened first, the earlier of the two times is treated as the time of suspected infection onset time.

*Dysregulated Host Response* is characterized by the worsening of organ functionality over time.
Since there is no gold standard for measuring the amount of "dysregulation" the Sepsis-3 consensus relies on the #acr("SOFA")-score introduced in (@SOFAscore@Sepsis3#todo[can we fix please?]).
The score is now regularly used to evaluate the functionality of organ systems and helps to predict the risk of mortality, also outside of a sepsis context.
The #acr("SOFA") score is calculated at least every 24 hours and assess six different organ systems by assigning a score from 0 (normal function) to 4 (high degree of dysfunction) to each.
The overall score is calculated as sum of each individual system.

It includes the respiratory system, the coagulation/clotting of blood, i.e. changing from liquid to gel, the liver system, the cardiovascular system, the central nervous system and the renal system/kidney function.
A more detailed listing of corresponding markers for each organ assessment can be found in table @tab:sofa in the @sec:appendix.
The magnitude of a patients initial #acr("SOFA")-score captures preexisting organ dysfunction.
An increase in SOFA score $>=2$ corresponds to an acute worsening of organ functionalities and a drastic worsening in the patients condition, the indicator for a dysregulated response.

=== Sepsis Classification
The Sepsis-3 definition not only provides the clinical critera of septic conditions, but also introduces the necessary time windows for sepsis classification.
An increase of #acr("SOFA") $>=2$ in the 48h before or 24h after the #acr("SI") time, the so called #acr("SI")-window, is per Sepsis-3 definition the "sepsis onset time".
A schematic of the timings is shown in figure @fig:ricu.

With respect to which value the increase in #acr("SOFA") is measured, i.e. the baseline score, is not clearly stated in the consensus and leaves room for interpretation, commonly used approaches include:
#(
  align(center, list(
    align(
      left,
      [the minimal value inside the #acr("SI")-window before the #acr("SOFA") increase,],
    ),
    align(left, [the first value of the #acr("SI")-window,]),
    align(left, [or the lowest value of the 24h previous to the increase.]),
  ))
)
Differences in definitions greatly influence the detection of sepsis, which are used for prevalence estimates for example @Johnsons2018Data.
Using the lowest #acr("SOFA") score as baseline, the increase $>=2$ for patients with inspected infection was associated with an 18% higher mortality rate according to @SOFAscore a retrospective #acr("ICU")-data analysis.

#figure(
  image("images/sofa-sep-3-1.png", width: 100%),
  caption: [
    Graphical representation of the timings in the Sepsis-3 definition, taken from @ricufig
  ],
)<fig:ricu>

Up until today, even though #acr("SOFA") was created as a clinical bedside score, some of the markers used in it are not always available to measure or at least not at every 24h @moreno2023sofaupdate.
For a faster bedside assessment @SOFAscore also introduced a clinical score termed #acr("qSOFA"), with highly reduced marker number and complexity, it includes:
#(
  align(center, list(
    align(left, [Respiratory rate $>=$ 22/min]),
    align(left, [Altered mentation]),
    align(left, [Systolic blood pressure $<=$ 100 mm Hg]),
  ))
)
Patients fulfilling at least two of these criteria have an increased risk of organ failure.
While the #acr("qSOFA") has a significantly reduced complexity and is faster to assess it is not as accurate as the #acr("SOFA") score, meaning it has less predictive validity for in-house mortality @SOFAscore.

== Biology of Sepsis <sec:sepbio>
This part tries to give an introduction into the biological phenomena that underlie sepsis.
First we take a look on the way tissue is reacting to local infections or injuries on a cellular level in @sec:cell and how this escalates to _cytokine storms_ in @sec:storm and this ends with systemic organ failure in @sec:fail.

Certain details and specificities are left out when not essential for the understanding of this project.
The interested reader should refer to the primary resources provided throughout this section.
=== Cellular Description <sec:cell>
Human organ tissue can be differentiated into two broad cell-families called _parenchymal_ and _stroma_ which are separated by a boundary consisting of _basal lamina_.
The parenchymal cells conduct the specific function of the organ, with every organ hosting distinct parenchymal cells, everything else is part of the stroma, including the structural or connective tissue, blood vessels and nerves.
When a pathogen enters the body the first line of non-specific defense, the innate immune system @InnateImmuneSystemWiki, gets activated.
Besides the so called resident-immune-cells (most prominently macrophages) also the stroma cells are able to detect the pathogen via pattern-recognition-receptors and will start releasing signaling proteins, so called _cytokines_ @Zhang2007cyto.

Cytokines are a diverse group of signaling proteins which play a special role in the communication between other, both neighboring and distant cells, and will attract circulating immune cells @Zhang2007cyto.
Generally cytokines, besides being involved in the growing process of blood cells, regulate the production of anti- and pro-inflammatory immune cells which help with the elimination of pathogens and trigger the healing process right after.
One specialty of these relatively simple proteins is that they can be produced by almost every other cell, with different cells being able to produce the same cytokine.
Further, cytokines are redundant, meaning targeted cells can show identical responses to different cytokines @House2007cyto, these features seems to fulfill some kind of safety mechanism to guarantee vital communication flow.
After release cytokines have relatively a short half-life (only a few minutes) but through cascading-effects the cytokines can have substantial impact on their micro-environment.

=== Cytokine Storms <sec:storm>
The hosts dysregulated response to an infection connected to the septic condition is driven by the release of an unreasonable amount of cytokines.
Normally, the release of inflammatory cytokines automatically fades out once the initial pathogen is controlled and the host returns to a healthy and balanced state, the _homeostasis_.
In certain scenarios a disturbance to the regulatory mechanisms triggers a chain reaction, followed by a massive release of cytokines.
It is further coupled with self-reinforcement of other regulatory mechanisms @Jarczak2022storm, leading to a continuous and uncontrolled release of cytokines that fails to shut down.
With this overreaction, called _cytokine storm_, the immune system's reaction damages the body while being magnitudes greater than the triggering infection itself.

Even though the quantity of cytokines roughly correlates with disease severity, concentrations of cytokines vary between patients and even different body-parts making a distinction between an appropriate reaction and a harmful one almost impossible @Jarczak2022storm.
Out of all cytokines, only a very small subset or secondary markers can be measured through blood samples to detect increased cytokine activity.
This makes them hard to study in general and little useful as direct indicators of pathogenesis or prediction purposes.
Since the 90s there has been a lot of research focused on cytokines and their role in the innate immune system and overall activation behavior.
But to this day no breakthrough has been done and underlying principles have not been uncovered.

=== Systemic Consequences and Organ Failure <sec:fail>
While more and more cytokines flood not only the infected areas, surrounding parts of the tissue and circulation are also affected.
This disrupts the metabolism of parenchymal cells due to a deficiency in oxygen and nutrients.
The cells switch from an oxygen-based metabolism to an anaerobic glycolysis @Prieto2016Anaerobic, generating energy less efficiently from glucose.
As a result, metabolic by-products such as lactate accumulate, leading to cellular dysfunction.
At the same time, the cells' mitochondria start to fail, blood vessels become leaky and tiny blood cogs form, further reduce cell functionality.
These processes cause progressive cell death and ultimately organ failure.
When multiple organs fail simultaneously, the condition becomes irreversible @Sepsis3.
Multi-organ-failure is the final and most lethal stage of sepsis, with each additional affected organ the mortality increases drastically.


== The need for sepsis prediction <sec:sepwhy>
#TODO[Important to finish]



= Problem definition <sec:problemdef>

This section provides some background on the specific research questions which are investigated in @sec:experiment using the methods introduced in @sec:dnm and @sec:ldm respectively.
As discussed in @sec:sepwhy, there is a substantial need for robust methods to identify patients sepsis onset and overall progression.
This work provides a proof of concept for such a prediction system.

The increasing availability of high-quality medical data, i.e. multiple physiological markers with high temporal resolution, enables both classical statistical and #acr("ML") (including #acr("DL")) methods (see @sec:sota).
While these purely data-driven approaches often achieve acceptable performance but the explainability of the prediction suffers and limits their adoption in clinical practice #todo[cite].

In parallel, recent advances in the field of network physiology have introduced new ways to model physiological systems as interacting subsystems rather than isolated organs @Ivanov2021Physiolome.
The #acr("DNM") introduced in @osc1 and adapted in @osc2, allows for a functional description of organ failure in sepsis and shows realistic system behavior in preliminary analysis.
An in-depth introduction to the #acr("DNM") is provided in @sec:dnm.
But up until now the dynamic model has not yet been verified on real data.
We want to investigte how real patients would translate to the model parameters, and how the temporal physiological evolution can be incorporated and if there is a benefit doing so.
// However, this model has not yet been validated against real-world observations, which will be addressed in this work #todo[eher project???].


To summarize, the specific research questions include:
#(
  list(
    [*Usability of the #acr("DNM")*: How and to what extent can the #acr("ML")-determined trajectories of the #acr("DNM") be used for detection and prediction, especially of critical infection states and mortality.],
    [*Comparison with data-based approaches*: How can the model-based predictions be compared with those of purely data-based approaches in terms of predictive power and interpretability.],
  )
)
#TODO[End this]

= The Data and Task problems

In @EiniPorat2022, a survey among clinicians regarding AI-assistance in healthcare, one participant emphasizes that specific vitals signs might not be to be of less importance, rather the change/trend of a patients trajectory.
Another piece of finding of the same study was the preference of trajectories over plain event predictions.
#figure(
  image("images/yaib_sets.svg", width: 100%),
  caption: [
    Sets of @yaib
  ],
)<fig:sets>
RICU and YAIB use delta_cummin function, i.e. the delta #acr("SOFA") increase is calculated with respect to the lowest observed #acr("SOFA") to this point.

= State of the Art <sec:sota>
== Model Based Methods
== Data Based Methods
=== Selected Works

= Dynamic Network Model (DNM) <sec:dnm>

As outlined in @sec:sepsis, the macroscopic multi-organ failure associated with sepsis is driven by a dysregulated cascade of signaling processes on a microscopic level (see @sec:sepbio).
This cascade involves a massive amount of interconnected components, where the connections mechanics and strengths vary over time and space.
For example, these interactions differ across tissues and evolve as sepsis progresses, with crossing biochemical thresholds the behavior of cells can be changed @Callard1999Cytokines.

In essence, cell-to-cell and cell-to-organ interaction in septic conditions form a highly dynamic, nonlinear and spatio-temporal network of relationships @Schuurman2023Complex, which cannot be fully understood by a reduction to single time-point analyzes.
Even though many individual elements of the inflammatory response are well characterized, we still fail to integrate them into a coherent system-level picture.

To address this complexity, the emerging field of _Network Physiology_ provides a promising conceptual framework.
Rather than studying components in isolation, network physiology focuses on the coordination and interconnection among the diverse organ systems and subsystems @Ivanov2021Physiolome.
It enables the study of human physiology as a complex, integrated system, where emergent macroscopic dynamics arise from interacting subsystems that cannot be explained by their individual behavior.
This perspective translates to the mesoscopic level, i.e. the in-between of things, where the coupling mechanisms collectively determine the overall physiological function.

In network physiology, the analytical basis of the bodies interacting systems is often graph based.
Nodes represent subsystem such as organs or cell populations and links represent functional couplings or communication pathways @Ivanov2021Physiolome.
Unlike classical graph theory, where dynamics are introduced by changing the graph topology (e.g. adding or removing links or nodes), in _Complex Networks_ the links themselves can evolve dynamically in response to other system variables.
These adaptive connectivities allow for information to propagate through the whole network, giving rise to emerging phenomena on global scales for otherwise identical network topologies.

Complex networks are well studied in physics and biology and have been applied to various physiological dimains.
Early works, such as @Guyton1972Circulation that have studied the cardiovascular system, while more recent studies have focused on the cadio-respiratory coupling @Bartsch2012Phase and large-scale brain network dynamics @Lehnertz2021Time.
Network approaches have also provided mechanistic insights into disease dynamics, for example Parkinson @Asl2022Parkinson and Epilepsy @Simha2022Epilepsy, just to name a few.

Building on these interaction centric principles has opened up new opportunities to study how the inflammatory processes, such as those underlying sepsis, emerge from the complex inter- and intra-organ communication.
In particular @osc1 and @osc2 have introduced a dynamical system that models the cytokine behavior in patients with sepsis and cancer.
This functional model will be referred to as #acl("DNM") and forms the conceptual foundation for this whole project.

The remainder of this chapter is structured as follows: In @sec:kuramoto introduces the theoretical backbone of the #acr("DNM"), the Kuramoto oscillator model, which provides a minimal description of synchronization phenomena in complex systems.
@sec:dnmdesc presents the formal mathematical definition of the #acr("DNM") and its medical interpretation, followed by implementation details in @sec:dnmimp and a presentation of selected simulation results in @sec:dnmres.

== Theoretical Background: The Kuramoto Oscillator Model <sec:kuramoto>
To mathematically describe natural or technological phenomena, _coupled oscillators_ have proven to be a useful framework @Placeholder, for example, to model the relative timing of neural spiking, reaction rates of chemical systems or dynamics of epidemics @Placeholder.
In these cases complex networks of coupled oscillators are often capable of bridging microscopic dynamics and macroscopic synchronization phenomena observed in biological systems.

One of the most influential system of coupled oscillators is the _Kuramoto Phase Oscillator Model_ which is often used to study how synchronization emerges from simple coupling rules.
In the simplest form it consists of $N$ identical, fully connected and coupled oscillators with phase $phi_i in [0, 2pi), " for" i in 1...N$ and an intrinsic frequency $omega_i$ @Placeholder.
The dynamics are given by:
$
  dot(phi)_i = omega_i - Kappa/N sum^N_(j=1) sin(phi_i - phi_j)
$ <eq:kuramoto>

Here the $dot(phi)$ is used as shorthand notation for the time derivative of the phase $(d phi)/(d t)$, the instantaneous phase velocity.
An additional parameter is the global coupling strength $Kappa$ between oscillators $i$ and $j$.

The model captures the essential mechanism of self-synchronization, and a fundamental collective transition from disorder to order, that underlie many real world processes, which is the reason the model has attracted so much research.
When evolving this system with time, oscillator $i$'s phase velocity depends on each other oscillator $j$.
If $phi_j > phi_i$ the phase oscillator $i$ accelerates $dot(phi)_i > 0$, if $phi_j < phi_i$ decelerates.
For sufficiently large $N$ the oscillator population can converge towards system-scale states of coherence or incoherence based on the choice of $Kappa$.
Coherent in this case means oscillators synchronize with each other, so they share the same phase and phase velocity, incoherence on the other hand is the absence of synchronization (desynchronized), see @fig:sync.
Synchronous states can be reached if the coupling is stronger than a certain threshold $Kappa>Kappa_c$, the critical coupling strength.
In between these two regimes there is a transition-phase of partial synchronization, where some oscillators phase- and frequency-lock and others do not.

#figure(
  kuramoto_fig,
  caption: [Schematic transition between the two stable regimes for the basic Kuramoto model. From an incoherent system state with desynchronized oscillators (heterogeneous phases and frequencies), to a synchronized system state with phase- and frequency-locked oscillators with increasing coupling strength $Kappa$).],
) <fig:sync>


=== Extensions to the Kuramoto Model <sec:extent>
To more accurately describe real world systems, various extensions of the basic Kuramoto model have been proposed and studied numerically and analytically.
Several extensions are directly relevant to the #acr("DNM") and their definitions and effects on synchronization will be shortly introduced:

*Phase Lag $alpha$* introduced in @Placeholder (Kuramoto Sakaguchi 86) #todo[cite], brings a frustration into the synchronization process:
$
  dot(phi)_i = omega_i - Kappa/N sum^N_(j=1) sin(phi_i - phi_j cmred(+ alpha))
$
Positive values of $alpha$ act as an inhibitor of synchronization by shifting the coupling function, so the coupling does not vanish even when the phases align.
As a result the critical coupling strength $K_c$ increases with $alpha$.

*Adaptive coupling $bold(Kappa) in RR^(N times N)$* moves from a global coupling strength $Kappa$ for all oscillator pairs to an adaptive coupling strength for each individual pair $kappa_(i j)$:
$
  dot(phi)_i = omega_i - 1/N sum^N_(j=1) cmred(kappa_(i j)) sin(phi_i - phi_j) \
  cmred(dot(kappa)_(i j) = - epsilon (kappa_(i j) + sin(phi_i - phi_j + beta^mu)))
$ <eq:kurasaka>
Here adaption rate $0 < epsilon << 1$ separates the fast moving oscillator dynamics from slower moving coupling adaptivity @Berner2020Birth.
Such adaptive couplings have been used to model neural plasticity and learning-like processes in physiological systems @Placeholder.
The so called new phase lag parameter $beta$ of the adaptation function (also called plasticity rule) plays an essential role.
At a value of $beta^mu=pi/2$ the coupling, and therefore the adaptivity, is at a maximum positive feedback, strengthening the link $kappa_(i j)$ (Hebbian Rule: fire together, wire together @Berner2020Birth) and encouraging synchronization between oscillators $i$ and $j$.
For other values $beta^mu != pi/2$ the feedback is delayed $phi^(mu)_i-phi^(nu)_j=beta^mu-pi/2$ by a phase lag, a value of $beta^mu=-pi/2$ we get an anti-Hebbian rule which inhibits synchronization.

*Multiplex Networks* represent systems with multiple interacting layers.
Multiplexing introduces a way how several Kuramoto networks can be coupled via interlayer links:
$
  dot(phi)_i^cmred(mu) = omega_i - Kappa/N sum^N_(j=1) sin(phi_i - phi_j cmred(+ alpha^(mu mu))) cmred(- sigma^(mu nu) sum^L_(nu=1, nu!=mu) sin(phi_i^mu - phi_i^nu + alpha^(mu nu)))
$
Here $mu$ and $nu$ represent distinct subsystems, and are connected via interlayer coupling weights $sigma^(mu nu)$, acting one-to-one.\

These extensions combined serve as the source of dynamics for the #acr("DNM") and give rise to more intricate system states than the straightforward synchronization in the base model.
Even for single layers, non-multiplexed but phase-lagged and adaptively coupled oscillators, one can observe several distinct system states, neither fully synchronized or desynchronized such as phase and frequency-clusters, chimera- and splay states.
The emergence of these states depends on the choice of the coupling strength $Kappa$ and the phase-lag parameters $alpha$ and $beta$.

In the frequency clustered state, the oscillator phases do not synchronize, but several oscillator groups can form that share a common frequency.
For the phase-clustered case, the groups additionally synchronize their phase.
Frequency clusters often emerge as intermediate regimes between full synchronization and incoherence @Berner2019Hiera.

Chimera states, a special type of partial synchronization, occur when only a subset of oscillators synchronizes in phase and frequency, while others remain desynchronized.
In contrast to "normal" partial synchronization they occur when the coupling symmetry breaks.
In splay states, all oscillators synchronize their frequencies but do not their phases, they instead uniformly distribute around the unit circle @Berner2020Birth.

The introduction changes the system behavior once more, for example single layers of a multiplexed system can result in the multi-clustered regime for parameters they wouldn't in the monoplexed case.
In multiplexed systems it is also possible connected layers end up in different stable state, for example, one in a clustered the other in a splay state.

== Description <sec:dnmdesc>
The #acr("DNM") is a *functional* model, that means it *does not try to model things accurately on any cellular, biochemical, or organ level*, it instead tries to model dynamic interactions.
At the core, the model does differentiate between two broad classes of cells, introduced in @sec:cell, the stroma and the parenchymal cells.
It also includes the cell interaction through cytokine proteins and an information flow through the basal membrane.
Importantly, the model only handles the case of already infected subjects and tries to grasp if the patients state is prone to a dysregulated host response.

Cells of one type are aggregated into layers, everything associated with parenchymal cells is indicated with an $""^1$ superscript and is called the _organ layer_, stroma cells are indicated with $""^2$ and is referred to as non specific _immune layer_.
Each layer consists of $N$ phase oscillators $phi^ot_i in [0, 2pi)$.
To emphasize again the function aspect of the model: individual oscillators do not correspond to single cells, rather the layer as a whole is associated with the overall state of all organs or immune system functionality respectively.

The metabolic cell activity is modeled by rotational velocity $dot(phi)$ of the oscillators, the faster the rotation, the faster the metabolism.
Each layer is fully coupled via an adaptive possibly asymmetric matrix $bold(Kappa)^ot in [-1, 1]^(N times N)$ with elements $kappa^ot_(i j)$, these couplings represent the activity of cytokine mediation.
Small absolute coupling values indicate a low communication via cytokines and grows with larger coupling strength.
For the organ layer there is an additional non-adaptive coupling part $bold(A)^1 in [0, 1]^(N times N)$ with elements $a^1_(i j)$, representing a fixed connectivity within an organ.

The dimensionless system dynamics are described with the following coupled #acr("ODE") terms, build on the classical Kuramoto model described in @sec:kuramoto and its extensions from @sec:extent:

$
  dot(phi)^1_i =& omega^1 - 1/N sum^N_(j=1) lr({ (a^1_(i j) + kappa^1_(i j))sin(phi^1_i - phi^1_j + alpha^(11)) }) - sigma sin(phi^1_i - phi^2_i + alpha^(12)) #<odep1> \
  dot(kappa)^1_(i j) &= -epsilon^1 (kappa^1_(i j) + sin(phi^1_i - phi^1_j - beta)) #<odek1> \
  dot(phi)^2_i =& omega^2 - 1/N sum^N_(j=1) kappa^2_(i j)sin(phi^2_i - phi^2_j + alpha^(22)) - sigma sin(phi^2_i - phi^1_i + alpha^(21)) #<odep2> \
  dot(kappa)^2_(i j) &= -epsilon^2 (kappa^2_(i j) + sin(phi^2_i - phi^2_j - beta)) #<odek2>
$ <eq:ode-sys>
Where the interlayer coupling, i.e. a symmetric information through the basal lamina, is modeled by the parameter $sigma in RR_(>=0)$.
The internal oscillator frequencies are modeled by the parameters $omega^ot$ and correspond to a natural metabolic activity.

Besides the coupling weights in $bold(Kappa)^ot$ the intralayer interactions also depend on the phase lag parameters $alpha^11$ and $alpha^22$ modeling cellular reaction delay.
To separate the fast moving oscillator dynamics from the slower moving coupling weights adaption rates $0 < epsilon << 1$ are introduced.
Since the adaption of parenchymal cytokine communication is assumed to be slower than the immune counterpart @osc1, it is chosen $epsilon^1 << epsilon^2 << 1$, which introduces dynamics on multiple timescales.

Lastly, the most influential parameter is $beta$ which controls they adaptivity of the cytokines.
Because $beta$ has such a big influence on the model dynamics it is called the _(biological) age parameter_ and summarizes multiple physiological concepts such as age, inflammatory baselines, adiposity, pre-existing illness, physical inactivity, nutritional influences and other common risk factors @osc2.

All the systems variables and parameters are summarized in <tab:dnm> #todo[why no ref?] together with their medical interpretation.
#figure(
  table(
    columns: (auto, auto, auto),
    // inset: 10pt,
    align: center,
    table.header([*Symbol*], [*Name*], [*Physiological Meaning*]),
    table.cell(colspan: 3)[*Variables*],
    [$phi_i$], [Phase], [Group of cells],
    [$dot(phi)_i$], [Phase Velocity], [Metabolic activity],
    [$kappa_(i j)$], [Coupling Weight], [Cytokine activity],

    table.cell(colspan: 3)[*Parameters*],
    [$alpha$], [Phase lag], [Metabolic interaction delay],

    [$beta$],
    [Plasticity rule],
    [Combined of risk factors],
    // [Age, inflammation, pre-existing illness, other risk factor],

    [$omega$],
    [Natural frequency],
    [Natural cellular metabolism],

    [$epsilon$], [Time scale ratios], [Temporal scale of cytokine activity],
    // [$C$], [Initial network pertubation], [-],

    [$a_(i j)$],
    [Connectivity],
    [Fixed intra-organ cell-to-cell interaction],

    [$sigma$],
    [Interlayer coupling],
    [Interaction between parenchymal and \ immune cells through the basal lamina],

    table.cell(colspan: 3)[*Measures*],
    [$s$],
    [Standard deviation of frequency \ (see @eq:std)],
    [Pathogenicity (Parenchymal Layer)],
  ),
  caption: [todo],
) <tab:dnm>
#todo[left out superscripts for better readability]

=== Pathology in the DNM
A biological organism, such as the human body, can be regarded as a self-regulating system that, under healthy conditions, maintains a homeostatic state @Placeholder.
Homeostasis refers to a dynamic but balanced equilibrium in which the physiological subsystems continuously interact to sustain stability despite external perturbations.
In the context of the #acr("DNM"), this equilibrium is represented by a synchronous regime of both layers in the duplex oscillator system.
In synchronous states, the organ layer and immune layer exhibit coordinated phase and frequency dynamics, reflecting balanced communication, collective frequency of cellular metabolism and stable systemic function.

Pathology, in contrast, is modeled by the breakdown of the synchronicity and the formation of frequency clusters in the parenchymal layer, i.e. loss of homeostatic balance.
In the #acr("DNM") least one cluster will exhibit increased frequency and one with lower or unchanged frequency.
This aligns with medical observation, where unhealthy parenchymal cells change to a less efficient anaerobic glycosis based metabolism, forcing them to increase their metabolic activity to keep up with the energy demand.
Remaining healthy cells are expected to stay frequency synchronized to a lower and "healthy" frequency.

There are two more cases, neither fully healthy nor fully pathologic, representing a vulnerable or resilient patient condition.
The healthy but vulnerable case corresponds to a splay state, where phases in the parenchymal layer are not synchronized, but the frequencies are, weakening the overall coherence @osc2.
A resilient state corresponds to cases where both the phase and frequency of the parenchymal layer are synchronized, but the immune layer exhibits both frequency and phase clustering.

// #figure(tree_fig)

== Implementation <sec:dnmimp>
For initial value problems of coupled #acr("ODE")-systems, such as the #acr("DNM"), analytical solutions rarely exist @osc2, mostly for trivial or other special configurations or by applying aggressive simplifications.
Instead, one traditionally relies on the numerical integration of the system, approximating the analytical solution.

This subsection describes the numerical implementation of the #acr("ODE")-system defined in @eq:ode-sys, the choice of initial parameter values and how (de-)synchronicity/disease severity is quantified .
One goal is to reproduce parts of the numerical results presented in @osc2, since they serve as a starting point when trying to representing real patient trajectories inside the #acr("DNM").

=== Technology and Details
The numerical integration was performed using diffrax @kidger2021diffrax which is built on-top of JAX @jax2018.
The backbone JAX, is a Python package for high-performance array computation, similar to NumPy or MATLAB, but designed for automatic differentiation, vectorization and #acr("JIT").
#acr("JIT")-compilation and vectorization allow high-level numerical code to be translated to highly optimized accelerator-specific machine code, for example #acr("GPU").
This way, performance benefits of massively parallelizable hardware can be utilized with minimal extra programming cost.
Diffrax implements several numerical differential equation solvers directly in JAX.

While @osc2 uses a fourth-order Runge-Kutta method and a fixed step-size, this implementation#footnote[The code is available at https://github.com/unartig/sepsis_osc/tree/main/src/sepsis_osc/dnm] uses the Tsitouras 5/4 Runge-Kutta method @Tsitouras2011Runge with adaptive step-sizing controlled by a #acr("PID") controller.
A relative tolerance of $10^(-3)$ and an absolute tolerance $10^(-6)$ were chosen, allowing for more efficient integration while keeping an equivalent accuracy.
All simulations were carried out in 64-bit floating point precision, necessary for accurate system integration.

Because of the element-wise differences used in the coupling terms $phi^ot_i-phi^ot_j in RR^(N times N)$ the computational cost scales quadratically with the number of oscillators $N$.
These differences are then transformed by the relatively expensive trigonometric $sin$ routine, to accelerate integration, these trigonometric evaluations were optimized following @KuramotoComp.
Terms in the form $sin(theta_l-theta_m)$ were expanded as:
$
  sin(theta_l-theta_m)=sin(theta_l)cos(theta_m) - cos(theta_l)sin(theta_m) "    " forall l,m in {1,...,N}
$
By caching the terms $sin(theta_l), sin(theta_m), cos(theta_l), cos(theta_m)$ once per iteration, the number of trigonometric evaluations per iteration is reduced from $2*[N (N-1)]$ to $2*[4N]$, significantly improving performance for mid to large oscillator populations.

Additionally, an alternative implementation based on Lie-algebra formulations was also explored, utilizing their natural representation for rotations in N-D-space.
Although theoretically promising in terms of numerical accuracy and integration stability, this approach did not yield practical advantages in performance.
Further details on this reformulation are provided in @sec:appendix #todo[schreiben].

=== Initialization and Parameterization <sec:init>
The #acr("DNM") is dimensionless and not bound to any physical scale, that means there is no ground medical truth of parameter values and their choice is somewhat arbitrary.
For the present implementation the parameterization is adopted from the original work @osc1 and @osc2 since they have already shown some desired properties of (de-)synchronization.

Initial phases $phi(0)^ot_i$ are randomly and uniformly distributed around the unit circle for both layers, i.e. $phi(0)^ot_i ~ cal(U)[0, 2pi)$.
The intralayer coupling of the parenchymal layer coupling is also chosen randomly and uniformly distributed in the interval between -1 and 1.
Since there is no self-coupling, the diagonal is set to 0.

For the immune layer an initial cytokine activation is models by clustering the initial intralayer coupling matrix.
A smaller cluster of $C*N$ oscillators and a bigger cluster of $(1-C)*N$ cells being connected within the cluster but no connection between the two.
Following @osc2 the cluster size $C in [0, 0.5]$ was chosen as 0.2, but as their findings suggest the size of the clusters does not have impact on the systems dynamics.
Simulations have shown that even without any clustering, meaning $bold(Kappa)^2=bb(0)$ or $bold(Kappa)^2=bb(1)$, the dynamics stay unchanged, making this initialization choice meaning-free, it is stated here just for completeness.
An exemplary initial variable values of a system with $N=200$ and $C=0.2$ is shown in @fig:init.

#figure(
  image("images/init.svg", width: 100%),
  caption: [
    This figure shows the initializations for the variable values of a #acr("DNM") with $N=200$ oscillators per layer.
    The middle two plots show the initializations for the oscillators of the two layers, with $phi^1_i$ for parenchymal and $phi^2_i$ for the immune layer, from a uniform random distribution from 0 to $2pi$.
    On the left hand side is the initialization of the parenchymal intralayer couling matrix $bold(Kappa)^1$ from a uniform distribution in the interval from -1 to 1.
    On the right hand side is the two cluster initialization for the immune intralayer coupling matrix $bold(Kappa)^2$ where each cluster is intra-connected, but no connection between the clusters.
    The cluster size is $C=0.2$, creating the smaller cluster to be 20% of the total number of oscillators.
  ],
) <fig:init>

Other parameter choices of the original heavily simplify the model.
First of all are the natural frequencies treated as equal and are set to 0 giving $omega^1 = omega^2 = omega = 0$, for any other choice of $omega$ just changes the frame of reference (co-rotating frame), the dynamics stay unchanged @osc2.
The phase lag parameters for the inter layer coupling are both set to $alpha^(1 2) = alpha^(2 1) = 0$, yielding instantaneous interactions, the intralayer phase lags are set to $alpha^11 = alpha^22 = -0.28pi$, which was the most most prominently used configuration in @osc2.
The constant intralayer coupling in the parenchymal is chosen as global coupling $a_(i j) = 1 " if " i!=j " else " 0$.

The adaptation rates are chosen as $epsilon^1=0.03$ and $epsilon^2=0.3$, creating the two dynamical time scales for slow parenchymal and faster immune cells.
The number of oscillators per layer is chosen as $N=200$ throughout all simulations.
To account for the randomly initialized variables, each parameter configuration is integrated for an ensemble of $M=50$ initializations.
An exhaustive summary of all variable initializations and parameter choices can be found in @tab:init.

#TODO[beta sigma]

#figure(
  table(
    columns: (auto, 13em, auto, 13em),
    align: center,
    table.header([*Symbol*], [*Value*], [*Symbol*], [*Value*]),
    table.cell(colspan: 4)[*Variables*],
    [$phi^1_i$], [$~cal(U)(0, 2pi)$],
    [$kappa^1_(i != j)$],
    [$~cal(U)(-1, 1)$],
    [$phi^2_i$], [$~cal(U)(0, 2pi)$],
    [$kappa^2_(i != j)$],
    [clusters of size $C$ and $1-C$],

    table.cell(colspan: 4)[*Parameters*],
    [$M$], [50], [$N$], [200],
    [$C$], [$20%$], [], [],
    [$beta$], [$[X, Y]pi$ #todo[what are the lims]], [$sigma$], [$[0.0, 1.5]$],
    [$alpha^11, alpha^22$], [$-0.28pi$], [$alpha^12, alpha^21$], [0],
    [$omega_1, omega_2$], [0.0], [$A^1$], [$bb(1) - I$],
    [$epsilon^1$], [0.03], [$epsilon^2$], [0.3],
  ),
  caption: [todo],
)<tab:init> #todo[Non breakable tables?]


=== Synchronicity Metrics
There are two relevant states or system configurations that should be identifiable and quantifiable to allow qualified state analyses: phase and frequency synchronization.
For each a distinct measure is required, for the phase synchronization of a layer the Kuramoto Order Parameter @Placeholder is most commonly used:
$
  R^ot_2 = 1/N abs(sum^N_j e^(i dot phi^ot_j (t))) "   with " 0<=R^ot_2<=1
$
where $R^mu_2=0$ corresponds to total desynchronization, the splay-state and $R^mu_2=1$ corresponds to fully synchronized state, for convinience from now on the subscript $""_2$ is ommited, denoting the Kuramoto Order Parameter simply as $R^ot$.

To measure frequency synchronization and detect frequency clustering we first have to introduce the notion of a layers _mean phase velocity_, which can be calculated as follows:

$
  overline(omega)^ot = 1/N sum^N_j dot(phi)^ot_j
$ <eq:mean>
The original definition in @osc1 and @osc2 uses an approximated version using the oscillators mean velocity.
This is most likely because they were not able to recover the actual derivatives $dot(phi)^ot_i$ from their integration scheme and had to work with the phases $phi^ot_i$ instead:
$
  mean(dot(phi)^ot_j) &= (phi^ot_j (t + T) - phi^ot_j (t))/T \
  overline(omega)^ot &= 1/N sum^N_j mean(dot(phi)^ot_j)
$ <eq:mean>
for some averaging time window $T$.
But since their choice of $T$ is not documented while having substantial influence on the calculation the direct calculation was used.

One can now calculate the standard deviation of the mean phase velocities:
$
  sigma_chi (overline(omega)^ot) = sqrt(1/N sum^N_j (mean(dot(phi)^ot_j) - overline(omega)^ot)^2)
$ <eq:stdsingle>
Where $sigma_chi = 0$ indicates full frequency synchronization and growing values indicate desynchronization and/or clustering.
But non-zero values only reveal that there is some desynchronization of the frequency, but it remains unknown if it is clustered, multiclustered or fully desynchronized.

Since there are multiple ensemble members $m in M$ for the same parameterization, and it expected that different initialization, even though equally parameterized, can exhibit dissimilar behaviors, one can also calculate the
 _ensemble averaged standard deviation of the mean phase velocity_:

$
  s^ot = 1/M sum^M_m sigma_chi (overline(omega)_m^ot)
$ <eq:std>
In @osc2 it was shown numerically that the quantity $s^ot$ is proportional to the fraction of ensemble members that exhibit frequency clusters containing at least one oscillator.
This makes it a viable measure for pathology, as increasing values of $s^1$ or increasing incoherence then indicate more dysregulated host responses and consequently higher risks of multiple organ failure.

=== Simulation Results <sec:dnmres>
The orignal findings of @osc2 identify $beta$, the combined age parameter, and $sigma$, the interlayer coupling strength which models the cytokine activity, as naturally important parameters in order to understand underlying mechanisms of sepsis progression.
The following subsection presents several simulation results, starting with "snapshots" of different stable system states for unique initialisations, followed by the transient and temporal behavior of the metrics $s^ot$ and $R^ot$ and wrapping with the introduction of the $beta, sigma$ phase space of these metrics.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center,
    table.header([], [*A*], [*B*], [*C*], [*D*]),
    [$beta$], [$0.5 pi$], [$0.58 pi$], [$0.7 pi$], [$0.5 pi$],
    [$sigma$], [$1.0$], [$1.0$], [$1.0$], [$0.2$],
  ),
    caption: [todo]
)<tab:siminit> 
In @fig:snap snapshots of the system variables are shown for different parameterizations, rows A, B, C and D, while rows C and C' share the same parameterization but are different samples from the same initialization distributions, which are introduced in @sec:init.
The left most columns depicts the coupling matrices for the organ layer $bold(Kappa)^1$ followed by two columns showing the phase velocities for each oscillator $dot(phi)_i^ot$ and two columns showing the oscillator phases each layer $phi_i^ot$, ending with the righ-most column showing the coupling matrix for the immune layer $bold(Kappa)^2$.
All snapshots are taken at time $t=2000$, the end of the integration time, and show the stationary values at that time point.
Each layer is first from lowest to highest frequency and secondary by lowest to highest phase for better clarity.

#figure(
  image("images/snapshots.svg", width: 100%),
  caption: [
  #TODO[colorbar]
  ],
) <fig:snap>
Row A in @fig:snap is fully synchronized/coherent since it not only has the frequencies synchronized but also the phases ($R^ot=1$) and can therefore interpreted as healthy.
The coherence can also be seen in the fully homogenuous coupling matrices where both show the same coupling strength for every oscillator pair.
Row B and C in contrast show signs of a pathological state, here both the frequencies three and phases have four distinct clusters respectively, which is also visible in the coupling matrices, where the coupling has different strength based on the coupling #todo[which is stronger?].
The row C', even though having the same parameterization as C, can be regarded vulnerable, since the phases uniformly distribute in the $[0, 2pi)$ interval ($R^ot = 0$), while the frequencies are synchronized.
Coupling matrices for C' show travelling waves, which are characteristic for splay states.
Observing different results for different initializations justifies the introduction of ensembles.
Lastly row D shows a resilient state, where the phases are clustered while the frequencies are synchronized.

For the next figure the ensembles were introduced, every parameterization A, B, C and D was simulated for $M=50$ different initializations over an interval of $t=2000$.
The plots show the temporal evolution of metrics for quantifying phase and frequency coherence, with the two left-most columns of @fig:ensemble show the temporal behavior of the Kuramoto Order Parameter for each individual ensemble member $m in 1...M$ and the two right-most show the ensemble averaged standard deviation of the mean phase velocities $s^ot$.
Where lower values for $R^ot$ indicate decoherence in phases, with its minimum $R^ot = 0$ coincides with a splay state, and for $s^ot$ higher values indicate a larger amount of frequency decoherence and clustering.
#figure(
  image("images/ensembles.svg", width: 100%),
  caption: [
  #TODO[Not mean over s?]
  ],
) <fig:ensemble>
Vvery ensemble in @fig:ensemble shows decoherence for early time-points, which is expected for randomly initialized variables, changes relatively fast through a transient phase into systematic stable behavior.
Aligning with the observations in @fig:snap, configuration A has, besides small jittering, mostly synchronized frequencies $s^ot tilde(=) 0$.
Also the phases are for most of the ensemble members synchronized with $R^12 tilde(=) 1$, only two initializations show decoherence and are oscillating between weak clustering and almost full incoherence.
For configuration B the amount of incoherence inside the ensemble is clearly visible, with $s^ot$ being positive and some more ensemble members exhibiting clustering, indicated by a Kuramoto Order Parameter slightly less than $1$.
In configuration C the incoherence is even more promoinent, even larger $s^ot$ and some ensemble members evolving into a splay state, i.e. $R^ot=0$.
For configuration D the overall phase incoherence is again a bit less compared to C, and lower for the organ compared to the immune layer.
The phases are coherent for the organ layer but seem almost chaotic for the immune layer, some are synchronized, while others are clustered, in a chimera or almost splay-like state.
Over the whole simulation period, the coherency in the immune layer seems not to fully stabilize, rather oscillate around an attractor.
= Latent Dynamics Model <sec:ldm>
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
#figure(fsq_fig)
#todo[Fix the edges]

=== Encoder
=== Decoder
=== Introducing time
=== Combining the building blocks
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
The metrics are detailed in.

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

= Metrics (How to validate performance?)

= Experimental Results <sec:experiment>
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
