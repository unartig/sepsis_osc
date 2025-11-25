#import "@preview/acrostiche:0.5.2": *
#import "thesis_template.typ": thesis
#import "@preview/drafting:0.2.2": inline-note, margin-note, note-outline, set-margin-note-defaults
#import "figures/helper.typ": cmalpha, cmbeta, cmred, cmsigma

#import "figures/tree.typ": tree_fig
#import "figures/fsq.typ": fsq_fig
#import "figures/kuramoto.typ": kuramoto_fig
#import "figures/high_level.typ": high_fig

#show: thesis.with(
  title: "Comprehensive Guidelines and Templates for Thesis Writing",
  summary: [
  ],
  // abstract_de: [
  // ],
  acronyms: (
    "ABX": "Antibiotics",
    "BCE": "Binary Cross Entropy",
    "DAMP": "Damage-Associated Molecular Patterns",
    "DL": "Deep Learning",
    "DNM": "Dynamic Network Model",
    "EHR": "Electronic Health Record",
    "FSQ": "Finite Scalar Quantization",
    // "GLM": "Generalized Linear Model",
    "GPU": "Graphics Processing Unit",
    "ICU": "Intensive Care Unit",
    "JIT": "Just In Time Compilation",
    "LDM": "Latent Dynamics Model",
    "ML": "Machine Learning",
    "MSE": "Mean Squared Error",
    "ODE": "Ordinary Differential Equation",
    "PAMP": "Pathogen-Associated Molecular Patterns",
    "PID": "Proportional-Integral-Derivative",
    "PINN": "Physics Informed Neural Networks",
    "PRR": "Pattern Recognition Receptors",
    "qSOFA": "Quick Sequential Organ Failure Assessment",
    "RAG": "Retrieval Augmented Generation",
    "SIRS": "Systemic Inflammatory Response Syndrome",
    "SI": "Suspected Infection",
    "SOFA": "Sequential Organ Failure Assessment",
    "TUHH": "Hamburg University of Technology",
    "YAIB": "Yet Another ICU Benchmark",
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
// #let multicite(x) = "[" +

#TODO[
  #list(
    [Sections to Chapters],
    [Styling],
    [Appendix to real Appendix],
    [Fix ACR separation],
    [Fix newline/linebreak after Headings],
    [Sources misc + link],
  )
]
#TODO[actual functional model
  what is learned
  connecting parts
]

= Introduction

= Medical Background (Sepsis) <sec:sepsis>

As the most extreme course of an infectious disease, sepsis poses a very serious health threat, with a high mortality rate and frequent long-term consequences for survivors.
In 2017, an estimated 48.9 million people worldwide suffered from sepsis and the same year, 11.0 million deaths were associated with sepsis @rudd2020global, which makes up 19.7% of yearly deaths, making it the most common cause of in-hospital deaths.
Untreated, the disease is always fatal and even with successful treatment, around 40\% of those affected suffer long-term consequences, such as cognitive, physical or physiological problems, the so called _post-sepsis syndrome_ @vanderSlikke2020post.
Overall, treated and untreated septic diseases in particular represent an enormous burden on the global healthcare system.
The observed risk of mortality significantly differs between lower to middle income countries with $>50%$ and high income countries with $<25%$.

Even though almost half of all sepsis-related deaths occur as a secondary complication of an underlying injury or a non-communicable, also known as chronic disease @fleischmann2022sepsis, the underlying triggers but also the individual progressions of sepsis remain highly diverse and heterogeneous.
Moreover, a septic condition can not be reduced to a single specific physiological phenomenon, instead it combines multiple complex and interdependent processes across different biological scales.

This complexity has historically made it difficult to define sepsis in a medical precise way compared to other conditions.
Multiple definitions have been proposed over time, and the terminology around sepsis and septic-shocks has often been blurry.
The most commonly used and accepted sepsis definition characterizes sepsis as a "life-threatening organ dysfunction caused by a dysregulated host response to infection" @Sepsis3.
The following @sec:sep3def provides a detailed overview to this definition, which is referred to as Sepsis-3.
Furthermore, @sec:sepbio introduces the both the pathology and underlying biology of sepsis in greater detail.

A recent study @seymour2017time highlights the importance of early recognition and subsequent treatment of infections in patients, reducing the mortality risk caused from sepsis.
Each hour of earlier detection can significantly increase the chance of survival @seymour2017time, it urges to develop accurate and robust detection and prediction methods, i.e. reducing the time to receive the appropriate medical attention.
In @sec:sepwhy the necessity for reliable and clinically practical sepsis prediction systems is discussed.

== The Sepsis-3 Definition <sec:sep3def>
Earlier definitions (Sepsis-1, Sepsis-2 @Placeholder) primarily emphasized #acr("SIRS") @Placeholder criteria, focusing on the inflammatory origins of sepsis.
These definitions were later criticized for low specificity and under-representation of the multi organ failure due to sepsis.
Out of the need for an update of these outdated definitions and partly misleading sepsis models a task force led by the "Society of Critical Care Medicine and the European Society of Intensive Care Medicine", was formed in 2016.
Their resolution, named "Third International Consensus Definitions for Sepsis and Septic Shock" @Sepsis3, provides until today the most widely used sepsis definition and guidance on sepsis identification.

In general, sepsis does not classify as a specific illness, rather a multifaceted condition of "physiologic, pathologic, and biochemical abnormalities" @Sepsis3, and septic patients are largely heterogeneous.
Also the trigger is explicitly non-specific, since different triggers can cause the same septic condition.
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

*Confirmed or Suspected Infection* has no strict medical definition and classification what counts as #acr("SI") remains a little vague, ultimately it is left for the medical personnel to classify infections or the suspicion of infections. For retrospective data-driven classification it is suggested to characterize any patient prescribed with #acr("ABX") followed by the cultivation of body fluids, or the other way around, with a #acr("SI") @Sepsis3.
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
An increase in #acr("SOFA") score $>=2$ corresponds to an acute worsening of organ functionalities and a drastic worsening in the patients condition, the indicator for a dysregulated response.

=== Sepsis Classification
The Sepsis-3 definition not only provides the clinical criteria of septic conditions, but also introduces the necessary time windows for sepsis classification.
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
    align(left, [the lowest value of the 24h previous to the increase.]),
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
    align(left, [Systolic blood pressure $<=$ 100 mmHg]),
  ))
)
Patients fulfilling at least two of these criteria have an increased risk of organ failure.
While the #acr("qSOFA") has a significantly reduced complexity and is faster to assess it is not as accurate as the #acr("SOFA") score, meaning it has less predictive validity for in-house mortality @SOFAscore.

There is also the notion of a septic shock, also defined in @Sepsis3, which an in-hospital mortality above 40%.
Patients with a septic shock are can be identified by:
#(
  align(center, list(
    align(left, [Sepsis]),
    align(left, [Persisting hypotension requiring\
      vasopressors to maintain MAP $>=$ 65mmHg]),
    align(left, [Serum lactate level > 2 mmol/L, despite volume resusciation.]),
  ))
)


== Biology of Sepsis <sec:sepbio>
This part tries to give an introduction into the biological phenomena that underlie sepsis.
Starting with an explanation on how human tissue is reacting to local infections or injuries on a cellular level in @sec:cell and how this can escalate to _cytokine storms_ in @sec:storm and ending with systemic organ failure in @sec:fail.

Certain details and specifities are left out when not essential for the understanding of this project.
More detailed explanations can be found in the primary resources provided throughout this section.

=== Cellular Origins <sec:cell>
Human organ tissue can be differentiated into two broad cell-families called _parenchymal_ and _stroma_ which are separated by a thin, specialized boundary layer known as the _basal lamina_.

The parenchymal cells perform the primary physiological functions of an organ, with every organ hosting distinct parenchymal cells @VanHara2020Guide#todo[correct source].

Everything not providing organ-specific functionalities forms the stroma, that includes the structural or connective tissue, blood vessels and nerves.
The stroma not only contributes to the tissues structure, but it also actively participates in biochemical signaling and immune regulation.
This way it helps to maintain a healthy and balanced tissue, the _homeostasis_, and enables coordinated responses to injury or infection @Honan2021stroma.

A pathogen is summarizes all types of organisms that can be harmful to the body, this includes germs, fungi, algae, or parasites.
When a pathogen enters the body through the skin, a mucous membrane or an open wound, the first line of non-specific defense, the innate immune system @Fischer2022Innit, gets activated.

This rapid response does not require the body to have seen the specific pathogen before.
Instead, the innate immune system can be triggered by sensing commonly shared features of pathogens, in case of germs known as #acr("PAMP"), for injury called #acr("DAMP") @Jarczak2021sepsis.
The #acr("PAMP")'s and #acr("DAMP")'s can be detected by #acr("PRR"), which are found in resident immune cells, as well as stroma cells.
Once a pathogen is detected a chain reaction inside the cell leads to the creation and release of signaling proteins called _cytokines_ @Zhang2007cyto.

Cytokines are a diverse group of small signaling proteins which play a special role in the communication between other cells, both neighboring and across larger distances through the bloodstream.
They are acting as molecular messengers that coordinate the recruitment of circulating immune cells and will guide them to the location of infection or injury @Zhang2007cyto.

Besides their role in immune activation where cytokines regulate the production of anti- and pro-inflammatory immune cells which help with the elimination of pathogens and trigger the healing process right after.
They are also participating in the growing process of blood cells.

One specialty of these relatively simple proteins is that they can be produced by almost every other cell, with different cells being able to produce the same cytokine.
Further, cytokines are redundant, meaning targeted cells can show identical responses to different cytokines @House2007cyto, these features seems to fulfill some kind of safety mechanism to guarantee vital communication flow.
After release cytokines have relatively a short half-life (only a few minutes) but through cascading-effects the cytokines can have substantial impact on their micro-environment.

=== Cytokine Storms <sec:storm>
The hosts dysregulated response to an infection connected to the septic condition is primarily driven by the excessive and uncontrolled release cytokines and other mediators.
Under normal circumstances, the release of inflammatory cytokines tightly regulated in time and magnitude.
After the pathogen detection the release is quickly initiated, peaks as immune cells are recruited and automatically fades out once the initial pathogen is controlled and the host returns to a healthy and balanced state, the homeostasis.

In certain scenarios a disturbance to the regulatory mechanisms triggers positive inflammatory feedback loop, followed by a massive release of pro-inflammatory cytokines.
These cells further activate additional immune and non-immune cells, which in turn amplify the cytokine production, creating a self-reinforcing cycle of immune activation @Jarczak2022storm.
This ultimately leads to a continuous and uncontrolled release of cytokines that fails to shut down.
With this overreaction, called _cytokine storm_, the immune response and release of inflammatory mediators can damage the body more than the infection itself.

Although the quantity of cytokines roughly correlates with disease severity, concentrations of cytokines vary between patients, time and even different body-parts, making a distinction between an appropriate reaction and a harmful overreaction almost impossible @Jarczak2022storm.
Out of all cytokines, only a small subset or secondary markers can be measured through blood samples to detect increased cytokine activity.
This limited accessibility cytokines difficult to study in general, they prove to be little useful as direct indicators of pathogenesis or diagnostic purposes.

Since the 90s there has been a lot of research focused on cytokines and their role in the innate immune system and overall activation behavior.
Multiple therapeutic interventions have been tested in clinical trials, yet none have achieved a significant improvement in survival outcomes @Jarczak2021sepsis.
This emphasizes the complexity of sepsis as a systemic syndrome rather than a single-cause disease, and suggests that cytokine storms are an emergent property rather than the result of any one molecular trigger.
To this day, the fundamental principles that govern the transition from a regulated immune response to a self-destructive cytokine storm remain not fully understood.

=== Systemic Consequences and Organ Failure <sec:fail>
While more and more cytokines are released, they flood not only infected areas, but also surrounding parts of the tissue and circulation, causing localized inflammatory response to become systemic.
The widespread cytokine reaction starts to disrupt the normal metabolism of parenchymal cells in organs due to a deficiency in oxygen and nutrients.

To compensate, cells switch from their usual oxygen-based metabolism to an _anaerobic glycolysis_ @Prieto2016Anaerobic, generating energy less efficiently from glucose.
As a result, metabolic by-products such as lactate accumulate making the surrounding environment more acidic, which further harms the cells and leads to more cellular dysfunction.

At the same time, the mitochondria, the "power house" of the cells, start to fail.
The walls of blood vessels become leaky, allowing fluids to move into surrounding tissue.
This causes swelling and lowers the blood pressure, which in turn reduces the oxygen supply even further @Jarczak2021sepsis.

Step by step, the death of cells spreads throughout the body and affects organ functionality.
When multiple organs fail simultaneously, the condition becomes irreversible @Sepsis3.
At this stage, multi-organ-failure is the final and most lethal form of sepsis, with each additional affected organ the mortality increases drastically.


== The need for sepsis prediction <sec:sepwhy>

To this day sepsis, and the more extreme septic shock, remains as an extreme burden to the worldwide healthcare system.
It is associated with high rates of incidence, high mortality and significant morbidity.
Despite overall advancements in medical care and slowly decreasing prevalence numbers, sepsis continues to be the leading cause of in-hospital death @Via2024Burden.

In germany it was estimated in 2022 that at least 17.9% of intensive care patients develop sepsis, and 41.7% of all hospital treated sepsis patients die during their stay @fleischmann2022sepsis.
The economic burden is equally severe, with the annual cost of sepsis treatment in germany estimated to be €7.7 billion based on extrapolated data from 2013.

Globally , the situation is even more concerning, as sepsis remains to be under-diagnosed significantly due to its non-specific symptoms.
Environmental and socioeconomic factors such as insufficient sanitation, limited access to clean water and healthcare increases the incidence particularly in low- to middle income countries @rudd2020global@Via2024Burden.

A meta-analysis of seven sepsis alert systems found no evidence for improvement in patient outcomes, suggesting insufficient predictive power of analyzed alert systems or inadequate system integration @Alshaeba2025Effect.
Nevertheless, positive treatment outcomes depend heavily on timely recognition and intervention @Via2024Burden.
Each hour of delayed treatment increases mortality risk, underscoring the critical importance of early detection @seymour2017time while structured screening and early warning systems have demonstrated reductions in time-to-antibiotics and improvements in outcomes @Westphal2009Early.
These findings confirm that in principle earlier identification of sepsis improves clinical results, even if existing tools are not yet capable enough, and emphasizes the need for more research in that direction.

A recent study suggests a paradigm shift in sepsis detection—from a symptom-based to a systems-based approach @Dobson2024Revolution.
Instead of waiting for clinical signs, early recognition should integrate multiple physiological and biochemical signals to capture the transition from infection to organ dysfunction.
This aligns with the findings of a survey among clinicians regarding AI-Assistance in healthcare @EiniPorat2022.
One participant emphasizes that specific vitals signs might be of less importance, rather the change/trend of a patients trajectory should be the prediction target.
Another piece of finding of the same study was the preference of trajectories over plain binary event predictions.

However, implementation any data-driven prediction approaches into clinical practice presents challenges.
Implementation studies consistently identify barriers such as alert fatigue, workflow disruption, and inconsistent screening uptake.
To be effective, predictive systems must integrate seamlessly into and existing workflows provide interpretable output and aid the clinical expertise @EiniPorat2022.

Taken together, these insights highlight both the need and the opportunity for improved sepsis prediction.
The global burden and clinical urgency justify the development of more reliable prediction systems.
At the same time, the limitations of current alert systems and implementation barriers underline the necessity for models that can integrate dynamic patient data and capture clinical trajectories.

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
The goal is to investigate how real patients would translate to the model parameters, and how the temporal physiological evolution can be incorporated and if there is a benefit doing so.
// However, this model has not yet been validated against real-world observations, which will be addressed in this work #todo[eher project???].


To summarize, the specific research questions include:
#(
  list(
    [*Usability of the #acr("DNM")*: How and to what extent can the #acr("ML")-determined trajectories of the #acr("DNM") be used for detection and prediction, especially of critical infection states and mortality.],
    [*Comparison with data-based approaches*: How can the model-based predictions be compared with those of purely data-based approaches in terms of predictive power and interpretability.],
  )
)
#TODO[End this]


= Model Background (Dynamic Network Model) <sec:dnm>

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
These adaptive connections allow for information to propagate through the whole network, giving rise to emerging phenomena on global scales for otherwise identical network topologies.

Complex networks are well studied in physics and biology and have been applied to various physiological domains.
Early works, such as @Guyton1972Circulation that have studied the cardiovascular system, while more recent studies have focused on the cardio-respiratory coupling @Bartsch2012Phase and large-scale brain network dynamics @Lehnertz2021Time.
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
Several extensions are directly relevant to the #acr("DNM") and their definitions and effects on synchronization will be shortly introduced, with additional terms being indicated by the red color:

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

The introduction changes the system behavior once more, for example single layers of a multiplexed system can result in the multi-clustered regime for parameters they would not in the monoplexed case.
In multiplexed systems it is also possible connected layers end up in different stable state, for example, one in a clustered the other in a splay state.

== Description <sec:dnmdesc>
#TODO[Figure bio vs oscillators]
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
    // [$C$], [Initial network perturbation], [-],

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

It is important to note, that the #acr("ODE") dynamics or system variable trajectories *do not* translate to the evolution of a patients pathological state.
Instead the amount of desynchronization of the parenchymal layer when reaching a steady system state can be interpreted as the current state of a patients organ functionality.
The "result" or solution of the coupled oscillator system does not provide any temporal insights, but only describe the current condition.

// #figure(tree_fig)

== Implementation <sec:dnmimp>
For initial value problems of coupled #acr("ODE")-systems, such as the #acr("DNM"), analytical solutions rarely exist @osc2, and if they exists it is mostly for trivial or other special configurations or by applying aggressive simplifications.
To solve these kind of systems one traditionally relies on the numerical integration, approximating the analytical solution.

This subsection describes the implementation for the numerical integration of the #acr("DNM") defined in @eq:ode-sys, the choice of initial parameter values and how (de-)synchronicity/disease severity is quantified.
One goal of this implementation is to partly reproduce the numerical results presented in @osc2, since they will be serving as a basis for following chapters.

=== Technology and Details
The backbone for the present numerical integration is JAX @jax2018, a Python package for high-performance array computation, similar to NumPy or MATLAB but designed for automatic differentiation, vectorization and #acr("JIT").
#acr("JIT")-compilation and vectorization allow high-level numerical code to be translated to highly optimized accelerator-specific machine code, for example #acr("GPU").
This way, performance benefits of massively parallel hardware can be utilized with minimal extra programming cost.
For the actual integration a differential equation solver from diffrax @kidger2021diffrax was used, which provides multiple solving schemes fully built on top of JAX.

While @osc2 uses a fourth-order Runge-Kutta method and a fixed step-size, this implementation#footnote[The code is available at https://github.com/unartig/sepsis_osc/tree/main/src/sepsis_osc/dnm] uses the Tsitouras 5/4 Runge-Kutta method @Tsitouras2011Runge with adaptive step-sizing controlled by a #acr("PID") controller.
A relative tolerance of $10^(-3)$ and an absolute tolerance $10^(-6)$ were chosen, allowing for more efficient integration while keeping an equivalent accuracy.
All simulations were carried out in 64-bit floating point precision, necessary for accurate and stable system integration.

Because of the element-wise differences used in the coupling terms $phi^ot_i-phi^ot_j in RR^(N times N)$ the computational cost scales quadratically with the number of oscillators $N$.
These differences are then transformed by the computationally expensive trigonometric $sin$ routine.
To accelerate integration, these trigonometric evaluations were optimized following @KuramotoComp.
Terms in the form $sin(theta_l-theta_m)$ were expanded as:
$
  sin(theta_l-theta_m)=sin(theta_l)cos(theta_m) - cos(theta_l)sin(theta_m) "    " forall l,m in {1,...,N}
$
By caching the terms $sin(theta_l)$, $sin(theta_m)$, $cos(theta_l)$, $cos(theta_m)$ once per iteration, the number of trigonometric evaluations per iteration is reduced from $2dot[N (N-1)]$ to $2dot[4N]$, significantly improving performance for mid to large oscillator populations.

Additionally, an alternative implementation based on Lie-algebra formulations was also explored, utilizing their natural representation for rotations in N-D-space.
Although theoretically promising in terms of numerical accuracy and integration stability, this approach did not yield practical advantages in performance.
Further details on this reformulation are provided in @sec:appendix #todo[schreiben].

=== Parameterization and Initialization <sec:init>
The #acr("DNM") is dimensionless and not bound to any physical scale, that means there is no medical ground truth of parameter values and their choice is somewhat arbitrary.
For the present implementation the parameterization is adopted from the original works @osc1 and @osc2 since they have already shown desired properties of (de-)synchronization and valid medical interpretations for these parameter choices.

The majority of their parameter choices heavily simplify the model.
First of all, the different natural frequencies are treated as equal and are set to 0 giving $omega^1 = omega^2 = omega = 0$, any other choice of $omega$ just changes the frame of reference (co-rotating frame), the dynamics stay unchanged @osc2.
The phase lag parameters for the inter layer coupling are both set to $alpha^(1 2) = alpha^(2 1) = 0$, yielding instantaneous interactions, the intralayer phase lags are set to $alpha^11 = alpha^22 = -0.28pi$, which was the prominently used configuration in @osc2 yielding the desired dynamical properties.
The constant intralayer coupling in the parenchymal is chosen as global coupling $a_(i j) = 1 " if " i!=j " else " 0$.

The adaptation rates are chosen as $epsilon^1=0.03$ and $epsilon^2=0.3$, creating the two dynamical timescales for slow parenchymal and faster immune cells.
The number of oscillators per layer is chosen as $N=200$ throughout all simulations.
To account for the randomly initialized variables, each parameter configuration is integrated for an ensemble of $M=50$ initializations.
This randomization of initial values is used to account for epistemic uncertainties, i.e. systemic errors introduced by modeling simplifications.

In @osc2 the influence of parameter values for $beta$ and $sigma$ was investigated and not constant throughout different simulations, with $beta in [0.4pi, 0.7pi]$ and $sigma in [0, 1.5]$, in this work the interval for $beta$ was increased to $[0.0, 1.0pi]$.
An exhaustive summary of all variable initializations and parameter choices can be found in @tab:init.

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
    [$beta$], [$[0.0, 1.0]pi$], [$sigma$], [$[0.0, 1.5]$],
    [$alpha^11, alpha^22$], [$-0.28pi$], [$alpha^12, alpha^21$], [0.0],
    [$omega_1, omega_2$], [0.0], [$A^1$], [$bb(1) - I$],
    [$epsilon^1$], [0.03], [$epsilon^2$], [0.3],
  ),
  caption: [Parameterization and initialization of the #acr("DNM") used for the numerical integration.],
)<tab:init> #todo[Non breakable tables?]

Initial values for the system variables, i.e. the phases and coupling strengths, were not parametrized explicitly, rather sampled from probability distributions.
The initial phases $phi(0)^ot_i$ are randomly and uniformly distributed around the unit circle for both layers, i.e. $phi(0)^ot_i ~ cal(U)[0, 2pi)$.
The intralayer coupling of the parenchymal layer coupling is also chosen randomly and uniformly distributed in the interval $[-1.0, 1.0]$.
Since there is no self-coupling, the diagonal is set to 0.

For the immune layer an initial cytokine activation is models by clustering the initial intralayer coupling matrix.
A smaller cluster of $C dot N$ oscillators and a bigger cluster of $(1-C) dot N$ cells.
Within the clusters oscillators are connected but not between the clusters.
Following @osc2 the cluster size $C in [0, 0.5]$ was chosen as 0.2, but as their findings suggest the size of the clusters does not have impact on the systems dynamics.
Simulations have shown that even without any clustering, meaning $bold(Kappa)^2=bb(0)$ or $bold(Kappa)^2=bb(1)$, the dynamics stay unchanged, making this initialization choice meaning-free, it is stated here just for completeness.
An example for initial variable values of a system with $N=200$ and $C=0.2$ is shown in @fig:init.

#figure(
  image("images/init.svg", width: 100%),
  caption: [
    Initializations for the variable values of a #acr("DNM") with $N=200$ oscillators per layer.
    The middle two plots show the phases of the oscillators, with $phi^1_i$ for parenchymal and $phi^2_i$ for the immune layer, sampled from a uniform random distribution from 0 to $2pi$.
    On the left-hand side is the initialization of the parenchymal intralayer coupling matrix $bold(Kappa)^1$ from a uniform distribution in the interval from -1 to 1.
    On the right-hand side is the two cluster initialization for the coupling matrix $bold(Kappa)^2$ of the immune layer, with a cluster size of $C=0.2$, where each cluster is intra-connected, but without connections between the clusters.
    #todo[index for immune]
  ],
) <fig:init>

To average out the influence of specific random initial values, simulations are performed for ensembles, combining $m in 1,2...M$ ensemble members.
Throughout this work an ensemble size of $M=50$ was used.

=== Synchronicity Metrics
As introduced in @sec:kuramoto, for the complex Kuramoto networks the synchronization behavior is usually the point of interest, in the following two metrics are introduced, relevant to connect the #acr("DNM")-dynamics to sepsis.
There are two relevant states or system configurations that should be identifiable and quantifiable to allow qualified state analyzes: phase and frequency synchronization, for each a distinct measure is required.

*Phase synchronization* of a layer is commonly measured by the _Kuramoto Order Parameter_ @Placeholder:

$
  R^ot_2 = 1/N abs(sum^N_j e^(i dot phi^ot_j (t))) "   with " 0<=R^ot_2<=1
$
where $R^mu_2=0$ corresponds to total desynchronization, the splay-state and $R^mu_2=1$ corresponds to fully synchronized state, for convenience from now on the subscript $""_2$ is omitted, denoting the Kuramoto Order Parameter simply as $R^ot$.

*Frequency synchronization* measurements are more involved, as a starting point first the notion of a layers _mean phase velocity_ has to be introduced, which can be calculated as follows:

$
  overline(omega)^ot = 1/N sum^N_j dot(phi)^ot_j
$ <eq:mean>
The original definition in @osc1 and @osc2 uses an approximated version using the oscillators mean velocity.
This is likely because they were not able to recover the actual derivatives $dot(phi)^ot_i$ from their integration scheme and had to work with the phases $phi^ot_i$ instead:
$
  mean(dot(phi)^ot_j) & = (phi^ot_j (t + T) - phi^ot_j (t))/T \
   overline(omega)^ot & = 1/N sum^N_j mean(dot(phi)^ot_j)
$ <eq:mean>
for some averaging time window $T$.
But since their choice of $T$ is not documented while having substantial influence on the calculation the direct calculation was preferred.

One can now calculate the standard deviation of the mean phase velocities:
$
  sigma_chi (overline(omega)^ot) = sqrt(1/N sum^N_j (mean(dot(phi)^ot_j) - overline(omega)^ot)^2)
$ <eq:stdsingle>
Where $sigma_chi = 0$ indicates full frequency synchronization and growing values indicate desynchronization and/or clustering.
But non-zero values only reveal that there is some desynchronization of the frequency, but it remains unknown if it is clustered, multi-clustered or fully desynchronized.

Since there are multiple ensemble members $m$ for the same parameterization, and it expected that different initialization, even though equally parameterized, can exhibit dissimilar behaviors, one can also calculate the
_ensemble averaged standard deviation of the mean phase velocity_:

$
  s^ot = 1/M sum^M_m sigma_chi (overline(omega)_m^ot)
$ <eq:std>
In @osc2 it was shown numerically that the quantity $s^ot$ is proportional to the fraction of ensemble members that exhibit frequency clusters containing at least one oscillator.
This makes $s^1$ a viable measure for pathology, as increasing values of $s^1$ or increasing system incoherence then indicate more dysregulated host responses and consequently higher risks of multiple organ failure.

=== Simulation Results <sec:dnmres>
The original findings of @osc2 identify $beta$, the combined age parameter, and $sigma$, the interlayer coupling strength which models the cytokine activity, as naturally important parameters in order to understand underlying mechanisms of sepsis progression.
In the following subsection multiple simulation results are presented, starting with time-snapshots for different parameterization and initializations.
Afterward, the transient and temporal behavior of the metrics $s^ot$ and $R^ot$ is for the same parameterization, as well as the introduction of the $beta, sigma$ phase space of these metrics.

In @fig:snap snapshots of the system variables are shown for different parameterization, differing only in the choice $beta$ and $sigma$, configurations A, B, C and D are listed in @tab:siminit, other parameters are shared between the configurations and are stated in @tab:init.
Each configuration is expected to represent the current physiological state a single patient.

All following results are for a system with $N=200$ oscillators, and snapshots taken at time $t=2000$, the end of the integration time, and show the stationary values at that time point.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center,
    table.header([], [*A*], [*B*], [*C*], [*D*]),
    [$beta$], [$0.5 pi$], [$0.58 pi$], [$0.7 pi$], [$0.5 pi$],
    [$sigma$], [$1.0$], [$1.0$], [$1.0$], [$0.2$],
  ),
  caption: [Specific $beta$-$sigma$ combinations to illustrate simulation results.],
)<tab:siminit>

In @fig:snap the left-most columns depicts the coupling matrices for the organ layer $bold(Kappa)^1$ followed by two columns showing the phase velocities for each oscillator $dot(phi)_i^ot$ and two columns showing the oscillator phases each layer $phi_i^ot$.
The right-most column shows the coupling matrix for the immune layer $bold(Kappa)^2$.
Each layer is sorted first from lowest to highest frequency and secondary by lowest to highest phase for better clarity.
Rows C and C' share the same parameterization but are different samples from the same initialization distributions.

#figure(
  image("images/snapshots.svg", width: 100%),
  caption: [
    Snapshots of different #acr("DNM") parametrization and initialization. Configuration A can be regarded as healthy, with phases and frequencies being fully synchronized.
    In contrast, B and C are pathologic, due to their clustering in $dot(phi)^1$. Configuration C' corresponds to a vulnerable state, because of uniformly distributed phases (splay state).
    Lastly, D is regarded as resilient, since the phases exhibit clustering, but the frequencies $dot(phi)^1$ are synchronized.
    #TODO[$Kappa$ colorbar]
  ],
) <fig:snap>
Row A in @fig:snap is fully synchronized/coherent since it not only has the frequencies synchronized but also the phases and can therefore interpreted as healthy.
The coherence can also be seen in the fully homogeneous coupling matrices where both $bold(Kappa)^ot$ show the same coupling strength for every oscillator pair.
The rows B and C in contrast show signs of a pathological state, here both the frequencies three and phases have four distinct clusters respectively.
The clusters are also visible in the coupling matrices, where the coupling strength differs based on the cluster #todo[which is stronger?].
The row for C', even though having the same parameterization as C, can be regarded vulnerable, since the phases uniformly distribute in the $[0, 2pi)$ interval ($R^ot = 0$), while the frequencies are synchronized.
Coupling matrices for C' show traveling waves, which are characteristic for splay states.
Observing different results for different initializations justifies the introduction of ensembles.
Lastly row D shows a resilient state, where the phases are clustered while the frequencies are synchronized.

For the next result, the ensembles were introduced, every configuration of A, B, C and D was simulated for $M=50$ different initializations over an interval of $t=2000$.
The two left-most columns show the standard deviation of the mean phase velocities $s^ot$ for each ensemble member $m$.
The plots show the temporal evolution of metrics for quantifying phase and frequency coherence, with the two right-most columns of @fig:ensemble show the temporal behavior of the Kuramoto Order Parameter for each individual ensemble member $m in 1,2...M$.
Where lower values for $R^ot$ indicate decoherence in phases, with its minimum $R^ot = 0$ coincides with a splay state, and for $s^ot$ higher values indicate a larger amount of frequency decoherence and clustering.
#figure(
  image("images/ensembles.svg", width: 100%),
  caption: [
    Transient and temporal evolution of the phase- and frequency-synchronization metrics $R^ot$ and $s^ot$, for ensembles of the #acr("DNM") for the configurations listed in @tab:siminit.
    Emphasizing the influence of $beta$ and $sigma$ on the systems synchronization behavior, and presenting different stable emergent system states.
  ],
) <fig:ensemble>

Every ensemble in @fig:ensemble shows decoherence for early time-points, which is expected for randomly initialized variables, but changes relatively fast through a transient phase $t in [0.0, 200]$ into systematic stable behavior.
The stable states align with the observations made for @fig:snap, configuration A has, besides small jitter, mostly synchronized frequencies $s^ot approx 0$.
Also the phases of configuration A are mostly synchronized with $R^ot approx 1$, only two initializations show decoherence and are oscillating between weak clustering and almost full incoherence.
Medically this can be interpreted as a low risk of a dysregulated host response for an otherwise healthy response to the initial cytokine activation.
For configuration B the amount of incoherence inside the ensemble is clearly visible, with $s^ot$ being positive and some more ensemble members exhibiting clustering, indicated by a Kuramoto Order Parameter slightly less than $1$.
In configuration C the incoherence is even more prominent, larger $s^ot$ and some ensemble members evolving into a splay state, i.e. $R^ot=0$.
For configuration D the overall phase incoherence is again a bit less compared to C, and lower for the organ compared to the immune layer.
The phases are coherent for the organ layer but seem almost chaotic for the immune layer, some are synchronized, while others are clustered, in a chimera or almost splay-like state.
Over the whole simulation period, the coherency in the immune layer seems not to fully stabilize, rather oscillate around an attractor.

Each of the configurations only differs in the parameter choices for $beta$ and $sigma$, yet they evolve into unique and distinct system states.
To put these four specific configurations into broader perspective, a grid of $beta$ and $sigma$ was simulated, in the interval $beta in [0, 1]$ with a grid resolution of $0.01$ and $sigma in [0, 1.5]$ with a resolution of $0.015$, creating a grid of $10,000$ points.
In @fig:phase the metric $s^ot$ is shown in the $beta-sigma$ phase space for both layers in the first row, where brighter colors indicate a larger risk of frequency desynchronization, or risk of dysregulated immune response.
The second row shows the ensemble mean over $overline(R)^ot$, i.e. $overline(R)^ot = 1/M sum^M_m R^ot_m$, with $M=50$, where darker colors indicate larger phase desynchronization.
The white rectangle indicates the simulated region in @osc2, $beta in [0.4, 0.7]$ and $sigma in [0, 1.5]$ for reference.
Coordinates of the configurations A, B, C, and D are also labeled.
#figure(
  image("images/phase.svg", width: 100%),
  caption: [
    Phase space of the parameters $beta$ and $sigma$ and illustrating the broader picture their influence on the frequency and phase synchronization of the #acr("DNM").
    White rectangle indicates the grid-limits of the original publication @osc2.
  ],
) <fig:phase>
Generally there is a similarity between phase and frequency desynchronization but no full equality, meaning there are parameter regions where the phase is synchronized and frequency desynchronized and vice versa.
Another observation, that smaller values of $beta < 0.55$ correspond to less desynchronization and stronger coherence, which is in line with the medical interpretation of $beta$ where smaller values indicate a younger and more healthy biological age.
When crossing a critical value of $beta_c approx 0.55$ for the frequency and $beta_c approx 0.6$ for the phases, the synchronization behavior suddenly changes and tends towards incoherence, clustering and pathological interpretations.

For small values of $sigma < 0.5$ the frequency synchronization and $sigma < 0.25$ for the phase synchronization, the behavior significantly differs between immune and organ layer.
The immune layer tends to fully desynchronize, instead the organ layer only the frequency desynchronizes for larger $beta > 0.7$ .
With larger values of $sigma > 0.5$ the dynamics more or less harmonize between layers and metrics and are mostly depend on $beta$.

== Why care about the DNM?
#todo[not a sequence]
Mechanistic baseline
Constraint setting


= Method (Latent Dynamics Model) <sec:ldm>

This chapter introduces the methodological framework used to address the first research question stated in @sec:problemdef:
#align(
  center,
  [*Usability of the #acr("DNM")*: How and to what extent can the #acr("ML")-determined trajectories of the #acr("DNM") be used for detection and prediction, especially of critical infection states and mortality.#todo[format]],
)

To investigate this, a deep learning pipeline has been developed, in which the #acr("DNM") is embedded as central part.
Instead of predicting the sepsis directly, the two components, #acl("SI") and increase in #acr("SOFA") scores are predicted as direct proxies creating more nuanced and therefore more interpretable results.
For the increase in #acr("SOFA") component, the main idea is to utilize the parameter level synchronization dynamics, particularly $cmbeta(beta)$ and $cmsigma(sigma)$, of the functional #acr("DNM") which are expected to describe organ failure on a systemic level.
The complete architecture, consisting of the #acr("DNM") and additional auxiliary modules, which will be referred to as the #acr("LDM") from now on.

This chapter proceeds with the prediction task to be reiterated formally and the introduction of desired prediction properties, and justification of modeling choices.
Afterwards, the individual modules of the #acr("LDM") will be discussed and focusing on what purpose each serves and how it is integrated into the broader system.
#todo[ref sections]
#todo[Notation table]

== Formalizing the Prediction Task <sec:formal>
In automated clinical prediction systems, a patient is typically represented through their #acr("EHR").
Where the #acr("EHR") aggregates multiple clinical variables, such as laboratory biomarker, for example from blood or urine tests, or physiological scores and, further demographic information, e.g. age and gender.
Using the information in the #acr("EHR"), the objective is to estimate the patient's risk of developing sepsis in the near future.

=== Patient Representation
Let $t$ be an arbitrary chosen time-point of a patients #acr("ICU")-stay and the available #acr("EHR") at that time consisting of $n$ variables.
After imputation of missing values, normalization, and encoding of non-numerical quantities, each variable $mu_j$ is mapped to a numerical value:
$
  mu_(t,j) in RR, " " j = 1,...,n
$
These values are collected into a column-vector:
$
  bold(mu)_t = (mu_(t,1),..., mu_(t,n))^T in RR^n
$
which is fully describing the current physiological state of the #acr("ICU")-patient.

=== Modeling the Sepsis-3 Target
The goal is calculate the risk of patient developing a septic condition given an initial observation $bold(mu)_(t=0)$ in the next $T$ future time-steps.
Following the Sepsis-3 definition, the risk requires both suspected infection and multi-organ failure.
Defining the _sepsis onset event_ $S$ as the occurrence of the Sepsis-3 criteria at any time point within the window $t=0,...,T$:
$
  S_(0:T) := union.big_(t=0)^T (A_t inter I_t)
$

Here $A_t={Delta O_t >= 2}$, is denoting an acute change in organ function, more specifically a worsening of the organ system, i.e. multi organ failure.
With $O_t$ being the #acr("SOFA")-score and $Delta O_t=O_t-O_(t-1)$ the change in #acr("SOFA")-score with respect to the previous time-step.
$I_t$ is an event indicator for a #acl("SI") at time $t$.
The target probability given the current #acr("EHR") is then:
$
  Pr(S_(0:T)|bold(mu)_0) = Pr(union.big^T_(t=0)(A_t inter I_t) | bold(mu)_0)
$
=== Heuristic Scoring and Risk Estimation <sec:heu>
The direct estimation of the conditional probability $Pr(S_(0:T)|bold(mu))$ is computationally and statistically challenging due to the temporal dependency between the binary Sepsis-3 criteria.
To make the prediction of this probability more tractable but still connect the statistical model to the clinical definition the following assumptions and modeling choices are made resulting in a _heuristic risk score_ $tilde(S)$:

#list(
[*Independence between infection and organ failure*\
The strongest assumption is the independence of infection $I_t$ and multi-organ failure $A_t$.
Clinically it is known that a majority of situations with multi-organ failure stem from an underlying infection, meaning they exhibits strong partial correlations.
Yet this assumptions allows to treat both components separately for the prediction and enhances the overall interpretability.],

[*Short-horizon infection stability*\
For small $T$, relative to the total #acr("SI")-window of 72 h, $I_t$ is approximated as constant over the sequence: $tilde(I) approx I_t$ for $t=0,...,T$.
This binary variable serves as a time-invariant proxy for the presence of a #acl("SI") and can be estimated from $bold(mu)_0$.
Purely clinically the presence of an infection is difficult to define generally, and exact starting and ending times complicated to measure.
Once a infection is detected and antibiotic treatment initiated, the patient likely to be infectious for the next couple of hours, but not for the next couple of days.],

[*Temporal independence of organ worsening events*\
The events $A_t$ are statistically independent across time steps.
This is necessary to aggregate the risk across time-points:
$ Pr(A_(0:T)) = 1 - product^T_(t=0)Pr(A_t) $
Instead of predicting $Pr(A_(0:T))$ directly, first the #acr("SOFA")-score for each time-step $hat(O)_t$ is estimated from $bold(mu)$.
These estimated scores are then used to create a non-linear summary statistic $tilde(A)$ that relates to the formula of the probability of a union of events:
$
  tilde(A) = o_(s,d)(hat(O)_(0:T)) =  1 - product^T_(t=1) "sigmoid"(s(hat(O)_t - hat(O)_(t-1) - d))
$
where the learnable parameters $d$ and $s$ of the function $o_(s,d) (dot)$ being a calibration threshold and scale respectively.
In the original Sepsis-3 definition $d$ is chosen as two.
The choice of the $"sigmoid"$ function in the product-sequence ensures monotonicity (larger increase $->$ more likely organ failure) and the aggregation of temporal risks into a single measure.
This risk function is used as a summary statistic for the overall risk of #acr("SOFA")-score increase within the window but is not a strict probability, rather a smoothed approximation.])

The high-dimensional $bold(mu)_0$ has now been condensed into two clinically motivated summary statistics $tilde(A)$ and $tilde(I)$.
The final sepsis risk is then estimated by combining these features treated as independent events:
$
 tilde(S) = S_(0:T) approx tilde(A) tilde(I)
$
The interaction term $tilde(A) tilde(I)$ is essential as the formal Sepsis-3 definition is based of the conjunction of the two events.

It is important to note that $tilde(S)$ is *not a calibrated probability* but a heuristically derived and empirical risk score based on the Sepsis-3 definition, serving as proxy to the real event probability $P(S_(0:T)|bold(mu)_0)$.

== Architecture

To estimate the components $tilde(A)$ and $tilde(I)$ from $bold(mu)_0$ two #acl("DL") modules have been designed.
A flow-chart overview  in @fig:flow summarizes same information provided in @sec:formal but also integrates the different learnable neural modules, indicated by the parameters $theta$.


#figure(
  scale(high_fig, 65%),
  caption: [
    Flow chart of the different steps taken to produce the heuristic sepsis risk measure $tilde(S)$ from an observed #acl("EHR") $bold(mu)_0$.
    Learnable neural function parameters are indicated by a $theta$ subscript.
  ]
) <fig:flow>

After explaining how the #acr("DNM") is used to model the severity of organ failure, which is not directly shown in the flow chart, each #acr("DL")-module is introduced in the following subsections.

=== Infection Indicator Module
The first module of the #acr("LDM") estimates the presence of a #acr("SI"), represented by the binary indicator $tilde(I)$.
Given $N$ pairs of #acr("EHR") vectors and ground truth #acr("SI")-indicator
$
(bold(mu)_i, tilde(I)_i), i = 1...N
$
a parameterized non-linear function
$
f_theta: RR^n -> [0,1]
$
is trained to map the patients physiological state to an estimated probability of suspected infection:

$
  hat(I)=f_theta (bold(mu)_0)
$

The model is implemented as a supervised neural network optimized with gradient descent.
To fit model, the #acr("BCE")-loss which measures the distance between true label $y_i$ and the predicted label $hat(y)_i$:
$
  L_"inf" = B C E (y, hat(y)) = -1/N sum^N_(i=1) [y_i log(hat(y)_i) + (1-y_i)log(1-hat(y)_i)]
$
is minimized during training.
The resulting estimator provides a stable time-invariant proxy for suspected infection over the short prediction horizon.

=== SOFA Predictor Module
The complete #acr("SOFA") predictor module is composed of multiple smaller submodules, described below.
 Most notably it connects the #acr("EHR") with #acr("SOFA") estimations through the #acr("DNM") parameters $cmbeta(beta)$ and $cmsigma(sigma)$, which is described in @sec:theory_fsq.
@sec:theory_enc presents how #acr("EHR") information is embedded into the #acr("DNM") parameter space, and @sec:theory_gru how the evolution of the patient state is modeled.

==== DNM Surrogate <sec:theory_fsq>
As discussed in @sec:heu the summarized organ-condition statistic $tilde(A)$ depends on estimates #acr("SOFA")-scores at future time-steps $hat(O)_(0:T)$.

These scores are obtained through a multi-stage process that integrates the dynamics of the #acr("DNM") with learned patient-specific latent representations.

Recalling that the pathological organ conditions within the #acr("DNM") are characterized by frequency clustering in the parenchymal layer.
The amount of frequency clustering is measured by the ensemble average standard deviation of the mean phase velocity $s^1$ (see @eq:std).
Naturally this measure can be used as a proxy for a patients #acr("SOFA")-score.
Increasing values of $s^1$ indicate a higher #acr("SOFA")-score and a worse condition of the patients organ system.

Numerical integration of the DNM equations for a given parameter pair $(cmbeta(beta), cmsigma(sigma))$ yields the corresponding #acr("SOFA") estimate $hat(O)_t$:
$
  hat(O) = s^1 (cmbeta(beta), cmsigma(sigma))
$
these two parameters where identified as the most influential quantities in the original #acr("DNM") publications @osc2.
Every other parameter is assumed constant and chosen as listed in @tab:init.

In order to massively reduce the computational burden of computing the $s^1$-metric for the continuous space in $cmbeta(beta) in [0.4pi, 0.7pi]$ and $cmsigma(sigma) in [0.0, 1.5]$ the space has been quantized and pre-computed.

For an estimated coordinate pair $hat(z)=(hat(z)_cmbeta(beta), hat(z)_cmsigma(sigma))$ in the continuous $(cmbeta(beta), cmsigma(sigma))$-space the quantized metrics are interpolated by smoothing nearby quantization points with a gaussian kernel, which is illustrated in @fig:fsq.
This allows for a continuous space approximation from the quantized space, while allowing for the pre-computation of the quantized space and therefore drastically reducing the computational expenses.

#figure(
fsq_fig,
caption: [Quantized latent lookup of precomputed synchronization metrics.
Point colors represent the amount of desynchronization.
Neighboring points (the $3times 3$ sub-grid indicated by the red outlines) are used smoothed using a gaussian kernel, represented by the color gradient around estimation point $hat(z)$.
This allows to continuously interpolate the parameter space.
],
) <fig:fsq>

This quantization strategy, called _latent lookup_ is closely related to #acr("FSQ") @Placeholder, used in Dreamer V3 @Placeholder for example.
Details on the latent lookup implementation, including grid-resolution and kernel size, can be found in @sec:impl_fsq.

#todo[explain STE (straight through estimation)]

==== Latent Parameter Encoder <sec:theory_enc>
To connect the high-dimensional #acr("EHR") to the dynamical regime of the #acr("DNM"), a neural encoder:

$
  g_theta: RR^n -> RR^2 times RR^h = RR^(2+h)
$
maps the patient state to a two-dimensional latent vector

$
  hat(z)_0 = (hat(z)_(0,cmbeta(beta)), hat(z)_(0,cmsigma(sigma))) = g_theta (bold(mu)_0)
$
This embedding locates the patient within a physiologically meaningful region of the #acr("DNM") parameter space, which in context of the #acr("LDM") is called the latent space.
The latent coordinate $hat(z)_0$ provides the initial condition for short-term dynamical organ condition forecasting.

Here the latent-lookup introduced in the previous @sec:theory_fsq comes in to play.
Unlike classical #acr("PINN") @Placeholder where the gradients of the #acr("ODE") integration provide useful information to the encoder module directly, here the gradient information is provided by the nearby quantized points which contribute to estimated synchronicity measure through the gaussian smoothing.

In addition to the estimated system parameter $bold(z)_0$, the encoder outputs another vector with dimension $h<<n$ that is a compressed representation of patient physiology relevant for short-term evolution of $hat(bold(z))$.
This vector $bold(h) in RR^h$ is referred to as the hidden state.

Since both output of the function $g_theta$ mark the initial step of the prediction horizon, they receive a $0$ as subscript $hat(bold(z))_0$ and $bold(h)_0$.

The placement of $N$ latent points $bold(z)$ is driven by a supervision signal between the known $O_0$ and the predicted #acr("SOFA")-score $hat(O)_0$
$
  L_"enc" = M S E(bold(O)_0, bold(hat(O))_0) = 1/N sum^N_(i=1) (O_(0,i) - hat(O)_(0,i))^2
$
with #acr("MSE") as the loss function.

==== Recurrent Parameter Dynamics <sec:theory_gru>
Since the heuristic #acr("SOFA") risk $tilde(A)$ depends on the evolution of organ function, it is necessary to estimate not only the initial state $hat(z)_0$ but also its evolution.
For this purpose a neural recurrent function:

$
  hat(bold(z))_t, bold(h)_t = r_theta (bold(z)_(t-1), bold(h)_(t-1)), "  " t = 1,...,T
$
is trained to propagate the latent #acr("DNM") parameters forward in time.
This recurrent mechanism captures how the underlying physiology influences the drift of the DNM parameters, and therefore how the level of synchrony changes across the prediction horizon, which translates to the pathological evolution of patients.

Here again the placement of latent points $bold(z)$ is guided by a supervision signal:
$
  L_"recurrent" = M S E(bold(O)_t, hat(O)_t) = 1/T sum^T_(i=1) (O_t - hat(O)_t)^2
$
and through the #acr("MSE")-loss.

=== Decoder <sec:theory_dec>
As shown in the visualization of the #acr("DNM") phase space in @fig:phase multiple latent coordinates $bold(z)$ result in the same amount of desynchronization, which is not surprising, since different physiological states share the same #acr("SOFA") level.
But even though different physiological states have a common #acr("SOFA")-score, their latent representations should be unique.
This should enable to distinguish different triggers of the organ failure inside the latent space, similarly to how it is possible to distinguish the different triggers from the #acr("EHR").

In a classical Auto-Encoder @Placeholder setting, to encourage a semantically structured latent space, a decoder module is added as an auxiliary regularization component.
A neural decoder network:

$
  d_theta: RR^2 times [0,1] times [0,1] -> RR^n
$
 attempts to reconstruct the original #acr("EHR") features from the latent representation, the resulting desynchronicity of that latent coordinate and the heuristic risk measures:

$
 hat(bold(mu))_t  = d_theta (hat(bold(z))_(t), tilde(s^1_t), tilde(I))
$
where the gradients only flow through $bold(z)$, the amount of desynchronicity $s^1_t$ and the risk measure $tilde(I)$ are provided as additional information.
This way the decoder only learns to disentangle the latent coordinates in $bold(z)_t$ based on the ground truth $bold(mu)_t$, through a supervised loss:
$
  L_"dec" = M S E(bold(mu)_t, bold(hat(mu))_t) = 1/T sum^T_(i=0) (bold(mu)_t - bold(hat(mu))_t)^2 
$


The formulation is based on the assumption:
$
  bold(mu)_t = hat(bold(mu))_t + epsilon
$
with $epsilon$ the measurement noise, to hold.

This latent regularization is motivated by _Representation Learning_ @Bengio2012Representation and ensures that nearby points in the latent $(cmbeta(beta), cmsigma(sigma))$-space correspond to physiologically similar patient states.
It promotes the encoder $f_theta$ to learn a meaningful alignment between #acr("EHR")-derived latent-embeddings and the dynamical #acr("DNM") landscape.

Using this regularization the recurrent predictor $g_theta$ is encouraged to map temporally consecutive to spatially near latent coordinates, since it is expected that consecutive #acr("EHR")s do not exhibit drastic changes.
Leading smooth patient trajectories through the latent space.

== Overall Training Objective


= State of the Art <sec:sota>
== Model Based Methods
== Data Based Methods
=== Selected Works


= Experiments
== Task - Definition of Ins and Outs
=== Latent Lookup Implementation <sec:impl_fsq> 
== Data
#figure(
  image("images/yaib_sets.svg", width: 100%),
  caption: [
    Sets of @yaib
  ],
)<fig:sets>
RICU and YAIB use delta_cummin function, i.e. the delta #acr("SOFA") increase is calculated with respect to the lowest observed #acr("SOFA") to this point.
=== MIMIC-III/IV
=== YAIB + (Further) Preprocessing
==== ricu-Concepts
== Metrics (How to validate performance?)

= Results <sec:experiment>

= Conclusion

= Appendix
<sec:appendix>
#figure(
  table(
    columns: (1fr, auto, auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    table.header([Category], [Indicator], [1], [2], [3], [4]),
    [Respiration], [$"PaO"_2$/$"FiO"_2$ [mmHg]], [< 400], [< 300], [< 200], [< 100],

    [], [Mechanical Ventilation], [], [], [yes], [yes],
    [Coagulation], [Platelets [$times 10^3/"mm"^3$]], [< 150], [< 100], [< 50], [< 20],

    [Liver], [Bilirubin [$"mg"/"dl"$]], [1.2-1.9], [2.0-5.9], [6.0-11.9], [> 12.0],

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
    [Central Nervous System], [Glasgow Coma Score], [13-14], [10-12], [6-9], [< 6],

    [Renal], [Creatinine [$"mg"/"dl"$]], [1.2-1.9], [2.0-3.4], [3.5-4.9], [> 5.0],

    [], [or Urine Output [$"ml"/"day"$]], [], [], [< 500], [< 200],
  ),
) <tab:sofa>
#todo[caption]
