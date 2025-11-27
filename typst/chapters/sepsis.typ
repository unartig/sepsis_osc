#import "../thesis_env.typ": *

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
  image("../images/sofa-sep-3-1.png", width: 100%),
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

