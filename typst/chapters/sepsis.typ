#import "../thesis_env.typ": *

= Medical Background (Sepsis) <sec:sepsis>
In 2017, an estimated 48.9 million people worldwide suffered from sepsis and the same year, 11 million deaths were associated with sepsis @rudd2020global.
This makes up 19.7% of yearly deaths, making it the most common cause of in-hospital deaths.
Even with successful treatment, around 40% of those affected suffer long-term consequences, such as cognitive, physical or physiological problems, the so called _post-sepsis syndrome_ @vanderSlikke2020post.
The observed risk of mortality significantly differs between lower to middle income countries with $>50%$ and high income countries with $<25%$.
Overall, septic diseases represent an enormous burden on the global healthcare system.

Quantitatively, almost half of all sepsis-related deaths occur as a secondary complication of an underlying injury or a non-communicable, also known as a chronic disease @fleischmann2022sepsis.
But the underlying biological triggers for sepsis as well as individual patient progressions are highly diverse and heterogeneous.
Moreover, a septic condition can not be reduced to a single specific physiological phenomenon, instead it combines multiple complex and interdependent processes across different biological scales.
This makes it notoriously difficult to study and diagnose in clinical practice.

Starting with @sec:sepbio, the pathology and underlying biology are described phenomenologically.
Followed by @sec:sep3def, which provides a more medical and technical overview, this subsection is structured around the most commonly used and widely accepted sepsis definition, which is referred to as _Sepsis-3_, it characterizes sepsis as a "life-threatening organ dysfunction caused by a dysregulated host response to infection" @Sepsis3..
In @sec:sepwhy, the necessity for reliable and clinically practical sepsis prediction systems is discussed and how these systems are utilized in clinical practice today.

== Biology of Sepsis <sec:sepbio>
This subsection tries to give an introduction into the biological phenomena that underlie sepsis.
Starting with an explanation on how human tissue is reacting to local infections or injuries on a cellular level in @sec:cell and how this can escalate to _cytokine storms_ in @sec:storm and ending with systemic organ failure in @sec:fail.

Certain details and specifities are out of scope for this project and are not addressed.
More detailed explanations can be found in the primary resources provided throughout this section.

=== Cellular Origins <sec:cell>
Human organ tissue can be differentiated into two broad cell-families called _parenchymal_ and _stroma_ which are separated by a thin, specialized boundary layer known as the _basal lamina_.

The parenchymal cells perform the primary physiological functions of an organ, with every organ hosting distinct parenchymal cells.
For example, Cardiomyocytes in the heart drive the contraction, relaxation and therefore blood pumping, Hepatocytes in the liver doing metabolism and detoxification and Neurons in the brain providing signal transmission @Chen2022Crosstalk.

Everything that is not providing organ-specific functionalities forms the stroma, that includes the structural or connective tissue, blood vessels and nerves.
The stroma not only contributes to the tissues structure, but it also actively participates in biochemical signaling and immune regulation.
In this way, it helps to maintain a healthy and balanced tissue, the _homeostasis_, and enables coordinated responses to injury or infection @Honan2021stroma.

#figure(
  grid(
  columns: 2,
  image("../images/overview_cells.jpg"),

  image("../images/cells_micro_cropped.jpg", width:60%)),

  caption: flex-caption(short: [Illustration and microscopic image of Stroma and Parenchymal Cells], long:[On the left hand side, an illustration of parenchymal cells sitting on top of the basel lamina (thin light-blue section) and the stroma in very light pinkish color building the base. The illustration shows the parenchymal cells in different shapes and organization @DigitalHistology_overview. On the right hand side a microscopic image of stomach tissue (400x magnification), with tightly packed parenchymal cells on the top and stroma cells as connective tissue in the light-pinkish color. The basal lamina is not visible at this level of magnification @DigitalHistology_cells.])
) <fig:cells>

Any organism that can cause a disease is called a pathogen, this includes bacteria, fungi, algae, and parasites.
When a pathogen enters the body through the skin, a mucous membrane or an open wound, the first line of nonspecific defense, the innate immune system, gets activated @Fischer2022Innit.

This rapid response does not require the body to have seen the specific pathogen before, as opposed to a slower more specific and adapted immune response.
Instead, the innate immune system can be triggered by sensing commonly shared features of pathogens, in case of germs known as #acr("PAMP"), for injury called #acr("DAMP") @Jarczak2021sepsis.
The #acr("PAMP")s and #acr("DAMP")s can be detected by #acr("PRR"), which are found in resident immune cells, as well as stroma cells.

Once a pathogen is detected a chain reaction inside the cell leads to the creation and release of signaling proteins called _cytokines_ @Zhang2007cyto.
Cytokines are a diverse group of small signaling proteins which play a special role in the communication between other cells, both neighboring and across larger distances through the bloodstream.
They act as molecular messengers that coordinate the recruitment of circulating immune cells and will guide them to the location of infection or injury @Zhang2007cyto.
Additionally, they play a role in immune activation where cytokines regulate the production of anti- and pro-inflammatory immune cells which help with the elimination of pathogens and trigger the healing process right after.

One specialty of these relatively simple proteins is that they can be produced by almost every other cell, with different cells can produce the same cytokine.
Further, cytokines are redundant, meaning targeted cells can show identical responses to different cytokines @House2007cyto, these features seem to fulfill some kind of safety mechanism to guarantee vital communication flow.
After release cytokines have relatively a short half-life (only a few minutes) but through cascading-effects the cytokines can have substantial impact on their micro-environment.

=== Cytokine Storms <sec:storm>
The septic condition is primarily driven by the excessive and uncontrolled release cytokines and other mediators.
Under normal circumstances, the release of inflammatory cytokines tightly regulated in time and magnitude.
After a pathogen is detected, the release of cytokines is quickly initiated.
The release peaks as immune cells are recruited and then automatically fades out once the initial pathogen is controlled and the host returns to a healthy and balanced state, the homeostasis.

In certain scenarios a disturbance to the regulatory mechanisms triggers positive inflammatory feedback loop, accompanied by a massive release of pro-inflammatory cytokines.
These cytokines further activate additional immune and non-immune cells, which in turn amplify the cytokine production, creating a self-reinforcing cycle of immune activation @Jarczak2022storm.
Ultimately, this leads to a continuous and uncontrolled release of cytokines that fails to shut down, this corresponds to the "hosts dysregulated response to an infection" of the Sepsis-3 definition.
With this overreaction, called _cytokine storm_, the immune response and release of inflammatory mediators can damage the body more than the infection itself.

Although the quantity of cytokines roughly correlates with disease severity, concentrations of cytokines vary between patients, time and even different body-parts, making it almost impossible to distinguish between an appropriate reaction and a harmful overreaction @Jarczak2022storm.
Out of all cytokines, clinicians can measure only a small subset or secondary markers through blood samples to detect increased cytokine activity.
This limited accessibility makes cytokines difficult to study in general, and generally, they prove to be little useful as direct indicators of pathogenesis or for diagnostic purposes.

This emphasizes the complexity of sepsis as a systemic syndrome rather than a single-cause disease, and suggests that cytokine storms are an emergent property rather than the result of any one molecular trigger.
To this day, the fundamental principles that govern the transition from a regulated immune response to a self-destructive cytokine storm remain not fully understood.
Since the 90s there has been a lot of research focused on cytokines and their role in the innate immune system and overall activation behavior.
Multiple therapeutic interventions have been tested in clinical trials, yet none have achieved a significant improvement in survival outcomes @Jarczak2021sepsis.

=== Systemic Consequences and Organ Failure <sec:fail>
As more cytokines accumulate, they flood not only infected areas, but also surrounding parts of the tissue and circulation, causing the localized inflammatory response to become systemic.
The widespread cytokine reaction starts to disrupt the normal metabolism of parenchymal cells in organs due to a deficiency in oxygen and nutrients.

To compensate, cells switch from their usual oxygen-based metabolism to an _anaerobic glycolysis_ @Prieto2016Anaerobic, generating energy less efficiently from glucose.
As a result, metabolic by-products such as lactate accumulate making the surrounding environment more acidic, which further harms the cells and leads to more cellular dysfunction.

At the same time, the cellular mitochondria start to fail.
The walls of blood vessels become leaky, allowing fluids to move into surrounding tissue.
This causes swelling and lowers the blood pressure, which in turn reduces the oxygen supply even further @Jarczak2021sepsis.

Step by step, the death of cells spreads throughout the body and affects organ functionality.
When multiple organs fail simultaneously, the condition becomes irreversible @Sepsis3.
A patient at this stage is in septic shock, the final and most deadly lethal form of sepsis, with each additional affected organ the mortality increases drastically.

== Sepsis-3 Definition <sec:sep3def>
#reset-acronym("SOFA")
As illustrated in the previous section, it is difficult to pin-point the exact moment at which the immune response switches from normal to dysregulated behavior.
To classify patients at septic multiple clinically grounded definitions have been proposed over time.
The most up to date and widely used definition, called Sepsis-3, will be introduced in this section.
The Sepsis-3 definition was created by a working group led by the "Society of Critical Care Medicine and the European Society of Intensive Care Medicine" in 2016.
Their resolution, named "Third International Consensus Definitions for Sepsis and Septic Shock" @Sepsis3, hence the name Sepsis-3, provides until today the most widely used sepsis definition and guidance on sepsis identification.

In general, sepsis does not classify as a specific illness, rather a multifaceted condition of "physiologic, pathologic, and biochemical abnormalities" @Sepsis3, and septic patient progressions are largely heterogeneous.
Also the trigger is explicitly nonspecific, since different triggers can cause the same septic condition.
Most commonly the underlying cause of sepsis is diarrhoeal disease, the most common underlying injury stems from road traffic injuries and maternal disorders the most common non-communicable disease causing sepsis @rudd2020global.

According to the Sepsis-3 definition, a patient is in a septic condition if the following two criteria are fulfilled:
#(
  align(center,     align(left, [
  + #text()[
      *Confirmed or Suspected Infection*, which has no strict medical definition or classification, meaning what counts as #acr("SI") remains vague.
      Ultimately it is left for the medical personnel to classify infections or the suspicion of infections.
      For retrospective data-driven classification it is suggested to characterize any patient prescribed with #acr("ABX") followed by the cultivation of body fluids, or the other way around, with a #acr("SI") @Sepsis3.\
      The timings of prescription and fluid samplings play a crucial role.
      If the antibiotics were administered first, then the cultivation has to be done in the first 24 hours after first prescription, if the cultivation happened first, the #acr("ABX") have to be prescribed in the following 72 hours @Sepsis3.
      These timings can be seen in the lower part of @fig:ricu (with the abbreviated #acr("ABX")).
      Regardless which happened first, the earlier of the two times is treated as the #acr("SI")-onset time.]

  + #text()[
      *Dysregulated Host Response* is characterized by the worsening of organ functionality over time.
      To measure the "amount of dysregulation" the Sepsis-3 consensus relies on the #acr("SOFA")-score introduced in @SOFAscore@Sepsis3.
      Nowadays, the score is regularly used to evaluate the functionality of organ systems and helps to predict the risk of mortality, also outside of a sepsis context.
      The #acr("SOFA")-score is calculated at least every 24 hours and assesses six different organ systems by assigning a score from 0 (normal function) to 4 (high degree of dysfunction) to each.
      The overall score is calculated as sum of each individual organ system.\
      Included organ systems are the respiratory system, the coagulation/clotting of blood, i.e. changing from liquid to gel, the liver system, the cardiovascular system, the central nervous system and the renal system/kidney function.
      A more detailed listing of corresponding markers for each organ assessment can be found in table @tab:sofa in the @a:sofa.
      The magnitude of a patients initial #acr("SOFA")-score captures preexisting organ dysfunction.
      An increase in #acr("SOFA")-score $>=2$, in consecutive assessments, corresponds to an acute worsening of organ functionalities and a drastic worsening in the patients condition, which is used as the indicator to a dysregulated host response.]
]),
  )
)
The combination of the two criteria represents an exaggerated immune reaction that results in organ dysfunction, when infection is first suspected, even modest organ dysfunction is linked to a 10% increase of in-hospital mortality.


=== Sepsis Classification
The Sepsis-3 definition not only provides the clinical criteria of septic conditions, but also introduces the necessary time windows for sepsis classification.
An increase of #acr("SOFA") $>=2$ in the 48 hours before or 24 hours after the #acr("SI") time, the so called #acr("SI")-window, is per Sepsis-3 definition the _sepsis onset time_.
A schematic of the timings is shown in @fig:ricu.

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
Using the lowest #acr("SOFA") score as baseline, the increase $>=2$ for patients with inspected infection was associated with an 18% higher mortality rate according to @SOFAscore, a retrospective #acr("ICU")-data analysis.

#figure(
  image("../images/sofa-sep-3-1.png", width: 100%),
  caption: flex-caption(
  short: [Timings of the Sepsis-3 definition],
  long: [
    Graphical representation of the timings in the Sepsis-3 definition, taken from @ricufig.
  ]),
)<fig:ricu>

Up until today, even though #acr("SOFA") was created as a clinical bedside score, some of the markers used in it are not always available to measure or at least not at every 24 hours @moreno2023sofaupdate.
For a faster bedside assessment a clinical score termed #acr("qSOFA") has been introduced @SOFAscore, with highly reduced marker number and complexity, it includes:
#(
  align(center, list(
    align(left, [Respiratory rate $>=$ 22/min]),
    align(left, [Altered mentation]),
    align(left, [Systolic blood pressure $<=$ 100 mmHg]),
  ))
)
Patients fulfilling at least two of these criteria have an increased risk of organ failure.
While the #acr("qSOFA") has significantly reduced complexity and is faster to assess, it is not as accurate as the #acr("SOFA") score, meaning it has less predictive validity for in-hospital mortality @SOFAscore.

There is also the notion of a septic shock, also defined in @Sepsis3, corresponding to an in-hospital mortality above 40%.
Patients with a septic shock are can be identified when they fulfill all of these criteria:
#(
  align(center, list(
    align(left, [Sepsis]),
    align(left, [Persisting hypotension requiring\
      vasopressors to maintain MAP $>=$ 65mmHg]),
    align(left, [Serum lactate level > 2 mmol/L, despite volume resusciation.]),
  ))
)

== Sepsis Prediction <sec:sepwhy>
To this day, sepsis and the more extreme septic shock, remain as an extreme burden to the worldwide healthcare system.
It is associated with high rates of incidence, high mortality and significant morbidity.
Despite overall advancements in medical care and slowly decreasing prevalence numbers, sepsis continues to be the leading cause of in-hospital death @Via2024Burden.

In Germany, it was estimated in 2022 that at least 17.9% of intensive care patients develop sepsis, and 41.7% of all hospital treated sepsis patients die during their stay @fleischmann2022sepsis.
The economic burden is equally severe, with the annual cost of sepsis treatment in Germany estimated to be €7.7 billion based on extrapolated data from 2013 @fleischmann2022sepsis.

Globally sepsis remains to be under-diagnosed significantly due to its nonspecific symptoms.
Environmental and socioeconomic factors, such as insufficient sanitation, limited access to clean water and healthcare increase the incidence particularly in low- to middle income countries @rudd2020global@Via2024Burden.

Traditional sepsis screening has relied on clinical scoring systems such as #acr("SOFA") or #acr("qSOFA").
While useful for standardizing assessment, these scores are inherently reactive, since they identify patients already experiencing organ dysfunction rather than those at risk of developing sepsis.
This clinical reality has motivated the development of automated prediction systems that can continuously monitor patients and alert clinicians to elevated sepsis risk before over organ failure develops.
Usually, these alerts are based on predetermined criteria that are derived from clinical and laboratory measurements.
// Prediction horizons can vary substantially between systems, from now-casting, essentially predicting sepsis in realtime, short-term where sepsis is predicted in the next few hours and long-term where it is predicted over the next few days.
With the help of these automated alerts, clinicians can potentially more rapidly initiate antibiotic or other treatment or intensify patient monitoring.

=== Limitations of Current Prediction Systems
A meta-analysis of seven sepsis alert systems implemented in clinical practice found no evidence for improvement in patient outcomes, suggesting insufficient predictive power of analyzed alert systems or inadequate system integration @Alshaeba2025Effect.
Nevertheless, positive treatment outcomes depend heavily on timely recognition and intervention @Via2024Burden.
Each hour of delayed treatment increases mortality risk, underscoring the critical importance of early detection @seymour2017time.
Furthermore, structured screening and early warning systems have demonstrated reductions in time-to-antibiotics and improvements in outcomes @Westphal2009Early.
These findings confirm that earlier identification of sepsis improves clinical results and emphasize the need for developing reliable alert systems.

A recent study suggests a paradigm shift in sepsis detection, namely from a purely measurement and symptom based to a systems-based approach @Dobson2024Revolution.
Instead of waiting for clinical signs, i.e. symptoms, early recognition should integrate multiple physiological and biochemical signals to capture the transition from infection to organ dysfunction.
This aligns with the findings of a survey among clinicians regarding Artificial Intelligence Assistance in healthcare @EiniPorat2022.
According to the survey, one participant emphasizes that specific vitals signs might be of less importance, rather the trend of a patient's trajectory itself should be the prediction target.
Another finding of the same study was the preference of trajectories over plain binary event predictions.

Yet, the translation of predictive models into clinical practice has proven to be challenging.
Implementation studies consistently identify barriers, such as alert fatigue, where excessive false positives or clinically non-actionable alarms disrupt workflow, reduce clinician trust, and ultimately lead to ignored warnings.
Additionally, prediction systems face a fundamental trade-off: higher sensitivity captures more true cases but generates more false alarms, while higher specificity reduces alert fatigue but risks missing sepsis cases where early treatment is critical.
To be effective, predictive systems must integrate seamlessly into existing workflows provide interpretable output and support clinical expertise @EiniPorat2022.

Taken together, these insights highlight both the need and the opportunity for improved sepsis prediction.
The global burden and clinical urgency justify the development of more reliable prediction systems.
At the same time, the limitations of current alert systems and implementation barriers underline the necessity for models that can integrate dynamic patient data and capture clinical trajectories.
The following chapter examines existing approaches to sepsis prediction and identifies key gaps that motivate the present work.
