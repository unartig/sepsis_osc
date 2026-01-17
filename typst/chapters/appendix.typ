#import "../thesis_env.typ": *

= Appendix
== SOFA - Details <a:sofa>
#figure(
  table(
    columns: (1fr, auto, auto, auto, auto, auto),
    inset: 10pt,
    align: (left, left, center, center, center, center),
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

  caption: flex-caption(short: [TODO], long: [TODO])
) <tab:sofa>

== DNM as Lie Formulation <a:lie>

== Input Concepts
#figure(table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, left, left, right, left),
  [*ricu - Name*], [*Unit*], [*Min*], [*Max*], [*Description*],
  
  [age], [Years], [0], [-], [Age at hospital admission],
  [sex], [-], [-], [-], [Female Sex],
  [height], [kg], [0], [-], [Patient height],
  [weight], [cm], [0], [-], [Patient weight],
  ),
  caption: [Static input features for the prediction task],
)
#figure(table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, left, left, right, left),
  [*ricu - Name*], [*Unit*], [*Min*], [*Max*], [*Description*],
  
  [alb], [g/dL], [0], [6], [albumin],
  [alp], [IU/L, U/l], [0], [-], [alkaline phosphatase],
  [alt], [IU/L, U/l], [0], [-], [alanine aminotransferase],
  [ast], [IU/L, U/l], [0], [-], [aspartate aminotransferase],
  [be], [mEq/L, mmol/l], [-25], [25], [base excess],
  [bicar], [mEq/L, mmol/l], [5], [50], [bicarbonate],
  [bili], [mg/dL], [0], [100], [total bilirubin],
  [bili_dir], [mg/dL], [0], [50], [bilirubin direct],
  [bnd], [%], [-], [-], [band form neutrophils],
  [bun], [mg/dL], [0], [200], [blood urea nitrogen],
  [ca], [mg/dL], [4], [20], [calcium],
  [cai], [mmol/L], [0.5], [2], [calcium ionized],
  [ck], [IU/L, U/l], [0], [-], [creatine kinase],
  [ckmb], [ng/mL], [0], [-], [creatine kinase MB],
  [cl], [mEq/L, mmol/l], [80], [130], [chloride],
  [crea], [mg/dL], [0], [15], [creatinine],
  [crp], [mg/L], [0], [-], [C-reactive protein],
  [dbp], [mmHg, mm Hg], [0], [200], [diastolic blood pressure],
  [fgn], [mg/dL], [0], [1500], [fibrinogen],
  [fio2], [%], [21], [100], [fraction of inspired oxygen],
  [glu], [mg/dL], [0], [1000], [glucose],
  [hgb], [g/dL], [4], [18], [hemoglobin],
  [hr], [bpm, /min], [0], [300], [heart rate],
  [inr_pt], [-], [-], [-], [prothrombin time/international normalized ratio],
  [k], [mEq/L, mmol/l], [0], [10], [potassium],
  [lact], [mmol/L], [0], [50], [lactate],
  [lymph], [%], [0], [100], [lymphocytes],
  [map], [mmHg, mm Hg], [0], [250], [mean arterial pressure],
  [mch], [pg], [0], [-], [mean cell hemoglobin],
  [mchc], [%], [20], [50], [mean corpuscular hemoglobin concentration],
  [mcv], [fL], [50], [150], [mean corpuscular volume],
  [methb], [%], [0], [100], [methemoglobin],
  [mg], [mg/dL], [0.5], [5], [magnesium],
  [na], [mEq/L, mmol/l], [110], [165], [sodium],
  [neut], [%], [0], [100], [neutrophils],
  [o2sat], [%, % Sat.], [50], [100], [oxygen saturation],
  [pco2], [mmHg, mm Hg], [10], [150], [CO2 partial pressure],
  [ph], [-], [6.8], [8], [pH of blood],
  [phos], [mg/dL], [0], [40], [phosphate],
  [plt], [K/uL, G/l], [5], [1200], [platelet count],
  [po2], [mmHg, mm Hg], [40], [600], [O2 partial pressure],
  [ptt], [sec], [0], [-], [partial thromboplastin time],
  [resp], [insp/min, /min], [0], [120], [respiratory rate],
  [sbp], [mmHg, mm Hg], [0], [300], [systolic blood pressure],
  [temp], [C, °C], [32], [42], [temperature],
  [tnt], [ng/mL], [0], [-], [troponin t],
  [urine], [mL], [0], [2000], [urine output],
  [wbc], [K/uL, G/l], [0], [-], [white blood cell count],
  [age], [years], [0], [100], [patient age],
  [sex], [-], [-], [-], [patient sex],
  [height], [cm], [10], [230], [patient height],
  [weight], [kg], [1], [500], [patient weight],
),
  caption: [Dynamic input features for the prediction task.]
) <tab:concepts>
