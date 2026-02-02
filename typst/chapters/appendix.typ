#import "../thesis_env.typ": *

= Appendix
== SOFA - Details <a:sofa>
#figure(
  table(
    columns: (1fr, auto, auto, auto, auto, auto),
    inset: 10pt,
    align: (left, left, center, center, center, center),
    table.header([*Category*], [*Indicator*], [*1*], [*2*], [*3*], [*4*]),
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

  caption: flex-caption(short: [#acl("SOFA")-score component definitions.], long: [#acr("SOFA")-score component definitions @SOFAscore.])
) <tab:sofa>

== DNM as Lie Formulation <a:lie>

#pagebreak()
== Input Concepts
The following two tables list the static and dynamic input features used by the #acr("YAIB") framework @yaib.
The column "ricu-name" refers to the R package `ricu` which #acr("YAIB") is built on top of @ricu.

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
#figure(
table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, left, left, right, left),
  stroke: .5pt,

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
),
  caption: [Dynamic input features for the prediction task.]
) <tab:concepts>

#pagebreak()
== Latent Dynamic Model Architecture Parameters <a:paramcount>
#figure(grid(
columns: (1fr,5pt, 1fr),
  table(
    columns: 3,
    align: (left, center, right),
    stroke: 0.5pt,
    [*Name*], [*Shape*], [*Count*],
    table.cell(colspan: 3, align: left, [*Infection Module*]),
    [`GRU-Cell weight_ih`], [(96, 104)], [9,984],
    [`GRU-Cell weight_hh`], [(96, 32)], [3,072],
    [`GRU-Cell bias`], [(96,)], [96],
    [`GRU-Cell bias_n`], [(32,)], [32],
    [`h0`], [(32,)], [32],
    [`Projection weight`], [(1, 32)], [32],
    [`Projection bias`], [(1,)], [1],
    table.cell(colspan: 2, align: right, [*Total:*]), [*13,249*],
    table.cell(colspan: 3, align: left, [*Decoder Module*]),
    [`linear1 weight`], [(16, 2)], [32],
    [`linear1 bias`], [(16,)], [16],
    [`norm1 weight`], [(16,)], [16],
    [`norm1 bias`], [(16,)], [16],
    [`linear2 weight`], [(32, 16)], [512],
    [`linear2 bias`], [(32,)], [32],
    [`norm2 weight`], [(32,)], [32],
    [`norm2 bias`], [(32,)], [32],
    [`linear3 weight`], [(32, 32)], [1,024],
    [`linear3 bias`], [(32,)], [32],
    [`norm3 weight`], [(32,)], [32],
    [`norm3 bias`], [(32,)], [32],
    [`linear4 weight`], [(52, 32)], [1,664],
    [`linear4 bias`], [(52,)], [52],
    table.cell(colspan: 2, align: right, [*Total:*]), [*3,524*],
    table.cell(colspan: 3, align: left, [*General*]),
    [`lookup_temp`], [(1,)], [1],
    [`causal_decay`], [(1,)], [1],
    [`detection_threshold`], [(1,)], [1],
    [`detection_sharpness`], [(1,)], [1],
    table.cell(colspan: 2, align: right, [*Total:*]), [*4*],),
    [],

 
  table(
    columns: 3,
    align: (left, center, right),
    stroke: 0.5pt,
    [*Name*], [*Shape*], [*Count*],
    table.cell(colspan: 3, align: left, [*SOFA Module (Encoder)*]),
    [`gating weight`], [(52, 52)], [2,704],
    [`norm1 weight`], [(104,)], [104],
    [`norm1 bias`], [(104,)], [104],
    [`linear1 weight`], [(64, 104)], [6,656],
    [`linear1 bias`], [(64,)], [64],
    [`norm2 weight`], [(64,)], [64],
    [`norm2 bias`], [(64,)], [64],
    [`linear2 weight`], [(64, 64)], [4,096],
    [`linear2 bias`], [(64,)], [64],
    [`norm3 weight`], [(64,)], [64],
    [`norm3 bias`], [(64,)], [64],
    [`linear3 weight`], [(64, 64)], [4,096],
    [`linear3 bias`], [(64,)], [64],
    [`linear4 weight`], [(16, 64)], [1,024],
    [`linear4 bias`], [(16,)], [16],
    [`z weight`], [(2, 16)], [32],
    [`z bias`], [(2,)], [2],
    [`h weight`], [(4, 16)], [64],
    [`h bias`], [(4,)], [4],
    table.cell(colspan: 2, align: right, [*Total:*]), [*19,350*],
    table.cell(colspan: 3, align: left, [*SOFA Module (RNN)*]),
    [`GRU-Cell weight_ih`], [(12, 106)], [1,272],
    [`GRU-Cell weight_hh`], [(12, 4)], [48],
    [`GRU-Cell bias`], [(12,)], [12],
    [`GRU-Cell bias_n`], [(4,)], [4],
    [`Proj weight`], [(2, 4)], [8],
    table.cell(colspan: 2, align: right, [*Total:*]), [*1,344*],)
),

    caption: [Detailed parameter count of the #acl("LDM") modules.]
) <tab:paramcount>
