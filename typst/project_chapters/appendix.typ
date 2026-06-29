#import "../thesis_env.typ": *

= Appendix
== SOFA - Details <a:sofa>
#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, center, center, center, center),
    table.header([*Category*], [*Indicator*], [*1*], [*2*], [*3*], [*4*]),
    [Respiration], [$"PaO"_2$/$"FiO"_2$ [mmHg]], [< 400], [< 300], [< 200], [< 100],
    [], [Mechanical Ventilation], [], [], [yes], [yes],

    table.hline(stroke: .5pt),
    [Coagulation], [Platelets [$times 10^3/"mm"^3$]], [< 150], [< 100], [< 50], [< 20],

    table.hline(stroke: .5pt),
    [Liver], [Bilirubin [$"mg"/"dl"$]], [1.2-1.9], [2.0-5.9], [6.0-11.9], [> 12.0],

    table.hline(stroke: .5pt),
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
    table.hline(stroke: .5pt),
    [Central Nervous System], [Glasgow Coma Score], [13-14], [10-12], [6-9], [< 6],
    table.hline(stroke: .5pt),

    [Renal], [Creatinine [$"mg"/"dl"$]], [1.2-1.9], [2.0-3.4], [3.5-4.9], [> 5.0],

    [], [or Urine Output [$"ml"/"day"$]], [], [], [< 500], [< 200],
  ),

  caption: flex-caption(
    short: [Components of the #acs("SOFA")-score definition.],
    long: [Components of the #acr("SOFA")-score definition @SOFAscore.],
  ),
) <tab:sofa>

#pagebreak()
== Features <a:feat>
The following two tables list the static and dynamic input features used by the #acr("YAIB") framework @yaib.
The column "ricu-name" refers to the R package `ricu` which #acr("YAIB") is built on top of @ricu.

#figure(
  table(
    // columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, left, right, left, left),
    columns: 7,
    [ricu-name], [unit], [min], [max], [description], [MIMIC-IV \ % missing], [eICU \ % missing],
    [age], [years], [0], [100], [patient age], [0.00], [0.00],
    [sex], [], [], [], [patient sex], [0.00], [0.00],
    [height], [cm], [10], [230], [patient height], [48.70], [1.01],
    [weight], [kg], [1], [500], [patient weight], [7.19], [4.72],
  ),
  caption: [Static input features for the prediction task],
) <tab:stat>
#figure(
  {
    show table.cell: set text(size: 10pt)
    table(
      columns: 7,
      align: (left, left, left, right, left, left),
      [ricu-name], [unit], [min], [max], [description], [MIMIC-IV \ % missing], [eICU \ % missing],
      [name], [], [], [], [], [], [],

      [alb], [g/dL], [0], [6], [albumin], [99.16], [98.19],
      [alp], [IU/L, U/l], [0], [], [alkaline phosphatase], [98.46], [98.47],
      [alt], [IU/L, U/l], [0], [], [alanine aminotransferase], [98.44], [98.46],
      [ast], [IU/L, U/l], [0], [], [aspartate aminotransferase], [98.42], [98.43],
      [be], [mEq/L, mmol/l], [-25], [25], [base excess], [95.20], [98.24],
      [bicar], [mEq/L, mmol/l], [5], [50], [bicarbonate], [93.28], [95.06],
      [bili], [mg/dL], [0], [100], [total bilirubin], [98.43], [98.45],
      [bili_dir], [mg/dL], [0], [50], [bilirubin direct], [99.87], [99.69],
      [bnd], [%], [], [], [band form neutrophils], [99.79], [99.75],
      [bun], [mg/dL], [0], [200], [blood urea nitrogen], [93.23], [94.79],
      [ca], [mg/dL], [4], [20], [calcium], [94.03], [94.98],
      [cai], [mmol/L], [0.5], [2], [calcium ionized], [96.93], [99.99],
      [ck], [IU/L, U/l], [0], [], [creatine kinase], [99.08], [99.46],
      [ckmb], [ng/mL], [0], [], [creatine kinase MB], [99.07], [99.62],
      [cl], [mEq/L, mmol/l], [80], [130], [chloride], [92.90], [94.74],
      [crea], [mg/dL], [0], [15], [creatinine], [93.21], [94.77],
      [crp], [mg/L], [0], [], [C-reactive protein], [99.95], [99.96],
      [dbp], [mmHg, mm Hg], [0], [200], [diastolic blood pressure], [13.33], [14.42],
      [fgn], [mg/dL], [0], [1500], [fibrinogen], [99.27], [99.77],
      [fio2], [%], [21], [100], [fraction of inspired oxygen], [91.32], [86.86],
      [glu], [mg/dL], [0], [1000], [glucose], [91.47], [82.14],
      [hgb], [g/dL], [4], [18], [hemoglobin], [93.53], [94.63],
      [hr], [bpm, /min], [0], [300], [heart rate], [7.29], [5.30],
      [inr_pt], [], [], [], [prothrombin time/international normalized ratio], [95.59], [98.45],
      [k], [mEq/L, mmol/l], [0], [10], [potassium], [92.73], [93.80],
      [lact], [mmol/L], [0], [50], [lactate], [96.81], [99.06],
      [lymph], [%], [0], [100], [lymphocytes], [99.15], [97.72],
      [map], [mmHg, mm Hg], [0], [250], [mean arterial pressure], [11.98], [14.31],
      [mch], [pg], [0], [], [mean cell hemoglobin], [93.67], [95.74],
      [mchc], [%], [20], [50], [mean corpuscular hemoglobin concentration], [93.67], [95.48],
      [mcv], [fL], [50], [150], [mean corpuscular volume], [93.67], [95.47],
      [methb], [%], [0], [100], [methemoglobin], [99.99], [99.52],
      [mg], [mg/dL], [0.5], [5], [magnesium], [93.52], [96.99],
      [na], [mEq/L, mmol/l], [110], [165], [sodium], [92.83], [94.30],
      [neut], [%], [0], [100], [neutrophils], [99.15], [98.01],
      [o2sat], [%, % Sat.], [50], [100], [oxygen saturation], [9.36], [11.96],
      [pco2], [mmHg, mm Hg], [10], [150], [CO2 partial pressure], [95.20], [97.89],
      [ph], [], [6.8], [8], [pH of blood], [94.51], [97.92],
      [phos], [mg/dL], [0], [40], [phosphate], [93.99], [97.90],
      [plt], [K/uL, G/l], [5], [1200], [platelet count], [93.55], [95.43],
      [po2], [mmHg, mm Hg], [40], [600], [O2 partial pressure], [95.51], [97.91],
      [ptt], [sec], [0], [], [partial thromboplastin time], [95.30], [98.74],
      [resp], [insp/min, /min], [0], [120], [respiratory rate], [8.27], [10.23],
      [sbp], [mmHg, mm Hg], [0], [300], [systolic blood pressure], [13.31], [14.42],
      [temp], [C, °C], [32], [42], [temperature], [73.12], [72.61],
      [tnt], [ng/mL], [0], [], [troponin t], [99.23], [99.81],
      [urine], [mL], [0], [2000], [urine output], [48.45], [74.18],
      [wbc], [K/uL, G/l], [0], [], [white blood cell count], [93.66], [95.46],
    )
  },
  caption: [Dynamic input features for the prediction task.],
) <tab:dyn>


#pagebreak()
== Cohort Statistics
#figure(
  {
    show table.cell: set text(size: 10pt)
    table(
      columns: 4,
      align: (left, right, right, right),
      [], [All patients], [SEP-3 positive], [SEP-3 negative],
      [N], [63425 (100.0%)], [3320 (5.2%)], [60105 (94.8%)],
      [Male n (%)], [35170 (55.5%)], [1881 (56.7%)], [33289 (55.4%)],
      [Age at admission, median (IQR)], [65.0 (53.0–76.0)], [65.0 (54.0–76.0)], [65.0 (53.0–76.0)],
      [Weight at admission, median (IQR)], [77.6 (65.1–92.3)], [77.6 (65.6–94.0)], [77.6 (65.0–92.2)],
      [SOFA median, median (IQR)], [3.0 (1.0–5.0)], [3.0 (1.0–5.0)], [3.0 (1.0–5.0)],
      [SOFA max, median (IQR)], [4.0 (2.0–6.0)], [5.0 (4.0–8.0)], [4.0 (2.0–6.0)],
      [hospital LOS hours, median (IQR)], [157.7 (92.8–268.9)], [335.1 (194.2–548.6)], [150.3 (90.9–256.0)],
      [Hospital Mortality, (%)], [4828 (7.6%)], [879 (26.5%)], [3949 (6.6%)],
      [SEP-3 onset time, median (IQR)], [--], [13.0 (8.0–34.0)], [--],
      table.cell(colspan: 4, align: left, [*Ethnicity*]),
      [White, (%)], [40364 (63.6%)], [2087 (62.9%)], [38277 (63.7%)],
      [Black, (%)], [5809 (9.2%)], [262 (7.9%)], [5547 (9.2%)],
      [Asian, (%)], [721 (1.1%)], [42 (1.3%)], [679 (1.1%)],
      [Hispanic, (%)], [630 (1.0%)], [32 (1.0%)], [598 (1.0%)],
      [Other/Unknown, (%)], [14924 (23.5%)], [897 (27.0%)], [14027 (23.3%)],
      table.cell(colspan: 4, align: left, [*Admission Type*]),
      [Medical, (%)], [45009 (71.0%)], [2817 (84.8%)], [42192 (70.2%)],
      [Surgical, (%)], [2239 (3.5%)], [45 (1.4%)], [2194 (3.7%)],
      [Other/Unknown, (%)], [15200 (24.0%)], [458 (13.8%)], [14742 (24.5%)],
    )
  },
  caption: [Cohort Characteristics of the MIMIC-IV cohort.],
) <tab:mimic>

#pagebreak()
#figure(
  {
    show table.cell: set text(size: 10pt)
    table(
      columns: 4,
      align: (left, right, right, right),
      [], [All patients], [SEP-3 positive], [SEP-3 negative],
      [N], [123412 (100.0%)], [5639 (4.6%)], [117773 (95.4%)],
      [Male n (%)], [66934 (54.2%)], [3099 (55.0%)], [63835 (54.2%)],
      [Age at admission, median (IQR)], [65.0 (53.0–76.0)], [66.0 (55.0–76.0)], [65.0 (52.0–76.0)],
      [Weight at admission, median (IQR)], [80.2 (66.6–97.0)], [79.8 (65.4–97.5)], [80.2 (66.7–97.0)],
      [SOFA median, median (IQR)], [2.0 (1.0–4.5)], [3.0 (1.0–5.0)], [2.0 (1.0–4.0)],
      [SOFA max, median (IQR)], [3.0 (1.0–6.0)], [5.0 (3.0–8.0)], [3.0 (1.0–6.0)],
      [hospital LOS hours, median (IQR)], [128.9 (69.6–236.0)], [242.3 (144.2–408.9)], [125.0 (68.0–226.0)],
      [Hospital Mortality, (%)], [8313 (6.7%)], [885 (15.7%)], [7428 (6.3%)],
      [SEP-3 onset time, median (IQR)], [--], [16.0 (10.0–42.0)], [--],
      table.cell(colspan: 4, align: left, [*Ethnicity*]),
      [White, (%)], [94808 (76.8%)], [4480 (79.4%)], [90328 (76.7%)],
      [Black, (%)], [13926 (11.3%)], [580 (10.3%)], [13346 (11.3%)],
      [Hispanic, (%)], [4963 (4.0%)], [110 (2.0%)], [4853 (4.1%)],
      [Asian, (%)], [1675 (1.4%)], [72 (1.3%)], [1603 (1.4%)],
      [Other/Unknown, (%)], [8040 (6.5%)], [397 (7.0%)], [7643 (6.5%)],
      table.cell(colspan: 4, align: left, [*Admission*]),
      [Medical, (%)], [49995 (40.5%)], [2337 (41.4%)], [47658 (40.5%)],
      [Surgical, (%)], [29654 (24.0%)], [897 (15.9%)], [28757 (24.4%)],
      [Other/Unknown, (%)], [43763 (35.5%)], [2405 (42.6%)], [41358 (35.1%)],
    )
  },
  caption: [Cohort Characteristics of the eICU cohort.],
) <tab:eicu>

== Model Variations
=== MLP
#figure(
  image("../images/project/mlp.png", width: 40%),
  caption: flex-caption(long:[#acs("MLP") structure of the mlp scenario, with each linear layer having a hidden dimension of 32.], short: [#acs("MLP") structure of the mlp scenario.])
) <fig:arch-mlp>
