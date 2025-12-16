#import "../thesis_env.typ": *

= Appendix
== SOFA - Details <a:sofa>
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

  caption: flex-caption(short: [TODO], long: [#todo[]])
) <tab:sofa>

== DNM as Lie Formulation <a:lie>
