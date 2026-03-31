#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.1"
#import "@preview/numbly:0.1.0": numbly

#import "figures/poster_fig.typ": *
#import "figures/on_vs_off.typ": *
#import "figures/modules.typ": *
#import "figures/fsq.typ": *
#import "figures/helper.typ": *
#import "figures/cohort.typ": cohort_fig
#import "figures/auto_encoder.typ": ae_fig
#import "thesis_env.typ": aurocp, auprcp

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#show figure.caption: it => [
  #it.body 
]
// #let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)
#show: university-theme.with(
  aspect-ratio: "16-9",
  // align: horizon,
  // align:left,
  // config-common(handout: true),
  // config-common(frozen-counters: (theorem-counter,)),  // freeze theorem counter for animation
  config-info(
    title: text(size: 36pt)[\ \ Combining Machine-Learning and Dynamic Network Models for Sepsis Prediction],
    subtitle: text(size: 20pt)[Juri Backes$""^(1,3)$, Artyom Tsanda$""^(1,2)$, Tobias Knopp$""^(1,2)$, Wolfgang Renz$""^(3)$, and Eckehard Schöll$""^(4)$],
    author: [#v(-2em)],
    date: datetime.today(),
    institution: [$""^1$TU Hamburg, $""^2$UKE Hamburg, $""^3$HAW Hamburg, $""^4$TU Berlin],
    logo: scale(55%, [#v(-1.5em)#image("images/logo_tuhh_uke.svg")]),

  ),
    header-right: image("images/logo_tuhh_uke.svg"),
    header: utils.display-current-heading(level: 1),
    // footer-a: self => text(size: 16pt)[#self.info.author],
    footer-b: self => [Combining Machine-Learning and Dynamic Network Models to Improve Sepsis Prediction],
)
// #set text(size: 20pt)

// Bio pics + pheno
// No sep3 def (in app) aber A*I
// X Into DNM (simplified? Volles aber gekürzt) 
// X gif phase in polar + hist of freqs 
// t-1 gifs + ensembles 
// ensembles + space
// Idea (full dnm step by step) left pic, right losses
// Quant loss progression? heat + heat_space +AUs
// Qual trajs
// Limitations and what to do against them (dnm valid? Who knows)

#set figure.caption(position: top)

// #{set text(size: 20pt)
// title-slide()}
#title-slide()

#set text(size: 16pt)
#set align(left)
#set list(indent: 1em)
#set table(
  // Increase the table cell's padding
  inset: 10pt, // default is 5pt
)

// #show heading.where(level: 1): set text(18pt)


// #show heading: set text(30pt)

#let dnm-short = text()[*Functional modeling* (adaptive cell communication)
  - Interaction of parenchymal (organ) and immune cells via cytokines\
  *Two-layer dynamic network model*
  - Adaptive phase oscillators]
= Sepsis and the Dynamic Network Model
#pagebreak()
#text()[]
#pagebreak()

// #v(3em)
// #dnm-short
// #grid(
// columns: 2,
// {

// v(20pt)
// scale(80%, create-poster-figure(
//   show-encoder: false,
//   show-predictor:false,
//   show-sofa: false,
//   show-connections: false,
//   show-pkappa: false,
//   show-ikappa: false,
//   show-immune: false,
//   show-sigma: false,
// ))},
// {
// v(30pt)
// $
//   cmb(dot(phi)^1_i) =& -1/N sum^N_(j=1) lr({ cmw((1 + kappa^1_(i j)))sin(cmb(phi^1_i) - cmb(phi^1_j) -0.28pi) }) cmw(- sigma sin(phi^1_i - phi^2_i)) \
//   cmw(dot(kappa)^1_(i j) &= -0.03 (kappa^1_(i j) + sin(phi^1_i - phi^1_j - beta)) \
//   dot(phi)^2_i =& - 1/N sum^N_(j=1) lr({kappa^2_(i j)sin(phi^2_i - phi^2_j -0.28pi) }) - sigma sin(phi^2_i - phi^1_i) \
//   dot(kappa)^2_(i j) &= -0.3 (kappa^2_(i j) + sin(phi^2_i - phi^2_j - beta)))  
// $}
// )
#pagebreak()
#v(3em)
#dnm-short
#grid(
columns: 2,
{
v(10pt)
scale(80%, create-poster-figure(
  show-encoder: false,
  show-predictor:false,
  show-sofa: false,
  show-connections: false,
  show-sigma: false,
  show-pkappa: false,
  show-ikappa: false,
))},
{
v(30pt)
$
  cmb(dot(phi)^1_i) =& -1/N sum^N_(j=1) lr({ cmw((1 + kappa^1_(i j)))sin(cmb(phi^1_i) - cmb(phi^1_j) -0.28pi) }) cmw(- sigma sin(phi^1_i - phi^2_i)) \
  cmw(dot(kappa)^1_(i j) &= -0.03 (kappa^1_(i j) + sin(phi^1_i - phi^1_j - beta))) \
  cmg(dot(phi)^2_i) =& - 1/N sum^N_(j=1) lr({cmw(kappa^2_(i j))sin(cmg(phi^2_i) - cmg(phi^2_j) -0.28pi)}) cmw(- sigma sin(phi^2_i - phi^1_i)) \
  cmw(dot(kappa)^2_(i j) &= -0.3 (kappa^2_(i j) + sin(phi^2_i - phi^2_j - beta)))  
$}
)
// #pagebreak()
// #v(3em)
// #dnm-short
// #grid(
// columns: 2,
// {
// v(10pt)
// scale(80%, create-poster-figure(
//   show-encoder: false,
//   show-predictor:false,
//   show-sofa: false,
//   show-connections: false,
//   show-sigma: false,
//   show-ikappa: false,
// ))},
// {
// v(30pt)
// $
//   cmb(dot(phi)^1_i) =& -1/N sum^N_(j=1) lr({ (1 + cmp(kappa^1_(i j)))sin(cmb(phi^1_i) - cmb(phi^1_j) -0.28pi) }) cmw(- sigma sin(phi^1_i - phi^2_i)) \
//   cmp(dot(kappa)^1_(i j)) &= -0.03 (cmp(kappa^1_(i j)) + sin(cmb(phi^1_i) - cmb(phi^1_j) - cmr(beta))) \
//   cmg(dot(phi)^2_i) =& - 1/N sum^N_(j=1) lr({cmw(kappa^2_(i j))sin(cmg(phi^2_i) - cmg(phi^2_j) -0.28pi)}) cmw(- sigma sin(phi^2_i - phi^1_i)) \
//   cmw(dot(kappa)^2_(i j) &= -0.3 (kappa^2_(i j) + sin(phi^2_i - phi^2_j - beta)))  
//   // cmpp(dot(kappa)^2_(i j)) &= -0.3 (cmpp(kappa^2_(i j)) + sin(cmg(phi^2_i) - cmg(phi^2_j) - beta))  
// $}
// )
#pagebreak()
#v(3em)
#dnm-short
#grid(
columns: 2,
{
v(10pt)
scale(80%, create-poster-figure(
  show-encoder: false,
  show-predictor:false,
  show-sofa: false,
  show-connections: false,
  show-sigma: false,
))},
{
v(30pt)
$
  cmb(dot(phi)^1_i) =& -1/N sum^N_(j=1) lr({ (1 + cmp(kappa^1_(i j)))sin(cmb(phi^1_i) - cmb(phi^1_j) -0.28pi) }) cmw(- sigma sin(phi^1_i - phi^2_i)) \
  cmp(dot(kappa)^1_(i j)) &= -0.03 (cmp(kappa^1_(i j)) + sin(cmb(phi^1_i) - cmb(phi^1_j) - cmr(beta))) \
  cmg(dot(phi)^2_i) =& - 1/N sum^N_(j=1) lr({cmpp(kappa^2_(i j))sin(cmg(phi^2_i) - cmg(phi^2_j) -0.28pi)}) cmw(- sigma sin(phi^2_i - phi^1_i)) \
  cmpp(dot(kappa)^2_(i j)) &= -0.3 (cmpp(kappa^2_(i j)) + sin(cmg(phi^2_i) - cmg(phi^2_j) - cmr(beta)))  
$}
)
#pagebreak()
#v(3em)
#dnm-short
#grid(
columns: 2,
{
v(10pt)
scale(80%, create-poster-figure(
  show-encoder: false,
  show-predictor:false,
  show-sofa: false,
  show-connections: false,
))},
{
v(30pt)
$
  cmb(dot(phi)^1_i) =& -1/N sum^N_(j=1) lr({ (1 + cmp(kappa^1_(i j)))sin(cmb(phi^1_i) - cmb(phi^1_j) -0.28pi) })  - cmo(sigma) sin(cmb(phi^1_i) - cmg(phi^2_i)) \
  cmp(dot(kappa)^1_(i j)) &= -0.03 (cmp(kappa^1_(i j)) + sin(cmb(phi^1_i) - cmb(phi^1_j) - cmred(beta))) \
  cmg(dot(phi)^2_i) =& - 1/N sum^N_(j=1) lr({cmpp(kappa^2_(i j))sin(cmg(phi^2_i) - cmg(phi^2_j) -0.28pi)}) - cmo(sigma) sin(cmg(phi^2_i) - cmb(phi^1_i)) \
  cmpp(dot(kappa)^2_(i j)) &= -0.3 (cmpp(kappa^2_(i j)) + sin(cmg(phi^2_i) - cmg(phi^2_j) - cmred(beta)))  
$}
)

#pagebreak()
#v(4em)
#figure(image("images/presentation/simulation0.png", width: 85%))

#pagebreak()
#v(11em)
#align(center, text()[Simulation Video])

#pagebreak()
#v(4em)
#figure(image("images/presentation/simulation1.png", width: 85%))

#pagebreak()
#v(1em)
#figure(image("images/presentation/phase.svg", width: 60%))

// #pagebreak()
// #v(4em)
// #grid(
//   columns: (1fr, 1fr),
//   figure(image("images/presentation/phase.svg", width: 100%)),
//   [#set text(size: 20pt)
//    #v(2em)
//     *Summary*
//     - Functional Model of critical transitions
//     - $beta $ and $ sigma$ modulate synchronization
//     #v(1em)
//     - Not yet validated on clinical data
//     *Can we utilize it to overcome black-box predictions?*
//   ]
// )


#let sep-def-not = text(20pt)[
  *Sepsis Definition (simplified)* $-> S_t$
    + Infection $-> I_t$
    + Rapid Organ Failure $-> A_t$
      - SOFA-score increase $>=$ 2
]

#pagebreak()
= Latent Dynamics Model
== Latent Dynamics Model

#grid(
  columns: (1fr, 1fr),
  [#v(3em) #scale(figure(create-oo-figure(off: false, control: false, labels: text(black)[$S_t$])), 120%) #v(3em) #sep-def-not],
  box(inset:(x:7.1em))[],
)
#grid(
  columns: (1fr, 1fr),
  [#v(3em) #scale(figure(create-oo-figure(off: false, control: false, labels: text(black)[$S_t$])), 120%) #v(3em) #sep-def-not],
  box(inset:(x:7.1em))[#set text(size: 12pt)
    #v(6em) #scale(figure(create-simple-ldm-figure(inf: false, sofa: false)), 110%)],
)
#grid(
  columns: (1fr, 1fr),
  [#v(3em)#scale(figure(create-oo-figure(off: false, control: false, labels: text(black)[$S_t$])), 120%) #v(3em) #sep-def-not],
  [#set text(size: 12pt)
    #v(6em) #scale(figure(create-simple-ldm-figure(inf: true, sofa: false)), 110%)],
)
#grid(
  columns: (1fr, 1fr),
  [#v(3em)#scale(figure(create-oo-figure(off: false, control: false, labels: text(black)[$S_t$])), 120%) #v(3em) #sep-def-not],
  [#set text(size: 12pt)
    #v(6em) #scale(figure(create-simple-ldm-figure(inf: true, sofa: true)), 110%)],
)

// #pagebreak()
// #v(8em)
// #block(inset: (x: 12em))[
//   #set text(size: 12pt)
//   #scale(fsq_fig, 200%)
// ]

= Experimental Results
#pagebreak()
#v(4em)
#figure(image("images/presentation/trajectory.svg", width: 105%))

#pagebreak()
#v(4em)
#figure(image("images/presentation/heat_space.svg", width: 100%))

// #pagebreak()
// #v(4em)
// #figure(image("images/presentation/heat.svg", width: 105%))

#pagebreak()
#v(2em)
#figure(
  table(
  columns: 3,
  align: (left, center, center),

  table.header(
    [*Model*],
    [*AUROC*],
    [*AUPRC*],
  ),

  table.cell(colspan: 3)[*YAIB*],
  table.hline(stroke: .5pt),

  [Regularized Logistic Regression],
  [77.1 $plus.minus (0.4)$],
  [4.6 $plus.minus (0.1)$],

  [Light Gradient Boosting Machine],
  [77.5 $plus.minus (0.3)$],
  [5.9 $plus.minus (0.2)$],

  [Transformer],
  [80.0 $plus.minus (0.8)$],
  [6.6 $plus.minus (0.2)$],

  [LSTM],
  [82.0 $plus.minus (0.3)$],
  [8.0 $plus.minus (0.2)$],

  [Temporal Convolutional Network],
  [82.7 $plus.minus (0.3)$],
  [8.8 $plus.minus (0.2)$],

  [GRU],
  [*83.6* $plus.minus (0.3)$],
  [*9.1* $plus.minus (0.3)$],

  table.hline(stroke: .5pt),
  table.cell(colspan: 3)[*This work*],
  table.hline(stroke: .5pt),

  [Latent Dynamics Model],
  [*84.01* $plus.minus (0.8)$],
  [*9.97* $plus.minus (0.8)$],
  ),
  caption:[Comparison against mean performance of YAIB @yaib, \ AUROC $times 100$ ($arrow.t$, higher is better) and AUPRC $times 100$ ($arrow.t$).
  ]
)


#pagebreak()
= Conclusion
== Conclusion

#v(2em)
#box(inset: (x: 2em))[#text(size: 20pt)[
// *Strengths*
- Interpretable
- Decomposition Strategy
- Competitive Performance
#v(2em)
// *Limitations and Outlook*
// - No ablation study (what parts are actually necessary?)
- Do other sepsis models exist, that work even better?

#v(1em)
#grid(
  columns: (1fr, 1fr),
  [#v(3em)
  #text(blue)[https://github.com/unartig/sepsis_osc]],
  [#figure(image("images/presentation/qrcode.svg", width: 40%))]
)
// - Backpropagate through Dynamic Network Model?
]]

#pagebreak()
= 
== 
#v(5em)
#bibliography("bibliography.bib", style:"ieee_custom.csl")

#pagebreak()
= Supplemental Material
== Supplemental Material
#v(5em)
#figure(
  image("images/sofa-sep-3-1.png", width: 100%),
  caption: [Graphical representation of the timings in the Sepsis-3 definition, taken from @ricufig.],
)

#pagebreak()
#v(3em)
*Initialization*
  - Uniform cytokine activity in the organ layer
  - Uniform initial phases
  - Cytokine activation clusters in the immune layer
#figure(image("images/presentation/init.svg", width: 100%))


#pagebreak()
#figure(image("images/presentation/ensembles.svg", width: 60%))

#pagebreak()
#v(4em)
#figure(image("images/presentation/snapshots.svg", width: 80%))

#pagebreak()
#[#set text(size: 12pt)
#v(5em)
#align(center, scale(120%, figure(ldm_fig)))]

#pagebreak()
#[#set text(size: 12pt)
#v(12em)
#align(center, scale(140%, figure(sofa_fig)))]

#pagebreak()
#[#set text(size: 12pt)
#v(12em)
#align(center, scale(140%, figure(inf_fig)))]

#pagebreak()
#[#set text(size: 12pt)
#figure(ae_fig)]

#pagebreak()
#[#set text(size: 12pt)
#v(7em)
#figure(cohort_fig)]

#pagebreak()
#v(6em)
#figure(table(
  columns: 4,
  align: (left, center, center, center),
  table.header(
    [*Characteristic*],
    [*All patients*],
    [*SEP-3 positive*],
    [*SEP-3 negative*]
  ),
  table.cell(colspan: 4)[*Demographics*],
  table.hline(stroke:.5pt),
  
  [N],
  [63425 (100.0)],
  [3320 (5.2)],
  [60105 (94.8)],
  
  [Male],
  [35170 (55.5)],
  [1881 (56.7)],
  [33289 (55.4)],  

  [Age at admission],
  [65.0 (53.0–76.0)],
  [65.0 (54.0–76.0)],
  [65.0 (53.0–76.0)],

  [Weight at admission],
  [77.6 (65.1–92.3)],
  [77.6 (65.6–94.0)],
  [77.6 (65.0–92.2)],
  )
)

#pagebreak()
#v(6em)
#figure(table(
  columns: 4,
  align: (left, center, center, center),
  table.header(
    [*Characteristic*],
    [*All patients*],
    [*SEP-3 positive*],
    [*SEP-3 negative*]
  ),
  table.hline(stroke:.5pt),
  table.cell(colspan: 4)[*Clinical Outcomes*],
  table.hline(stroke:.5pt),

  [SOFA median],
  [3.0 (1.0–5.0)],
  [3.0 (1.0–5.0)],
  [3.0 (1.0–5.0)],

  [SOFA max],
  [4.0 (2.0–6.0)],
  [5.0 (4.0–8.0)],
  [4.0 (2.0–6.0)],

  [Hospital LOS hours],
  [157.7 (92.8–268.9)],
  [335.1 (194.2–548.6)],
  [150.3 (90.9–256.0)],

  [Hospital Mortality],
  [4828 (7.6)],
  [879 (26.5)],
  [3949 (6.6)],

  [Sepsis-3 onset time],
  [-],
  [13.0 (8.0–34.0)],
  [-],
  )
)

#pagebreak()
#v(6em)
#figure(table(
  columns: 4,
  align: (left, center, center, center),
  table.header(
    [*Characteristic*],
    [*All patients*],
    [*SEP-3 positive*],
    [*SEP-3 negative*]
  ),
  table.hline(stroke:.5pt),
  table.cell(colspan: 4)[*Ethnicity*],
  table.hline(stroke:.5pt),

  [White],
  [40364 (63.6)],
  [2087 (62.9)],
  [38277 (63.7)],

  [Black],
  [5809 (9.2)],
  [262 (7.9)],
  [5547 (9.2)],

  [Asian],
  [721 (1.1)],
  [42 (1.3)],
  [679 (1.1)],

  [Hispanic],
  [630 (1.0)],
  [32 (1.0)],
  [598 (1.0)],

  [Other/Unknown],
  [14924 (23.5)],
  [897 (27.0)],
  [14027 (23.3)],
  )
  
)

#pagebreak()
#v(6em)
#figure(table(
  columns: 4,
  align: (left, center, center, center),
  table.header(
    [*Characteristic*],
    [*All patients*],
    [*SEP-3 positive*],
    [*SEP-3 negative*]
  ),
  table.hline(stroke:.5pt),
  table.cell(colspan: 4)[*Admission Type*],
  table.hline(stroke:.5pt),

  [Medical],
  [45009 (71.0)],
  [2817 (84.8)],
  [42192 (70.2)],

  [Surgical],
  [2239 (3.5)],
  [45 (1.4)],
  [2194 (3.7)],

  [Other/Unknown],
  [15200 (24.0)],
  [458 (13.8)],
  [14742 (24.5)],
  )
)

#pagebreak()
#v(5em)
#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    [*Loss*], [*Type*], [*Purpose*], [*Supervises*],
    [$cal(L)_"sepsis"$], [BCE], [Primary sepsis prediction], [$f_theta_f, g^e_theta^e_g, g^r_theta^r_g$],
    [$cal(L)_"sofa"$], [Weighted MSE], [SOFA estimation], [$g^e_theta^e_g, g^r_theta^r_g$],
    [$cal(L)_"inf"$], [BCE], [Infection indicator], [$f_theta_f$],
    [$cal(L)_"spread"$], [Covariance], [Latent diversity], [$g^e_theta^e_g, g^r_theta^r_g$],
    [$cal(L)_"boundary"$], [Positional], [Latent Space], [$g^e_theta^e_g, g^r_theta^r_g$],
    [$cal(L)_"dec"$], [MSE], [Latent semantics], [$d_theta_d$, ($g^e_theta^e_g, g^r_theta^r_g$)],
  ),
  caption: [Overview of loss components in the LDM training objective.]
)

#pagebreak()
#v(4em)
#figure(
  image("images/losses.svg", width: 90%),
  caption: [Progression of training and validation losses],
)


#pagebreak()
#v(3em)
#grid(
columns: (0.7fr, 1fr, 0.2fr),
[$
  delta(tilde(S)_t) = II (tilde(S)_t > tau)
$
$
  "TPR" = "TP"/("TP" + "FN")
$
$
  "FPR" = "FP"/("TP" + "FN")
$
$
  P = "TP" / ("TP" + "FP")
$
$
  R = "TP" / ("TP" + "FN")
$],
[
#figure(
  image("/images/presentation/areas.svg", width: 140%),
  caption:[Receiver Operating Characteristic and Precision-Recall Curve.],
) <fig:areas>

]
)
