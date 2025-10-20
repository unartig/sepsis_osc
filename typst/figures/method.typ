#import "@preview/cetz:0.4.2": canvas, draw
#import draw: bezier, circle, content, line
#import "helper.typ": *

#set page(width: auto, height: auto, margin: 8pt, fill: none)
#set text(font: "poppins")

#let node-radius = 0.4
#let enc_offset = 0.5
#let for_yoffset = 2.8
#let for_xoffset = 7
#let ode_scale = 0.3
#let ode_yoffset = 2
#let ode_xoffset = 16
#let sofa_xoffset = 9
#let title_offset = 4.5
#let l1pos = (ode_xoffset, 1.2 + ode_yoffset)
#let l2pos = (ode_xoffset, -1.2 + ode_yoffset)
#let rng1 = gen-rng-f(123)
#let rng2 = gen-rng-f(3124)
#let nc = rgb("#eee")
#let arrow-style = (
  stroke: .5pt + black,
  mark: (
    start: "stealth",
    end: "stealth",
    fill: fuchsia,
    scale: .4,
    offset: 0.03,
  ),
)

#let mred(x) = text(fill: red, $#x$)
#let emark = (end: "stealth", fill: black, scale: .4, offset: 0.03)
#let bmark = (
  start: "stealth",
  end: "stealth",
  fill: black,
  scale: .4,
  offset: 0.03,
)



#let method_fig = canvas({
  let layers_enc = (
    // (x-pos, neuron-count, fill-color, label, y-offset)
    (
      0,
      5,
      nc,
      ($mu_1$, " ... ", " ... ", " ... ", $mu_n$),
      3 + enc_offset,
      none,
    ),
    (
      2,
      4,
      nc,
      (" ... ", " ... ", " ... ", " ... "),
      2.7 + enc_offset,
      none,
    ),
    (
      4,
      3,
      nc,
      (
        $cmalpha(alpha_0)$,
        $cmbeta(beta_0)$,
        $cmsigma(sigma_0)$,
      ),
      2.3 + enc_offset,
      none,
    ), // Latent layer
  )
  content(
    (layers_enc.at(1).at(0), title_offset),
    align(
      text("Encoder\n" + $cal(N)^"Enc"_(theta_1)$, size: 16pt),
      center,
    ),
    anchor: "south",
    padding: .3em,
  )

  for idx in range(layers_enc.len() - 1) {
    let (x1, n1, _, _, y1, _) = layers_enc.at(idx)
    let (x2, n2, _, _, y2, _) = layers_enc.at(idx + 1)
    connect-layers(x1, n1, x2, n2, y1, y2)
  }

  // Layer labels
  content((layers_enc.at(0).at(0), layers_enc.at(0).at(-2) + .7), align(
    center,
  )[Input Layer])
  content((layers_enc.at(2).at(0), layers_enc.at(0).at(-2) + .2), align(
    center,
  )[Latent\ Representation])

  draw-neurons(layers_enc)


  let layers_for = (
    (
      0 + for_xoffset,
      3,
      nc,
      (
        $cmalpha(alpha_t)$,
        $cmbeta(beta_t)$,
        $cmsigma(sigma_t)$,
      ),
      0 + for_yoffset,
      none,
    ),
    (
      1.5 + for_xoffset,
      3,
      nc,
      (" ... ", " ... ", " ... "),
      0 + for_yoffset,
      none,
    ),
    (
      3 + for_xoffset,
      3,
      nc,
      (
        $cmalpha(alpha_(t+1))$,
        $cmbeta(beta_(t+1))$,
        $cmsigma(sigma_(t+1))$,
      ),
      0 + for_yoffset,
      true,
    ),
  )
  for idx in range(layers_for.len() - 1) {
    let (x1, n1, _, _, y1, _) = layers_for.at(idx)
    let (x2, n2, _, _, y2, _) = layers_for.at(idx + 1)
    connect-layers(x1, n1, x2, n2, y1, y2)
  }
  draw-neurons(layers_for)
  content(
    (layers_for.at(1).at(0), title_offset),
    align(
      text("Latent Predictor\n" + $cal(N)^"Pred"_(theta_2)$, size: 16pt),
      center,
    ),
    anchor: "south",
    padding: .3em,
  )
  line(
    (layers_enc.at(2).at(0) + .5, for_yoffset - .8),
    (layers_for.at(0).at(0) - .5, ode_yoffset),
    mark: emark,
  )
  line(
    (layers_for.at(2).at(0) + .5, for_yoffset - .8),
    (ode_xoffset - 3, ode_yoffset),
    mark: emark,
    name: "param",
  )
  content("param", [Parametrize], anchor: "south", padding: .3em)
  let midlt = midp(
    (layers_enc.at(2).at(0) + .5, for_yoffset - .8),
    (layers_for.at(0).at(0) - .5, ode_yoffset),
  )
  let midparam = midp(
    //start
    (layers_for.at(2).at(0) + .5, for_yoffset - .8),
    (ode_xoffset - 3, ode_yoffset),
  )
  let midfor = (layers_for.at(1).at(0), for_yoffset - 2.5)
  line(
    midfor,
    (midfor, "-|", midlt), //mid
    midlt, //end
    mark: emark,
  )
  line(
    midparam,
    (midparam, "|-", midfor), //mid
    midfor, //end
  )
  content(midfor, [Auto regressive], anchor: "north", padding: .3em)

  let p1 = cadd(l1pos, (2.5 * ode_scale, 2.7 * ode_scale))
  let p2 = cadd(l1pos, (8.0 * ode_scale, 0.5 * ode_scale))
  let p3 = cadd(l1pos, (4.0 * ode_scale, -2.2 * ode_scale))
  let p4 = cadd(l1pos, (-4.0 * ode_scale, -2.2 * ode_scale))
  let p5 = cadd(l1pos, (-8.0 * ode_scale, 0.5 * ode_scale))
  let p6 = cadd(l1pos, (-2.5 * ode_scale, 2.7 * ode_scale))
  content(
    p1,
    text("Parenchymal Layer Cells", stroke: rgb("#add8e6") + .01pt),
    anchor: "west",
    padding: 2em,
  )
  let layer1 = (p1, p2, p3, p4, p5, p6)
  for pcoord in layer1 {
    neuron(pcoord, fill: blue, label: "   ", s: none, st: true)
  }
  let layer1 = (p1, p2, p3, p4, p5, p6)
  lines-between(layer1, rng1, node-radius: node-radius)

  let i1 = cadd(l2pos, (2.5 * ode_scale, 2.7 * ode_scale))
  let i2 = cadd(l2pos, (8.0 * ode_scale, 0.5 * ode_scale))
  let i3 = cadd(l2pos, (4.0 * ode_scale, -2.2 * ode_scale))
  let i4 = cadd(l2pos, (-4.0 * ode_scale, -2.2 * ode_scale))
  let i5 = cadd(l2pos, (-8.0 * ode_scale, 0.5 * ode_scale))
  let i6 = cadd(l2pos, (-2.5 * ode_scale, 2.7 * ode_scale))
  content(
    i3,
    text("Immune Layer Cells", stroke: olive + .01pt),
    anchor: "west",
    padding: 2em,
  )
  let layer2 = (i1, i2, i3, i4, i5, i6)
  for pcoord in layer2 {
    neuron(pcoord, fill: olive, label: "   ", s: none, st: true)
  }
  lines-between(layer2, rng2, node-radius: node-radius)
  content(
    (midp(p1, p6).at(0), title_offset + 0.3),
    align(
      text("Dynamic Network Model (DNM)\n", size: 16pt)
        + text("eqs. (1)-(4)", size: 10pt),
      center,
    ),
    anchor: "south",
    padding: .3em,
  )

  dashed(p1, i1, node-radius: node-radius)
  dashed(p2, i2, node-radius: node-radius)
  dashed(p3, i3, node-radius: node-radius)
  dashed(p4, i4, node-radius: node-radius)
  dashed(p5, i5, name: "dash", node-radius: node-radius)
  dashed(p6, i6, node-radius: node-radius)
  content(
    "dash",
    text($cmsigma(sigma)$, size: 12pt),
    anchor: "south-west",
    padding: 0.7em,
  )
  content(
    p5,
    text($phi^1_j(cmalpha(alpha))$, size: 12pt),
    anchor: "south-east",
    padding: 0.4em,
  )
  // neuron(p5, fill: rgb(fuchsia), label: text(" ", size: 2pt))
  content(
    midp(p5, p6),
    text($kappa^1_(i j (cmbeta(beta)))$, size: 12pt),
    anchor: "south",
    padding: 0.5em,
  )
  content(
    i5,
    text($phi^2_j(cmalpha(alpha))$, size: 12pt),
    anchor: "north-east",
    padding: 0.3em,
  )
  content(
    midp(i3, i4),
    text($kappa^2_(i j (cmbeta(beta)))$, size: 12pt),
    anchor: "north",
    padding: 0.5em,
  )
  // neuron(i5, fill: rgb(fuchsia), label: text(" ", size: 2pt))


  let stdc = (ode_xoffset + sofa_xoffset, ode_yoffset)
  content(stdc, text($hat(S)(s^1_0), ..., hat(S)(s^1_T)$, size: 14pt))
  line(
    (ode_xoffset + 3, ode_yoffset),
    cadd(stdc, (-2, 0)),
    mark: emark,
    name: "integrate",
  )
  content("integrate", [Numerically Integrate], anchor: "south", padding: .3em)
  content(
    (stdc.at(0), title_offset + 0.3),
    align(
      text("SOFA-score Trajectory\n", size: 16pt)
        + text("eqs. (5) and (6)", size: 10pt),
      center,
    ),
    anchor: "south",
    padding: .3em,
  )
})
#figure(method_fig)

// #repr("c1".from-unicode()())
// #repr(normal-f(rng, loc:0, scale:0.5))
