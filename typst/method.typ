#import "@preview/cetz:0.3.4": canvas, draw
#import draw: bezier, circle, content, line
#import "@preview/suiji:0.4.0": *

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
#let mfuchsia(x) = text(fill: fuchsia, $#x$)
#let mpurple(x) = text(fill: purple, $#x$)
#let morange(x) = text(fill: orange, $#x$)
#let emark = (end: "stealth", fill: black, scale: .4, offset: 0.03)
#let bmark = (
  start: "stelth",
  end: "stealth",
  fill: black,
  scale: .4,
  offset: 0.03,
)
#let cadd = (a, b) => (a.at(0) + b.at(0), a.at(1) + b.at(1))
#let cscale = (v, s) => (v.at(0) * s, v.at(1) * s)
#let midp = (a, b) => cscale(cadd(a, b), 0.5)
#let csub = (a, b) => (a.at(0) - b.at(0), a.at(1) - b.at(1))
#let clen = v => calc.sqrt(v.at(0) * v.at(0) + v.at(1) * v.at(1))
#let cnorm = v => {
  let len = clen(v)
  if len == 0 {
    (0, 0)
  } else {
    (v.at(0) / len, v.at(1) / len)
  }
}
#let cscale = (v, s) => (v.at(0) * s, v.at(1) * s)
#let cadd = (a, b) => (a.at(0) + b.at(0), a.at(1) + b.at(1))
#let sign(x) = {
  if x > 0 {
    1
  } else if x < 0 {
    -1
  } else {
    0
  }
}

#let curved(p1, p2, rng) = {
  let dir = cnorm(csub(p2, p1))
  let p1a = cadd(p1, cscale(dir, node-radius))
  let p2a = csub(p2, cscale(dir, node-radius))

  let mid = midp(p1a, p2a)
  let d1 = 0.0
  let d2 = 0.0
  let th = 0.0
  // (irng, d1) = uniform(irng, high:1, low:-1)
  // (irng, d2) = uniform(irng, high:1, low:-1)
  (rng, d1) = choice(rng, (.15, -.15))
  (rng, d2) = choice(rng, (.15, -.15))
  // d1 = sign(p1.at(0)+p2.at(0))*0.5
  // d2 = sign(p1.at(0)+p2.at(1))*0.5
  let ctrl-offset = (d1, d2)
  let ctrl = cadd(mid, ctrl-offset)
  (rng, th) = uniform(rng, low: 0, high: 1, size: none)
  // bezier(p1a, p2a, ctrl, stroke: (paint: black, thickness: th))
  bezier(p1a, p2a, ctrl, ..(
    // stroke: .5pt + black,
    stroke: (paint: purple, thickness: th * 1pt),
    mark: (
      start: "stealth",
      end: "stealth",
      fill: purple,
      scale: .4,
      offset: 0.03,
    ),
  ))
  // return rng
}

#let lines-between(points, rng) = {
  for i in range(0, points.len() - 1) {
    let point1 = points.at(i)
    for j in range(i + 1, points.len()) {
      let point2 = points.at(j)
      curved(point1, point2, rng)
    }
  }
}

#let dashed(p1, p2, name: none) = {
  let dir = cnorm(csub(p2, p1))
  let p1a = cadd(p1, cscale(dir, node-radius))
  let p2a = csub(p2, cscale(dir, node-radius))
  line(p1a, p2a, stroke: (dash: "dashed", paint: orange), name: name)
}

#let neuron(pos, fill: white, label: none, s: none, st: none) = {
  draw.content(
    pos,
    text(label, size: if s != none { 9pt } else { 12pt }),
    frame: "circle",
    fill: fill,
    stroke: if st != none { 1pt + fuchsia } else { 0.5pt + black },
    padding: 1pt,
  )
}

#let draw-neurons(lis) = {
  for (x, count, fill, labels, y-offset, s) in lis {
    for idx in range(count) {
      let y-pos = y-offset - idx * 0.8
      let label = if labels != none { labels.at(idx) } else { "  " }
      neuron((x, y-pos), fill: fill, label: label, s: s)
    }
  }
}


#let connect-layers(
  start-pos,
  start-count,
  end-pos,
  end-count,
  yoff1,
  yoff2,
) = {
  let start-y = start-count / 2 * 0.8
  let end-y = end-count / 2 * 0.8

  for ii in range(start-count) {
    for jj in range(end-count) {
      let start = (start-pos, yoff1 - ii * 0.8)
      let end = (end-pos, yoff2 - jj * 0.8)
      draw.line(start, end, stroke: rgb("#aaa") + .5pt)
    }
  }
}

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
        $mfuchsia(alpha_0)$,
        $mpurple(beta_0)$,
        $morange(sigma_0)$,
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
        $mfuchsia(alpha_t)$,
        $mpurple(beta_t)$,
        $morange(sigma_t)$,
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
        $mfuchsia(alpha_(t+1))$,
        $mpurple(beta_(t+1))$,
        $morange(sigma_(t+1))$,
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
  lines-between(layer1, rng1)

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
  lines-between(layer2, rng2)
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

  dashed(p1, i1)
  dashed(p2, i2)
  dashed(p3, i3)
  dashed(p4, i4)
  dashed(p5, i5, name: "dash")
  dashed(p6, i6)
  content(
    "dash",
    text($morange(sigma)$, size: 12pt),
    anchor: "south-west",
    padding: 0.7em,
  )
  content(
    p5,
    text($phi^1_j(mfuchsia(alpha))$, size: 12pt),
    anchor: "south-east",
    padding: 0.4em,
  )
  // neuron(p5, fill: rgb(fuchsia), label: text(" ", size: 2pt))
  content(
    midp(p5, p6),
    text($kappa^1_(i j (mpurple(beta)))$, size: 12pt),
    anchor: "south",
    padding: 0.5em,
  )
  content(
    i5,
    text($phi^2_j(mfuchsia(alpha))$, size: 12pt),
    anchor: "north-east",
    padding: 0.3em,
  )
  content(
    midp(i3, i4),
    text($kappa^2_(i j (mpurple(beta)))$, size: 12pt),
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
