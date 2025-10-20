#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/suiji:0.4.0": *
#import draw: bezier, circle, content, line


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
#let cround = a => (calc.round(a.at(0)), calc.round(a.at(1)))
#let sign(x) = {
  if x > 0 {
    1
  } else if x < 0 {
    -1
  } else {
    0
  }
}

#let emark = (end: "stealth", fill: black, scale: .4, offset: 0.03)
#let bmark = (
  start: "stealth",
  end: "stealth",
  fill: black,
  scale: .4,
  offset: 0.03,
)
#let cmalpha(x) = text(fill: fuchsia, $#x$)
#let cmbeta(x) = text(fill: purple, $#x$)
#let cmsigma(x) = text(fill: orange, $#x$)

#let connect-layers(
  xoff_start,
  start-count,
  xoff_end,
  end-count,
  yoff_start,
  yoff_end,
) = {
  let start-y = start-count / 2 * 0.8
  let end-y = end-count / 2 * 0.8

  for ii in range(start-count) {
    for jj in range(end-count) {
      let start = (xoff_start, yoff_start - ii * 0.8)
      let end = (xoff_end, yoff_end - jj * 0.8)
      draw.line(start, end, stroke: rgb("#aaa") + .5pt)
    }
  }
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


#let dashed(p1, p2, name: none, node-radius: 1, c: orange) = {
  let dir = cnorm(csub(p2, p1))
  let p1a = cadd(p1, cscale(dir, node-radius))
  let p2a = csub(p2, cscale(dir, node-radius))
  line(p1a, p2a, stroke: (dash: "dashed", paint: c), name: name)
}


#let curved(p1, p2, rng, node-radius: 1) = {
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

#let lines-between(points, rng, node-radius: 1) = {
  for i in range(0, points.len() - 1) {
    let point1 = points.at(i)
    for j in range(i + 1, points.len()) {
      let point2 = points.at(j)
      curved(point1, point2, rng, node-radius: node-radius)
    }
  }
}
