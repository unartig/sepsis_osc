#import "@preview/cetz:0.4.2": canvas, draw
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
#let cmred(x) = text(fill: red, $#x$)

#let cmw(x) = text(fill: gray, $#x$)
#let cmb(x) = text(fill: blue, $#x$)
#let cmg(x) = text(fill: olive, $#x$)
#let cmp(x) = text(fill: purple, $#x$)
#let cmpp(x) = text(fill: fuchsia, $#x$)
#let cmo(x) = text(fill: orange, $#x$)
#let cmr(x) = text(fill: red, $#x$)

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
  line(p1a, p2a, stroke: (dash: "dashed", paint: c, thickness: 1.5pt), name: name)
}

#let curved(p1, p2, node-radius: 1, c: purple, r: true) = {
  let dir = cnorm(csub(p2, p1))
  let p1a = cadd(p1, cscale(dir, node-radius))
  let p2a = csub(p2, cscale(dir, node-radius))
  
  // Use simple pseudo-random offsets based on position
  let seed = calc.rem(int(p1.at(0) * 100 + p1.at(1) * 100 + p2.at(0) * 100 + p2.at(1) * 100), 100)
  let color-seed = if c == color.fuchsia {6} else {2}
  seed = color-seed + seed
  
  let d-range = 0.3
  let d1 = calc.rem(seed * 100, int(d-range * 20)) / 10.0 - d-range
  let d2 = calc.rem(seed * 23, int(d-range * 20)) / 10.0 - d-range
  // let (d1, d2) = (0.1, 0.1)

  let th = 0.0
  if r {
    let (th-min, th-max) = (0.6, 1.8)
    th = th-min + calc.rem(seed * 31 + color-seed, int((th-max - th-min) * 10)) / 10.0
  } else {
    th = 0.7
  }
  
  let ctrl = cadd(midp(p1a, p2a), (d1, d2))
  let ctrl = cadd(midp(p1a, p2a), (d1, d2))
  
  draw.bezier(p1a, p2a, ctrl,
    stroke: (paint: c, thickness: th * 1pt),
    mark: (
      start: "stealth",
      end: "stealth",
      fill: c,
      scale: .4,
      offset: 0.03,
    ),
  )
}


#let lines-between(points, node-radius: 1, c: purple, r: true) = {
  let curves = ()
  let curve = none
  for i in range(0, points.len() - 1) {
    for j in range(i + 1, points.len()) {
      curved(points.at(i), points.at(j), node-radius: node-radius, c: c, r: r)
      curves.push(curve)
    }
  }
}
