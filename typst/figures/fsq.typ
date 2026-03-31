#import "@preview/cetz:0.4.2": canvas, draw
#import draw: bezier, circle, content, line, rect
#import "helper.typ": *

#set page(width: auto, height: auto, margin: 8pt, fill: white)

#let x-range = 4
#let y-range = 4
#let kernel-size = 1
#let ratio = 1
#let zc = (2.8, 2.3)
#let pradius = 3.5pt

#let fsq_fig = canvas({

  // Draw kernel
  circle(
    zc,
    radius: 1.6,
    stroke: none,
    fill: gradient.radial(
      red.transparentize(0%),
      red.transparentize(100%),
    ),
    fill-opacity: 1%,
    name: "kernel",
  )
  // Draw axes with arrows
  let arrow-style = (mark: (end: "stealth", fill: black))
  let x-end = (x-range + 0.5, 0)
  let y-end = (0, ratio * y-range + 0.5)
  line((0, 0), x-end, ..arrow-style)
  line((0, 0), y-end, ..arrow-style, name: "y-axis")

  // Add axis labels
  content((rel: (-.25, .25), to: x-end), $beta$, anchor: "south-west")
  content(
    (rel: (.55, 0), to: "y-axis.end"),
    $sigma$,
    anchor: "north-east",
  )

  // Draw coordinate points
  for x in range(0, x-range + 1) {
    for y in range(0, y-range + 1) {
      circle(
        (x, ratio * y),
        radius: pradius,
        fill: gradient
          .linear(..color.map.viridis)
          .sample(calc.sin(x - 1.75) * 50% + calc.sin(y - 1.3) * 50%),
        stroke: none,// .5pt + black,
      )
      
    }
  }
  
  for x in range(-kernel-size, kernel-size + 1) {
    for y in range(-kernel-size, kernel-size + 1) {
      circle(
        cadd(cround(zc), (x, ratio * y)),
        radius: pradius,
        fill: none,
        // stroke: none,
        stroke: 1pt + red,
      )
    }
  }


  // Draw z
  line(zc, cround(zc), node-radius: 0, c: black, stroke: (dash: "densely-dotted"))
  circle(zc, radius: pradius, fill: red, stroke: black, name: "zp")
  content(
    (rel: (.15, 0.15), to: zc),
    text(size: 8pt)[$(beta,sigma)$],
    anchor: "west",
  )
  content(
    (rel: (.15, -0.15), to: cround(zc)),
    text(size: 7pt)[$(tilde(beta),tilde(sigma))$],
    anchor: "west",
  )


  let netend = 2.3
  connect-layers(
    -netend - 1,
    6,
    -netend,
    2,
    y-range,
    y-range - ((1.5 / 4) * y-range),
  )
  line((-netend, y-range / 2), (-0.5, y-range / 2), name: "flow1", mark: emark)
  content(
    (rel: (0.2, -0.2), to: "flow1"),
    $beta,sigma$,
    anchor: "north",
  )
  line((5, y-range/2), (7, y-range / 2), name: "flow1", mark: emark)
  content(
    (rel: (0.2, -0.2), to: "flow1"),
    $tilde(s)^1 (beta,sigma)$,
    anchor: "north",
  )


  rect((rel: (-1, -1.5), to: zc), (rel: (1.4, 0.9), to: zc), stroke: (thickness: 1.5pt, paint: red.lighten(1%), dash: "dotted"), radius: 0.1)
  content((rel: (0.3, -1.8), to: zc), text(size:8pt)[$cal(N)_(k times k)(tilde(beta),tilde(sigma))$])
  // content((rel: (3.3, -1.7), to: zc), text(size:8pt)[$cal(N)_(k times k)(tilde(beta),tilde(sigma))$])
  // line((rel: (1.5, -1), to: zc), (rel: (2.5, -1.5), to: zc), stroke: (dash: "dashed"))


  
  rect(
    (rel: (2.8, 0.25), to: zc),
    (rel: (3.0, 1.8), to: zc),
    stroke: 1pt,
    fill: gradient.linear(..color.map.viridis, angle: -90deg),
  )
  content(
    (rel: (2.9, 1.8), to: zc),
    text(
      "Desynchronization",
      size: 8pt,
    ),
    anchor: "south",
    padding: .3em,
  )
  // line((rel: (1.2, 1.7), to: zc), (rel: (2.7, 1.2), to: zc), stroke: (dash: "dashed"))
})
#figure(fsq_fig)
