#import "@preview/cetz:0.4.2": canvas, draw
#import draw: bezier, circle, content, line, rect
#import "helper.typ": *

#set page(width: auto, height: auto, margin: 8pt, fill: none)

#let x-range = 4
#let y-range = 4
#let kernel-size = 1
#let ratio = 1
#let zc = (2.8, 2.3)

#let fsq_fig = canvas({

  // Draw kernel
  circle(
    zc,
    radius: 1.6,
    stroke: none,
    fill: gradient.radial(
      // ..color.map.viridis.rev(),
      // yellow,
      // lime,
      // aqua,
      // white,
      // TODO maybe after 0.13.1 update we can do this
      yellow,
      lime,
      aqua.transparentize(100%),
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
        radius: 2pt,
        fill: gradient
          .linear(..color.map.viridis)
          .sample(calc.sin(x) * 50% + calc.sin(y) * 50%),
        stroke: none,// .5pt + black,
      )
      
    }
  }
  
  for x in range(-kernel-size, kernel-size + 1) {
    for y in range(-kernel-size, kernel-size + 1) {
      circle(
        cadd(cround(zc), (x, ratio * y)),
        radius: 2pt,
        fill: none,
        // fill: gradient
        // .linear(..color.map.plasma)
        // .sample(calc.cos(x) * 30% + calc.sin(y) * 50%),
        stroke: 1.2pt + red,
      )
    }
  }


  // Draw z
  dashed(zc, cround(zc), node-radius: 0, c: black)
  circle(zc, radius: 2pt, fill: red, stroke: none, name: "zp")
  content(
    (rel: (.15, 0.15), to: zc),
    $hat(z)$,
    anchor: "west",
  )
  content(
    (rel: (.15, -0.15), to: cround(zc)),
    $tilde(z)$,
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
    $hat(bold(z))=(hat(z)_beta, hat(z)_sigma)$,
    anchor: "north",
  )
  line((5, y-range/2), (7, y-range / 2), name: "flow1", mark: emark)
  content(
    (rel: (0.2, -0.2), to: "flow1"),
    $s^1 (bold(hat(z)))$,
    anchor: "north",
  )


  rect((rel: (-1, -1.5), to: zc), (rel: (1.4, 0.9), to: zc), stroke: (paint: red.lighten(1%), dash: "dotted"))

})
#figure(fsq_fig)
