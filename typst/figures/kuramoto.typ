#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/suiji:0.4.0": *
#import draw: circle, content, line, rect

#set page(width: auto, height: auto, margin: 8pt)

#let text-size = 9pt
#let rng1 = gen-rng-f(123)
// Define styles and constants
#let radius = 1.
#let vertex-rad = 0.05 * radius
#let x-sep = 3.6

// Helper function for dressed vertices
#let dressed-vertex(pos, label: none, rel-label: none, name: none, ..rest) = {
  circle(pos, radius: vertex-rad, fill: blue, name: name)
  if label != none {
    let label-pos = if rel-label != none { (rel: rel-label, to: pos) } else {
      pos
    }
    content(label-pos, $#label$, ..rest)
  }
}

#let kuramoto_fig = canvas({
  circle((0, 0), radius: radius, stroke: 1pt, name: "loop")
  content(
    (0, 1.1),
    text(
      "Desynchronized",
      size: text-size
    ),
    anchor: "south",
    padding: .3em,
  )
  let ang = 0.0deg
  let col = 0.0
  for i in range(0, 50) {
    (rng1, ang) = uniform(rng1, high: 0, low: 360) // random angle
    (rng1, col) = uniform(rng1, high: 0, low: 100) // random color
    ang = ang * 1deg
    col = col * 1%
    circle(
      (calc.cos(ang) * radius, calc.sin(ang) * radius),
      radius: vertex-rad,
      fill: gradient.linear(..color.map.plasma).sample(col), // random color
      stroke: none,
    )
  }

  circle((x-sep * radius, 0), radius: radius, stroke: 1pt, name: "loop2")
  content(
    (x-sep, 1.1),
    text(
      "Partially Synchronized",
      size: text-size
    ),
    anchor: "south",
    padding: .3em,
  )
  let o = 0.0
  let d = 0.0
  let high = 90
  for i in range(0, 100) {
    (rng1, ang) = uniform(rng1, high: 0, low: high) // random angle
    (rng1, d) = uniform(rng1, high: 0, low: 0.2)
    (rng1, col) = uniform(rng1, high: 60, low: 95)
    o = (1 - calc.abs(ang - high / 2) / (high / 2))
    d = o * o * d
    ang = ang * 1deg
    col = col * 1%
    circle(
      (
        calc.cos(ang + 30deg) * (radius + d) + x-sep,
        calc.sin(ang + 30deg) * (radius + d),
      ),
      radius: vertex-rad,
      fill: gradient.linear(..color.map.plasma).sample(col), // random color
      stroke: none,
    )
  }
  for i in range(0, 10) {
    (rng1, ang) = uniform(rng1, high: 0, low: 360) // random angle
    (rng1, col) = uniform(rng1, high: 60, low: 80) // random color
    ang = ang * 1deg
    col = col * 1%
    circle(
      (calc.cos(ang) * radius + x-sep, calc.sin(ang) * radius), // random color
      radius: vertex-rad,
      fill: gradient.linear(..color.map.plasma).sample(col),
      stroke: none,
    )
  }
  circle((2 * x-sep * radius, 0), radius: radius, stroke: 1pt, name: "loop2")
  content(
    (2 * x-sep, 1.1),
    text(
      "Fully Synchronized",
      size: text-size
    ),
    anchor: "south",
    padding: .3em,
  )
  let high = 40
  for i in range(0, 100) {
    (rng1, ang) = uniform(rng1, high: 0, low: high) // random angle
    (rng1, d) = uniform(rng1, high: 0, low: 0.2)
    (rng1, col) = uniform(rng1, high: 69, low: 75) // random color
    o = (1 - calc.abs(ang - high / 2) / (high / 2))
    d = o * o * d
    ang = ang * 1deg
    col = col * 1%
    circle(
      (
        calc.cos(ang + 30deg) * (radius + d) + 2 * x-sep,
        calc.sin(ang + 30deg) * (radius + d),
      ),
      radius: vertex-rad,
      fill: gradient.linear(..color.map.plasma).sample(col), // random color,
      stroke: none,
    )
  }


  rect(
    (2.65 * x-sep, 0.7),
    (2.65 * x-sep + 0.2, -0.9),
    stroke: 1pt,
    fill: gradient.linear(..color.map.plasma, angle: -90deg),
  )
  content(
    (2.65 * x-sep + 0.1, 0.8),
    text(
      "Frequency",
      size: 8pt,
    ),
    anchor: "south",
    padding: .3em,
  )


  content(
    (2.67 * x-sep + 0.6, 0.4),
    text(
      $dot(phi)>0$,
      size: 8pt,
    ),
    anchor: "south",
    padding: .3em,
  )
  content(
    (2.67 * x-sep + 0.3, -0.3),
    text(
      $0$,
      size: 8pt,
    ),
    anchor: "south",
    padding: .3em,
  )
  content(
    (2.67 * x-sep + 0.6, -1),
    text(
      $dot(phi)<0$,
      size: 8pt,
    ),
    anchor: "south",
    padding: .3em,
  )

  line(
    (0, -1.5),
    (2 * x-sep, -1.5),
    mark: (end: "stealth", fill: black),
    name: "x-axis",
  )
  content((x-sep, -1.6), text("Increasing " + $Kappa$, size: text-size), anchor: "north")
})

#figure(kuramoto_fig)
