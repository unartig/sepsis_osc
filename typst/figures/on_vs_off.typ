#import "@preview/cetz:0.4.2": canvas, draw
#import draw: bezier, circle, content, line, rect
#import "helper.typ": *

#set page(width: auto, height: auto, margin: 8pt, fill: none)
#set text(size: 16pt)

#let x-range = 6
#let y-range = 1
#let ratio = 1

#let dx = 9
#let hatched = tiling(size: (.1cm, .1cm))[]


#let graph = (x-offset, y-offset) => {
  let arrow-style = (mark: (end: "stealth", fill: black))
  let x-end = (x-range + 0.5 + x-offset, 0 + y-offset)
  let y-end = (x-offset, ratio * y-range + 0.5 + y-offset)
  line((x-offset, 0 + y-offset), x-end, ..arrow-style)
  line((x-offset, 0 + y-offset), y-end, ..arrow-style, name: "y-axis")

  content((rel: (0, -0.2), to:x-end), $t$, anchor:"north-east")
}

#let label-pos = (x, y) => {
  circle((x,y), radius:0.15, stroke:(red))
  content((x,y), text(red, size:8pt)[*1*])
}
#let label-neg = (x, y) => {
  circle((x,y), radius:0.15, stroke:(blue))
  content((x,y), text(blue, size:8pt)[*0*])
}

#let onset = (x, y) => {
  line((x, 0 +y ), (x, y-range + y), stroke:red)
  content((rel: (0, 0.1), to: (x, y-range + y)), text(red, size:10pt)[Sepsis onset], anchor: "south")
}

#let draw-oo(on: true, off: true, control: true, labels: [Labels]) = {

  if off {
    if on {
      content((1, 6), text(size: 16pt)[*A*] + [ Offline Prediction])
    } else {
      content((1, 6), text(size: 16pt)[Offline Prediction])
    }
    let yoff = 3
    graph(0, yoff)
    onset(3, yoff)
    rect((0.2, yoff), (0.4, y-range + yoff), fill: gray)
    line((0.2  + 0.1, yoff), (0.2 + -0.3, yoff - 0.5), mark:(start:"stealth", fill:black))
    rect((0.4, yoff), (x-range - 1, y-range + yoff))
    content((rel: (0, -0.3), to:(x-range - 1, 0 + yoff)), $T$)
    content((rel: (-1, -.7), to:(0.2  + 0.1, yoff)), [Observation $bold(mu)_t$])
    content((0, 1.75 + yoff), text(red)[#labels])
    for i in range(2, 2*(x-range - 1)) {
      label-pos(i/2, 1.7 + yoff)
    }

    if control {
      content((-1, yoff + 0.5),[Case])
      yoff = 0
      content((-1, yoff + 0.5),[Control])
      graph(0, yoff)
      content((0, 1.75 + yoff), text(blue)[#labels])
      rect((0.2, yoff), (0.4, y-range + yoff), fill: gray)
      line((0.2  + 0.1, yoff), (0.2 + -0.3, yoff - 0.5), mark:(start:"stealth", fill:black))
      rect((0.4, yoff), (x-range - 1, y-range + yoff))
      content((rel: (0, -0.3), to:(x-range - 1, 0 + yoff)), $T$)
      for i in range(2, 2*(x-range - 1)) {
        label-neg(i/2, 1.7 + yoff)
      }
    }
  }

  if on {
    let yoff = 3
    if off {
      content((1 + dx, 6), text(size: 16pt)[*B*] + [ Online Prediction])
    } else {
      content((1 + dx, 6), text(size: 16pt)[Online Prediction])
    }
    graph(0 + dx, yoff)
    onset(3 + dx, yoff)
    // rect((0.2 + dx, yoff), (0.4 + dx, y-range + yoff), fill: gray)
    rect((0.4 + dx, yoff), (x-range - 1 + dx, y-range + yoff))
    content((0 + dx, 1.75 + yoff), text(red)[#labels])
    for i in range(2, 2*(x-range - 3) -1) {
      label-neg(i/2 + dx, 1.7 + yoff)
    }
    for i in range(2*(x-range - 2), 2*(x-range - 1)) {
      label-neg(i/2 + dx, 1.7 + yoff)
    }
    for i in range(5, 2*(x-range - 2)) {
      label-pos(i/2 + dx, 1.7 + yoff)
    }
    for i in range(0, 2*(x-range - 1)) {
      rect((0.2 + dx + i/2, yoff), (0.4 + dx + i/2, y-range + yoff), fill: gray)
      line((0.2 + dx + i/2 + 0.1, yoff), (0.2 + dx + i/2 -0.3, yoff - 0.5), mark:(start:"stealth", fill:black))
    }
    if not off {
      content((rel: (-1, -.7), to:(0.2  + 0.1 + dx, yoff)), [Observation $bold(mu)_t$])
    }
    if control {
      content((-1 + dx, yoff + 0.5),[Case])
      yoff = 0
      content((-1 + dx, yoff + 0.5),[Control])
      graph(0 + dx, yoff)
      // onset(3 + dx, yoff)
      // rect((0.2 + dx, yoff), (0.4 + dx, y-range + yoff), fill: gray)
      rect((0.4 + dx, yoff), (x-range - 1 + dx, y-range + yoff))
      content((0 + dx, 1.75 + yoff), text(blue)[#labels])
      for i in range(2, 2*(x-range - 1)) {
        label-neg(i/2 + dx, 1.7 + yoff)
      }
      for i in range(0, 2*(x-range - 1)) {
        rect((0.2 + dx + i/2, yoff), (0.4 + dx + i/2, y-range + yoff), fill: gray)
        line((0.2 + dx + i/2 + 0.1, yoff), (0.2 + dx + i/2 -0.3, yoff - 0.5), mark:(start:"stealth", fill:black))
      }
    }
  }
}

#let create-oo-figure(on: true, off: true, control: true, labels: [Labels]) = {
  canvas({draw-oo(off: off, on: on, control: control, labels: labels)})
}
#let oo_fig = create-oo-figure()
#figure(oo_fig)
