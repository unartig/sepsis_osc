#import "@preview/cetz:0.4.2": canvas, draw

#set page(width: auto, height: auto, margin: 8pt, fill: none)
#set text(size: 10pt)

#let show-shapes = true   // set to false to hide all boxes / arrows
#let show-dims  = false    // set to false to hide dimension numbers like (104), (52→64)



// Returns its argument only when show-dims is true, otherwise empty content.
#let dim(d) = if show-dims { [(#d)] } else { [] }
// Arrow form: "a → b"  — pass two values
#let dimarr(a, b) = if show-dims { [(#a → #b)] } else { [] }

#let input-dim = 2
#let hidden-dim = 32

#let mlp_fig = canvas({
    import draw: *

    // Helper function to create a box with text
    let box-node(pos, width, height, _text, fill: white) = {
      if show-shapes {
        rect(pos, (rel: (width, height)), fill: fill, stroke: black, radius: 0.1)
      }
      content(
        (pos.at(0) + width/2, pos.at(1) + height/2),
        _text,
        anchor: "center"
      )
    }

    // Helper function to draw arrow
    let arrow-down(from-y, to-y, x) = {
      if show-shapes {
        line((x, from-y), (x, to-y), mark: (end: ">"))
      }
    }

    let arrow-right(from-x, to-x, y) = {
      if show-shapes {
        line((from-x, y), (to-x, y), mark: (end: ">"))
      }
    }

    // Conditional polyline helper (used for split/merge arrows)
    let cline(..args) = {
      if show-shapes { line(..args) }
    }

    // Conditional circle
    let ccircle(..args) = {
      if show-shapes { circle(..args) }
    }

    // Layout parameters
    let x-center = 0
    let box-width = 2.2
    let box-height = 0.6
    let y-gap = 1.
    let y-current = 0
    let eps = 0.1

    // Input
    box-node((x-center - box-width/2, y-current), box-width, box-height,
      [Input $(beta_t, sigma_t)$], fill: rgb("#e3f2fd"))
    y-current -= y-gap
    arrow-down(y-current + y-gap, y-current + box-height, x-center)

    // layer
    let pw = 1
    box-node((x-center - box-width/2 - pw/2, y-current), box-width + pw, box-height,
      align(center, text()[Linear + tanh\ #dimarr(2, hidden-dim)]), fill: rgb("#c8e6c9"))
    y-current -= y-gap

    arrow-down(y-current + y-gap, y-current + box-height, x-center)
    box-node((x-center - box-width/2 - pw/2, y-current), box-width + pw, box-height,
      align(center, text()[Linear + tanh\ #dimarr(2, hidden-dim)]), fill: rgb("#c8e6c9"))
    y-current -= y-gap

    arrow-down(y-current + y-gap, y-current + box-height, x-center)
    box-node((x-center - box-width/2 - pw/2, y-current), box-width + pw, box-height,
      align(center, text()[Linear + sigmoid\ #dimarr(2, hidden-dim)]), fill: rgb("#c8e6c9"))

    y-current -= y-gap
    arrow-down(y-current + y-gap, y-current + box-height, x-center)

    // Output head
    let x-gap = 3.5
    box-node((x-center - box-width/2, y-current), box-width, box-height,
      [$s^1_t$ #dimarr(hidden-dim, 1)], fill: rgb("#ffccbc"))

  })

#figure(mlp_fig)
