#import "@preview/cetz:0.4.2": canvas, draw


#set page(margin: 1cm)
#set text(size: 2pt)

#let cohort_fig = canvas({
  import draw: *
  
  let box-width = 4.2
  let box-height = 2
  let x-spacing = 2.5
  let y-spacing = 3
  let y-diff= 3
  
  // Helper function to draw a box with text
  let flow-box(pos, head, n-number, name, exc: false) = {
    let x = pos.at(0)
    let y = if exc {pos.at(1) - y-spacing} else {pos.at(1)}
    let s = if exc {gray + 1pt} else {black + 1pt}
    let f = if exc {gray.lighten(50%)} else {none}
    rect(
      (x - box-width/2, y - box-height/2),
      (x + box-width/2, y + box-height/2),
      stroke: s,
      fill: f,
      name: name
    )
    
    let t = if exc {black.lighten(30%)} else {black}
    content(
      name + ".center",
      anchor: "center",
      align(center, text(t)[*#head* \ N = #n-number])
    )
  }
  
  // Flow boxes
  let x-pos = 0
  let y-pos = y-spacing
  
  flow-box((x-pos, 0), "All ICU stays", [73181], "base")
  
  x-pos += x-spacing + 1
  y-pos -= y-diff
  flow-box((x-pos, y-pos), "Length of stay ≤ 6h", [1004], "exc-los", exc: true)
  line((x-pos, 0), "exc-los.north", mark: (end: "stealth"))
  
  x-pos += x-spacing
  y-pos -= y-diff
  flow-box((x-pos, y-pos), "<4h with any \n measurement", [50], "exc-4h", exc: true)
  line((x-pos, 0), "exc-4h.north", mark: (end: "stealth"))
  
  x-pos += x-spacing
  y-pos += y-diff
  flow-box((x-pos, y-pos), $>=$ + "12h between any \n measurements", [165], "exc-gap", exc: true)
  line((x-pos, 0), "exc-gap.north", mark: (end: "stealth"))
  
  // x-pos += x-spacing
  // y-pos -= y-diff
  // flow-box((x-pos, y-pos), "Age < 18 years", [0], "exc-age", exc: true)
  // line((x-pos, 0), "exc-age.north", mark: (end: "stealth"))
  
  x-pos += x-spacing
  y-pos -= y-diff
  flow-box((x-pos, y-pos), "Sepsis onset before 6h", [8537], "exc-sepsis", exc: true)
  line((x-pos, 0), "exc-sepsis.north", mark: (end: "stealth"))
  
  x-pos += x-spacing + 1
  y-pos -= y-diff
  flow-box((x-pos, 0), "Sepsis Cohort", [63425], "cohort")
  line("base.east", "cohort.west", mark: (end: "stealth"))
})
