#import "@preview/cetz:0.4.2": canvas, draw


#set page(margin: 1cm)

#let cohort_fig = canvas({
  import draw: *
  
  let box-width = 5
  let box-height = 2
  let y-spacing = 2.5
  let x-spacing = 7
  
  // Helper function to draw a box with text
  let flow-box(pos, head, n-number, name, exc: false) = {
    let x = if exc {pos.at(0) + x-spacing} else {pos.at(0)}
    let y = pos.at(1)
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
  let y-pos = 0
  
  flow-box((0, y-pos), "All ICU stays", [73181], "base")
  
  y-pos -= y-spacing
  flow-box((0, y-pos), "Length of stay ≤ 6h", [1004], "exc-los", exc: true)
  line((0, y-pos), "exc-los.west", mark: (end: "stealth"))
  
  y-pos -= y-spacing
  flow-box((0, y-pos), "<4h with any \n measurement", [50], "exc-4h", exc: true)
  line((0, y-pos), "exc-4h.west", mark: (end: "stealth"))
  
  y-pos -= y-spacing
  flow-box((0, y-pos), ">12h between any \n measurements", [165], "exc-gap", exc: true)
  line((0, y-pos), "exc-gap.west", mark: (end: "stealth"))
  
  y-pos -= y-spacing
  flow-box((0, y-pos), "Age < 18 years", [0], "exc-age", exc: true)
  line((0, y-pos), "exc-age.west", mark: (end: "stealth"))
  
  y-pos -= y-spacing
  flow-box((0, y-pos), "Sepsis onset before 6h", [8537], "exc-sepsis", exc: true)
  line((0, y-pos), "exc-sepsis.west", mark: (end: "stealth"))

  y-pos -= y-spacing
  flow-box((0, y-pos), "Sepsis Cohort", [63425], "cohort")
  line("base.south", "cohort.north", mark: (end: "stealth"))
})
