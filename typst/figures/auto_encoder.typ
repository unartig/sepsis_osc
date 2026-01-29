#import "@preview/cetz:0.4.2": canvas, draw

#set text(size: 4pt)

#let input-dim = 104
#let half-dim = calc.quo(input-dim, 2)
#let hidden-dim = 64
#let pre-head-dim = 16
#let h-dim = 4
#let z-dim = 2

#let ae_fig = canvas({
    import draw: *
    
    // Helper function to create a box with text
    let box-node(pos, width, height, _text, fill: white) = {
      rect(pos, (rel: (width, height)), fill: fill, stroke: black)
      content(
        (pos.at(0) + width/2, pos.at(1) + height/2),
        _text,
        anchor: "center"
      )
    }
    
    // Helper function to draw arrow
    let arrow-down(from-y, to-y, x) = {
      line((x, from-y), (x, to-y), mark: (end: ">"))
    }
    
    let arrow-right(from-x, to-x, y) = {
      line((from-x, y), (to-x, y), mark: (end: ">"))
    }
    
    // Layout parameters
    let x-center = 0
    let box-width = 4.2
    let box-height = 0.6
    let y-gap = 1.
    let y-current = 0
    let eps = 0.1
    
    content((-5, 1), text(14pt)[*A*])
    // Input
    box-node((x-center - box-width/2, y-current), box-width, box-height, 
      [Input $bold(mu)_0$ (#input-dim)], fill: rgb("#e3f2fd"))
    y-current -= y-gap
    
    // Split arrows
    let split-y = y-current
    let left-x = x-center - 2.7
    let right-x = x-center + 2.7
    
    line((x-center - eps, 0), (x-center - eps, -y-gap/2), (left-x, -y-gap/2), (left-x, -y-gap + eps), mark: (end: ">"))
    line((x-center + eps, 0), (x-center + eps, -y-gap/2), (right-x, -y-gap/2), (right-x, -y-gap + eps), mark: (end: ">"))
    
    y-current -= 0.5
    
    // Features branch (left)
    box-node((left-x - 1.5, y-current), 3, box-height, 
      [Features (#half-dim)], fill: rgb("#fff3e0"))
    
    // Indicators branch (right)
    box-node((right-x - 1.5, y-current), 3, box-height, 
      [Indicators (#half-dim)], fill: rgb("#fff3e0"))
    
    y-current -= y-gap
    
    // Gating
    box-node((left-x - 2.5, y-current), 5, box-height, 
      [Linear + Sigmoid (#half-dim → #half-dim)], fill: rgb("#ffe0b2"))
    arrow-down(y-current + y-gap, y-current + box-height, left-x)
    
    y-current -= y-gap * 1.25
    
    // Arrows to concat
    let concat-y = y-current
    line((left-x, concat-y + y-gap*1.25), (left-x, concat-y + y-gap), (x-center - eps, concat-y + y-gap), (x-center - eps, concat-y + box-height), mark: (end: ">"))
    line((right-x, y-current + 2*y-gap + box-height - 3.4*eps),  (right-x, concat-y + y-gap),  (x-center + eps, concat-y + y-gap), (x-center + eps, concat-y + box-height), mark: (end: ">"))
    
    // Concatenate
    box-node((x-center - 2.75, y-current), 5.5, box-height, 
      [Concatenate (#half-dim + #half-dim → #input-dim)], fill: rgb("#e1bee7"))
    
    y-current -= y-gap + box-height*1.7
    arrow-down(y-current + y-gap*2, y-current + y-gap*1.5 + eps, x-center)
    
    // First layer
    let pw = 1
    box-node((x-center - box-width/2 - pw/2, y-current), box-width + pw, box-height*2.7, 
      align(center, text()[Residual Block 1 \ LayerNorm + Linear + GELU \ (#input-dim → #hidden-dim)]), fill: rgb("#c8e6c9"))
    
    y-current -= y-gap + 2*eps
    arrow-down(y-current + y-gap + 2 * eps, y-current + box-height, x-center)
    
    y-gap = 1.5
    // Residual blocks
    for i in range(2) {
      box-node((x-center - box-width/2, y-current), box-width, box-height, 
        [Residual Block #(i + 2) (#hidden-dim)], fill: rgb("#c8e6c9"))
      arrow-down(y-current, y-current + box-height - y-gap, x-center)
      
      // Draw residual connection
      let mid-y = y-current + box-height + y-gap/4
      if i < 2 {
      line((x-center, mid-y), 
           (x-center + 2*box-width/3, mid-y),
           (x-center + 2*box-width/3, mid-y - y-gap + 3*eps),
           (x-center + eps, mid-y - y-gap + 3* eps),
           mark: (end: ">"), stroke: (dash: "dashed"))
            }
      circle((x-center, mid-y - y-gap + 3* eps), radius: 0.15, fill: luma(221))
      content((x-center, mid-y - y-gap + 3* eps), text(8pt)[$+$])
      y-current -= y-gap
    }
    
    // Pre-head
    box-node((x-center - box-width/2 - pw/2, y-current), box-width + pw, box-height, 
      [Linear + GELU (#hidden-dim → #pre-head-dim)], fill: rgb("#b2dfdb"))
    
    y-gap = 1
    y-current -= y-gap
    
    // Split to heads
    let head-split-y = y-current + box-height
    line((x-center - eps, head-split-y + y-gap - box-height), (x-center - eps, head-split-y + y-gap/4), (left-x, head-split-y + y-gap/4), (left-x, y-current + box-height), mark: (end: ">"))
    line((x-center + eps, head-split-y + y-gap - box-height), (x-center + eps, head-split-y + y-gap/4), (right-x, head-split-y + y-gap/4), (right-x, y-current + box-height), mark: (end: ">"))
    
    // Output heads
    let x-gap = 3.5
    box-node((left-x - x-gap/2, y-current), x-gap, box-height, 
      [$bold(h)^g_0$ (#pre-head-dim → #h-dim)], fill: rgb("#ffccbc"))
    box-node((right-x - x-gap/2, y-current), x-gap, box-height, 
      [$bold(z)_0^"raw"$ (#pre-head-dim → #z-dim)], fill: rgb("#ffccbc"))
    
    //
    // B
    // 
    let x-off = 6
    y-current = 0
    box-width = 4
    content((x-off -1, 1), text(14pt)[*B*])
    box-node((x-off +box-width/4, y-current), box-width, box-height, 
      [$bold(z)_t$ (#z-dim)], fill: rgb("#ffccbc"))

    box-width = 6
    box-height = 1.2
    y-gap = 2
    y-current -= y-gap
    box-node((x-center + x-off, y-current), box-width, box-height, 
      align(center)[Linear + LayerNorm + GELU \ (#z-dim → 16)], fill: rgb("#fff3e0"))
    arrow-down(y-current + y-gap, y-current + box-height, x-off + box-width/2)

    y-current -= y-gap
    box-node((x-center + x-off, y-current), box-width, box-height, 
      align(center)[Linear + LayerNorm + GELU \ (16 → 32)], fill: rgb("#fff3e0"))
    arrow-down(y-current + y-gap, y-current + box-height, x-off + box-width/2)

    y-current -= y-gap
    box-node((x-center + x-off, y-current), box-width, box-height, 
      align(center)[Linear + LayerNorm + GELU \ (32 → 32)], fill: rgb("#fff3e0"))
    arrow-down(y-current + y-gap, y-current + box-height, x-off + box-width/2)

    box-height = 0.6
    y-gap = 1.5
    y-current -= y-gap
    box-node((x-center + x-off, y-current), box-width, box-height, 
      [Linear (32 → #half-dim)], fill: rgb("#b2dfdb"))
    arrow-down(y-current + y-gap, y-current + box-height, x-off + box-width/2)


    y-current -= y-gap
    arrow-down(y-current + y-gap, y-current + box-height, x-off + box-width/2)
    box-width = 4
    box-node((x-center + x-off + box-width/4, y-current), box-width, box-height, 
      [$bold(hat(mu))_t$ (#half-dim)], fill: rgb("#fff3e0"))
  })

