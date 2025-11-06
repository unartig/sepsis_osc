#import "@preview/cetz:0.3.4": canvas, draw
#import draw: circle, content, line

#set page(width: auto, height: auto, margin: 8pt)

#let tree_fig = canvas({
  let node-sep = 3.5 // Horizontal separation between nodes
  let level-sep = 3. // Vertical separation between levels
  let node-radius = 1
  let arrow-style = (
    mark: (end: "stealth", fill: black, scale: .5, offset: 0.03),
  )

  // Helper to draw a node with label
  let draw-node(pos, label, name: none, full_radius: true) = {
    circle(
      pos,
      radius: if full_radius { node-radius } else { node-radius / 1.6 },
      name: name,
      stroke: none,
    )
    content(pos, label)
  }

  // Helper to draw edge label
  let draw-edge-label(from, to, label, left: true) = {
    let anchor = if left { "east" } else { "west" }
    content(
      (rel: (if left { -0.3 } else { 0.3 }, 0), to: from + "-" + to + ".mid"),
      text($#label$, size: 0.3cm),
      anchor: anchor,
    )
  }

  content((2 * node-sep, -0.5 * level-sep), text(
    "Parenchymal Layer",
    style: "oblique",
    weight: "semibold",
  ))
  content((3 * node-sep, -1.5 * level-sep), text(
    "Immune Layer",
    style: "oblique",
    weight: "semibold",
  ))
  // Draw nodes level by level
  // Root (level 0)
  draw-node((0, 0), "Base State", name: "p0")

  // Level 1
  draw-node(
    (-node-sep, -level-sep),
    text(gray)[Pathological] + "\nOrgan Dysfunction\nFig5 B,C",
    name: "p1",
  )
  //   draw-node((-node-sep, -level-sep), "Pathological" +
  // figure(
  //   image("images/fig5B.svg", width: 130%, fit: "stretch")
  // ), name: "p1")
  draw-node((node-sep, -level-sep), "Healthy", name: "p2")
  draw-node((node-sep, -0.9 * level-sep), "", name: "p2a")

  // Level 2
  draw-node(
    (0 * node-sep, -2 * level-sep),
    text(gray)[Vulnerable] + "\n(Suspected) Infection\nFig5 C'",
    name: "p3",
  )
  draw-node(
    (1 * node-sep, -2 * level-sep),
    text(gray)[Resilient] + "\n" + text(gray)[Fig5 D,E],
    name: "p4",
    full_radius: false,
  )
  draw-node((2 * node-sep, -2 * level-sep), "Fully Healthy\nFig5 A", name: "p5")


  // Draw edges
  line("p0", "p1", ..arrow-style, name: "p0-p1")
  line("p0", "p2", ..arrow-style, name: "p0-p2")
  line("p2", "p3", ..arrow-style, name: "p2-p3")
  line("p2a", "p4", ..arrow-style, name: "p2a-p4")
  line("p2", "p5", ..arrow-style, name: "p2-p5")

  // Draw edge labels
  draw-edge-label("p0", "p1", "Desync " + $chevron.l dot(phi)^1_j chevron.r$)
  draw-edge-label(
    "p0",
    "p2",
    "Sync " + $chevron.l dot(phi)^1_j chevron.r$,
    left: false,
  )
  draw-edge-label(
    "p2",
    "p5",
    "Sync " + $chevron.l dot(phi)^2_j chevron.r$,
    left: false,
  )
  draw-edge-label(
    "p2a",
    "p4",
    "Desync " + $\ chevron.l dot(phi)^2_j chevron.r$,
    left: false,
  )
  draw-edge-label("p2", "p3", "Splay " + $phi^2_j$ + " or " + $phi^1_j$)
})
