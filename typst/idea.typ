#import "@preview/cetz:0.3.4": canvas, draw
#import draw: line, content, circle, rect

#set page(width: auto, height: auto, margin: 8pt)

#let cadd(p1, p2) = {
  (p1.at(0)+p2.at(0), p1.at(1) + p2.at(1))
}

#let mapping_figure = canvas({
  let arrend = (mark: (end: "stealth", fill: black, scale: .5, offset: 0.03))
  let arrb = (mark: (start: "stealth", end: "stealth", fill: black, scale: .5, offset: 0.03))

  let dnode(pos, label, name: none, rect-size: 1) = {
    circle(pos, radius: rect-size,  name: name, stroke: none,)
    content(pos, label)
  }

  let hd = 8
  let vd = 5
  dnode((hd, 0), "Latent Representation\n" + align(center, $(alpha, beta, sigma)$), name: "latent")
  
  dnode((0, 0), "Medical Record\n" + align(center,$(mu_1, ..., mu_n)$), name: "dr", rect-size: 1.3)
  dnode((hd, -vd), text("SOFA\nscore"), name: "da", rect-size: 1)

  line("dr", "da", ..arrend, name: "dra")
  line("dr", "latent", ..arrend, name: "rmd")
  line("latent", "da", ..arrend, name: "ode")
  content((rel:(.4, -.4), to: "dra" + ".mid"), "Diagnosis", anchor: "east", padding: .7cm)
  content("ode", "Cluster Ratio\n" + align(center, $f^1(alpha, beta, sigma)$), anchor: "west", padding: .3cm)
  content("rmd", $cal(N)_theta^"Enc" (mu_1,..., mu_n)$, anchor: "south", padding: .3cm)
  
})

#figure(mapping_figure)
