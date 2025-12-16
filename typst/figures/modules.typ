#import "@preview/cetz:0.4.2": canvas, draw, tree
#import "helper.typ": cmbeta, cmsigma

// Infection Indicator Module
#let inf_fig = canvas({
    import draw: *
    rect((-6, 0), (-4, 0.8), name: "ehr-1", stroke: (paint: blue.lighten(50%), dash: "dotted"), fill: blue.lighten(95%))
    content("ehr-1.center", text(fill: gray.darken(20%))[EHR $bold(mu)_(t-1)$])
    
    rect((-6, -1.5), (-4, -0.5), name: "rnn-1", stroke: (paint: green.lighten(50%), dash: "dotted"), fill: green.lighten(95%))
    content("rnn-1.center", text(fill: gray.darken(20%))[RNN $f_theta$])
    
    rect((-6, -2.7), (-4, -2), name: "inf-1", stroke: (paint: red.lighten(50%), dash: "dotted"))
    content("inf-1.center", text(fill: gray.darken(20%))[$tilde(I)_(t-1)$])
    
    line("ehr-1.south", "rnn-1.north", mark: (end: ">"), stroke: 0.8pt + gray.lighten(30%))
    line("rnn-1.south", "inf-1.north", mark: (end: ">"), stroke: 0.8pt + gray.lighten(30%))
    
    rect((-1, 0), (1, 0.8), name: "ehr", stroke: 2pt + blue)
    content("ehr.center", [EHR $bold(mu)_t$])
    
    rect((-3.5, -1.3), (-1.5, -0.7), name: "hprev", stroke: 1pt + gray)
    content("hprev.center", [$bold(h)_(t-1) in RR^h$])
    
    rect((-1, -1.5), (1, -0.5), name: "rnn", stroke: 2pt + green, fill: green.lighten(90%))
    content("rnn.center", [RNN $f_theta$])
    
    line("ehr.south", "rnn.north", mark: (end: ">"), stroke: 1.5pt)
    
    rect((-1, -2.7), (1, -2), name: "inf", stroke: 2pt + red)
    content("inf.center", [$tilde(I)_t in (0,1)$])
    
    rect((2, -1.3), (3, -0.7), name: "h", stroke: 1pt + gray)
    content("h.center", [$bold(h)_t$])
    
    line("rnn.south", "inf.north", mark: (end: ">"), stroke: 1.5pt)
    line("rnn.east", "h.west", mark: (end: ">"), stroke: 1pt + gray)
    line("hprev.east", "rnn.west", mark: (end: ">"), stroke: 1pt + gray)
    
    line("rnn-1.east", "hprev.west", mark: (end: ">"), stroke: (paint: gray, dash: "dotted"))
    
    rect((4, 0), (6, 0.8), name: "ehr+1", stroke: (paint: blue.lighten(50%), dash: "dotted"), fill: blue.lighten(95%))
    content("ehr+1.center", text(fill: gray.darken(20%))[EHR $bold(mu)_(t+1)$])
    
    rect((4, -1.5), (6, -0.5), name: "rnn+1", stroke: (paint: green.lighten(50%), dash: "dotted"), fill: green.lighten(95%))
    content("rnn+1.center", text(fill: gray.darken(20%))[RNN $f_theta$])
    
    rect((4, -2.7), (6, -2), name: "inf+1", stroke: (paint: red.lighten(50%), dash: "dotted"))
    content("inf+1.center", text(fill: gray.darken(20%))[$tilde(I)_(t+1)$])
    
    line("ehr+1.south", "rnn+1.north", mark: (end: ">"), stroke: 0.8pt + gray.lighten(30%))
    line("rnn+1.south", "inf+1.north", mark: (end: ">"), stroke: 0.8pt + gray.lighten(30%))
    line((3, -1), (4, -1), mark: (end: ">"), stroke: (paint: gray, dash: "dotted"))
  })

#let sofa_fig = canvas({
    import draw: *
    
    rect((-5, 2), (-3, 2.8), name: "ehr0", stroke: 2pt + blue)
    content("ehr0.center", [EHR $bold(mu)_0$])
    
    rect((-5.2, 0.5), (-2.8, 1.5), name: "enc", stroke: 2pt + green, fill: green.lighten(90%))
    content("enc.center", [Encoder $e_theta$])
    
    line("ehr0.south", "enc.north", mark: (end: ">"), stroke: 1.5pt)
    
    rect((-5, -1.75), (-3, -1.15), name: "z0", stroke: 2pt + red)
    content("z0.center", [$hat(bold(z))_0 in RR^2$])
    
    rect((-2, 0.65), (-0.25, 1.35), name: "h0", stroke: 1pt + gray)
    content("h0.center", [$bold(h)_0 in RR^h$])
    
    line("enc.south", "z0.north", mark: (end: ">"), stroke: 1.5pt)
    line("enc.east", "h0.west", mark: (end: ">"), stroke: 1pt + gray)
    
    rect((0.75, 0.5), (3.25, 1.5), name: "rnn0", stroke: 2pt + green, fill: green.lighten(90%))
    content("rnn0.center", [RNN $r_theta$])
    line("h0.east", "rnn0.west", stroke: 1pt +  gray, mark: (end: ">"))
    
    rect((1, 2), (3, 2.8), name: "ehr1", stroke: 2pt + blue)
    content("ehr1.center", [EHR $bold(mu)_1$])
    line("ehr1.south", "rnn0.north", mark: (end: ">"), stroke: 1.5pt)
    
    rect((4.2, 0.65), (5.95, 1.35), name: "h1", stroke: 1pt + gray)
    content("h1.center", [$bold(h)_1$])
    line("rnn0.east", "h1.west", mark: (end: ">"), stroke: 1pt + gray)
    
    rect((1, -0.125), (3, -0.625), name: "dz", stroke: 2pt + orange)
    content("dz.center", [$Delta hat(bold(z))_t$])
    
    rect((3, -1.75), (5, -1.25), name: "z1", stroke: 2pt + red)
    content("z1.center", [$hat(bold(z))_1$])
    
    line("rnn0.south", "dz.north", mark: (end: ">"), stroke: 1.5pt)
    
    // Addition
    circle((2, -1.5), radius: 0.3, stroke: 1.5pt)
    content((2, -1.5), text(10pt, weight: "bold")[+])
    line("z0.east", (1.7, -1.5), mark: (end: ">"), stroke: 1.5pt)
    line("dz.south", (2.0, -1.2), mark: (end: ">"), stroke: 1.5pt)
    line((2.3, -1.5), "z1.west", mark: (end: ">"), stroke: 1.5pt)
    
    // t=2 step
    rect((6.75, 0.5), (9.25, 1.5), name: "rnn2", stroke: (paint: green.lighten(50%), dash: "dotted"), fill: green.lighten(95%))
    content("rnn2.center", text(fill: gray.darken(20%))[RNN $r_theta$])
    
    rect((7, 2), (9, 2.8), name: "ehr2", stroke: (paint: blue.lighten(50%), dash: "dotted"), fill: blue.lighten(95%))
    content("ehr2.center", text(fill: gray.darken(20%))[EHR $bold(mu)_2$])
    line("ehr2.south", "rnn2.north", mark: (end: ">"), stroke: (paint: gray.lighten(30%), dash: "dotted"))
    
    line("rnn2.east", (10, 1.0), mark: (end: ">"), stroke: (paint: gray.lighten(30%), dash: "dotted"))
    
    rect((7, -0.125), (9, -0.625), name: "dz2", stroke: (paint: orange.lighten(50%), dash: "dotted"))
    content("dz2.center", text(fill: gray.darken(20%))[$Delta hat(bold(z))_2$])
    
    line("rnn2.south", "dz2.north", mark: (end: ">"), stroke: (paint: gray.lighten(30%), dash: "dotted"))
    
    circle((8, -1.5), radius: 0.3, stroke: (paint: gray.lighten(30%), dash:"dotted"))
    content((8, -1.5), text(10pt, fill: gray.darken(20%))[$+$])
    line("z1.east", (7.7, -1.5), mark: (end: ">"), stroke: (paint: gray.lighten(30%), dash: "dotted"))
    line("dz2.south", (8.0, -1.2), mark: (end: ">"), stroke: (paint: gray.lighten(30%), dash: "dotted"))
    line((8.3, -1.5), (10, -1.5), mark: (end: ">"), stroke: (paint:gray.lighten(30%), dash: "dotted"))
    
    line("h1.east", "rnn2.west", stroke: (paint: gray.lighten(20%), dash: "dotted"), mark: (end: ">") )
  })

// Decoder Module
#let dec_fig = canvas({
    import draw: *
    
    // Latent input
    rect((-4, 0.25), (-3, 0.75), name: "z", stroke: 2pt + red)
    content("z.center", [$hat(bold(z))_t$])
    
    rect((-1.5, 0.0), (1.5, 1.0), name: "dec", stroke: 2pt + green, fill: green.lighten(90%))
    content("dec.center", [Decoder $d_theta$])
    
    line("z.east", "dec.west", mark: (end: ">"), stroke: 1.5pt + red)
    
    // Output
    rect((3, 0.25), (4, 0.75), name: "ehr", stroke: 2pt + blue)
    content("ehr.center", [$hat(bold(mu))_t$])
    
    line("dec.east", "ehr.west", mark: (end: ">"), stroke: 1.5pt)
    
    // // Original EHR for comparison (to the side)
    // rect((3, 2), (5, 2.6), name: "ehrgt", stroke: 2pt + blue, fill: blue.lighten(95%))
    // content("ehrgt.center", [EHR $bold(mu)_t$])
    
    // // Loss connection
    // line("ehrgt.south", (4, 1.8), (4, -1.5), (1, -1.5), stroke: 1.5pt + orange)
    // line((1, -1.5), "ehr.east", mark: (end: ">"), stroke: 1.5pt + orange)
    
    // // Loss label
    // rect((1.8, -1.7), (3.2, -1.3), stroke: 1.5pt + orange, fill: orange.lighten(90%))
    // content((2.5, -1.5), [$cal(L)_"dec"$])
  })

// Complete System Overview
#let ldm_fig = canvas({
    import draw: *
    
    // Input
    rect((0, 5), (2, 5.6), name: "ehr", stroke: 2pt + blue, fill: blue.lighten(95%))
    content("ehr.center", text(9pt)[EHR $bold(mu)_t$])
    
    // Infection module
    rect((-2, 3.5), (0.5, 4.5), name: "inf", stroke: 2pt + red.lighten(20%), fill: red.lighten(95%))
    content("inf.center", text(8pt)[Infection\nModule])
    
    line("ehr.south", (-1, 5), (-1, 4.5), mark: (end: ">"), stroke: 1.5pt)
    
    rect((-2, 2.8), (0.5, 3.2), name: "iout", stroke: 1.5pt + red)
    content("iout.center", text(7pt)[$tilde(I)_t$])
    line("inf.south", "iout.north", mark: (end: ">"), stroke: 1pt)
    
    // SOFA module
    rect((2.5, 3.5), (5.5, 4.5), name: "sofa", stroke: 2pt + purple.lighten(20%), fill: purple.lighten(95%))
    content("sofa.center", text(8pt)[SOFA\nPredictor])
    
    line("ehr.south", (1, 5), (4, 4.5), mark: (end: ">"), stroke: 1.5pt)
    
    rect((2.5, 2.8), (5.5, 3.2), name: "zout", stroke: 1.5pt + purple)
    content("zout.center", text(7pt)[$hat(bold(z))_t, hat(O)_t$])
    line("sofa.south", "zout.north", mark: (end: ">"), stroke: 1pt)
    
    // Heuristic combination
    rect((0.5, 1.2), (4.5, 2.2), name: "heur", stroke: 2pt + orange, fill: orange.lighten(95%))
    content("heur.center", text(8pt)[Heuristic Risk\n$tilde(A)_t = f(tilde(I), hat(O)_(0:t))$])
    
    line("iout.south", (0, 2.8), (0, 2.2), mark: (end: ">"), stroke: 1pt)
    line("zout.south", (5, 2.8), (5, 2.2), mark: (end: ">"), stroke: 1pt)
    
    // Output
    rect((1, 0.2), (4, 0.8), name: "risk", stroke: 3pt + green, fill: green.lighten(90%))
    content("risk.center", text(9pt, weight: "bold")[Sepsis Risk $tilde(A)_t$])
    
    line("heur.south", "risk.north", mark: (end: ">"), stroke: 2pt)
    
  })
