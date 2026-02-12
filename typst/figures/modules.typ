#import "@preview/cetz:0.4.2": canvas, draw, tree
#import "helper.typ": cmbeta, cmsigma

#set page(width: auto, height: auto, margin: 8pt, fill: none)
// Infection Indicator Module
#let inf_fig = canvas({
    import draw: *

    let inf_c = olive
    rect((-6, 0), (-4, 0.8), name: "ehr-1", stroke: (paint: blue.lighten(1%), dash: "dotted"), fill: blue.lighten(95%))
    content("ehr-1.center", text(fill: gray.darken(20%))[EHR $bold(mu)_(t-1)$])
    
    rect((-6, -1.5), (-4, -0.5), name: "rnn-1", stroke: (paint: inf_c.lighten(1%), dash: "dotted"), fill: inf_c.lighten(95%))
    content("rnn-1.center", text(fill: gray.darken(20%))[RNN $f_theta_f$])
    
    rect((-6, -2.7), (-4, -2), name: "inf-1", stroke: (paint: red.lighten(1%), dash: "dotted"))
    content("inf-1.center", text(fill: gray.darken(20%))[$tilde(I)_(t-1)$])
    
    line("ehr-1.south", "rnn-1.north", mark: (end: ">"), stroke: (paint: gray.lighten(1%), dash: "dotted"))
    line("rnn-1.south", "inf-1.north", mark: (end: ">"), stroke: (paint: gray.lighten(1%), dash: "dotted"))
    
    rect((-1, 0), (1, 0.8), name: "ehr", stroke: 2pt + blue)
    content("ehr.center", [EHR $bold(mu)_t$])
    
    rect((-3.6, -1.3), (-1.4, -0.7), name: "hprev", stroke: 1pt + gray)
    content("hprev.center", [$bold(h)^f_(t-1) in RR^(H_f)$])
    
    rect((-1, -1.5), (1, -0.5), name: "rnn", stroke: 2pt + inf_c, fill: inf_c.lighten(90%))
    content("rnn.center", [RNN $f_theta_f$])
    
    line("ehr.south", "rnn.north", mark: (end: ">"), stroke: 1.5pt)
    
    rect((-1, -2.7), (1, -2), name: "inf", stroke: 2pt + red)
    content("inf.center", [$tilde(I)_t in (0,1)$])
    
    rect((2, -1.3), (3, -0.7), name: "h", stroke: 1pt + gray)
    content("h.center", [$bold(h)^f_t$])
    
    line("rnn.south", "inf.north", mark: (end: ">"), stroke: 1.5pt)
    line("rnn.east", "h.west", mark: (end: ">"), stroke: 1pt + gray)
    line("hprev.east", "rnn.west", mark: (end: ">"), stroke: 1pt + gray)
    
    line("rnn-1.east", "hprev.west", mark: (end: ">"), stroke: (paint: gray, dash: "dotted"))
    
    rect((4, 0), (6, 0.8), name: "ehr+1", stroke: (paint: blue.lighten(1%), dash: "dotted"), fill: blue.lighten(95%))
    content("ehr+1.center", text(fill: gray.darken(20%))[EHR $bold(mu)_(t+1)$])
    
    rect((4, -1.5), (6, -0.5), name: "rnn+1", stroke: (paint: inf_c.lighten(1%), dash: "dotted"), fill: inf_c.lighten(95%))
    content("rnn+1.center", text(fill: gray.darken(20%))[RNN $f_theta_f$])
    
    rect((4, -2.7), (6, -2), name: "inf+1", stroke: (paint: red.lighten(1%), dash: "dotted"))
    content("inf+1.center", text(fill: gray.darken(20%))[$tilde(I)_(t+1)$])
    
    line("ehr+1.south", "rnn+1.north", mark: (end: ">"), stroke: (paint: gray.lighten(1%), dash: "dotted"))
    line("rnn+1.south", "inf+1.north", mark: (end: ">"), stroke: (paint: gray.lighten(1%), dash: "dotted"))
    line((3, -1), (4, -1), mark: (end: ">"), stroke: (paint: gray, dash: "dotted"))
  })

#let sofa_fig = canvas({
    import draw: *
    let sofa_c = olive

    let dh = 0.3
    rect((-5, 2), (-3, 2.8), name: "ehr0", stroke: 2pt + blue)
    content("ehr0.center", [EHR $bold(mu)_1$])
    
    rect((-5.2, 0.5), (-2.8, 1.5), name: "enc", stroke: 2pt + sofa_c, fill: sofa_c.lighten(90%))
    content("enc.center", [Encoder $g^e_theta^e_g$])
    
    line("ehr0.south", "enc.north", mark: (end: ">"), stroke: 1pt)
    
    rect((-5, -1.75), (-3, -1.15), name: "z0", stroke: 2pt + red)
    content("z0.center", [$hat(bold(z))^"raw"_1 in RR^2$])
    
    rect((rel: (0, dh), to: (-2.1, 0.65)), (rel: (0, dh), to: (-0.25, 1.35)), name: "h0", stroke: 1pt + gray)
    content("h0.center", [$bold(h)^g_1 in RR^(H_g)$])
    
    line("enc.south", "z0.north", mark: (end: ">"), stroke: 1pt)
    line((rel: (0, dh), to: "enc.east"), "h0.west", mark: (end: ">"), stroke: 1pt + gray)
    
    rect((0.75, 0.5), (3.25, 1.5), name: "rnn0", stroke: 2pt + sofa_c, fill: sofa_c.lighten(90%))
    content("rnn0.center", [RNN $g^r_theta^r_g$])
    line("h0.east", (rel: (0, dh), to: "rnn0.west"), stroke: 1pt +  gray, mark: (end: ">"))
    
    rect((1, 2), (3, 2.8), name: "ehr1", stroke: 2pt + blue)
    content("ehr1.center", [EHR $bold(mu)_2$])
    line("ehr1.south", "rnn0.north", mark: (end: ">"), stroke: 1pt)
    
    rect((rel: (0, dh), to: (4.2, 0.65)), (rel: (0, dh), to: (5.95, 1.35)), name: "h1", stroke: 1pt + gray)
    content("h1.center", [$bold(h)^g_2$])
    line((rel: (0, dh), to:"rnn0.east"), "h1.west", mark: (end: ">"), stroke: 1pt + gray)


    // line("z0.east", (rel: (0, -0.4), to: "rnn0.west"), mark: (end: ">"))
    bezier((rel: (1, 0), to: "z0.east"), (rel: (0, -dh), to: "rnn0.west"), (-2, 1), mark: (end: ">"), stroke: 1pt)
    
    rect((1, -0.125), (3, -0.625), name: "dz", stroke: 2pt + orange)
    content("dz.center", [$Delta hat(bold(z))^"raw"_2$])
    
    rect((3, -1.75), (5, -1.25), name: "z1", stroke: 2pt + red)
    content("z1.center", [$hat(bold(z))^"raw"_2$])
    
    line("rnn0.south", "dz.north", mark: (end: ">"), stroke: 1pt)
    
    // Addition
    circle((2, -1.5), radius: 0.3, stroke: 1.5pt)
    content((2, -1.5), text(10pt, weight: "bold")[+])
    line("z0.east", (1.7, -1.5), mark: (end: ">"), stroke: 1pt)
    line("dz.south", (2.0, -1.2), mark: (end: ">"), stroke: 1pt)
    line((2.3, -1.5), "z1.west", mark: (end: ">"), stroke: 1pt)
    
    // t=2 step
    rect((6.75, 0.5), (9.25, 1.5), name: "rnn1", stroke: (paint: sofa_c.lighten(1%), dash: "dotted"), fill: sofa_c.lighten(95%))
    content("rnn1.center", text(fill: gray.darken(20%))[RNN $g^r_theta^r_g$])
    
    rect((7, 2), (9, 2.8), name: "ehr2", stroke: (paint: blue.lighten(1%), dash: "dotted"), fill: blue.lighten(95%))
    content("ehr2.center", text(fill: gray.darken(20%))[EHR $bold(mu)_t$])
    line("ehr2.south", "rnn1.north", mark: (end: ">"), stroke: (paint: gray.lighten(1%), dash: "dotted"))
    
    line("rnn1.east", (10, 1.0), mark: (end: ">"), stroke: (paint: gray.lighten(1%), dash: "dotted"))
    
    rect((7, -0.125), (9, -0.625), name: "dz2", stroke: (paint: orange.lighten(1%), dash: "dotted"))
    content("dz2.center", text(fill: gray.darken(20%))[$Delta hat(bold(z))^"raw"_t$])
    
    line("rnn1.south", "dz2.north", mark: (end: ">"), stroke: (paint: gray.lighten(1%), dash: "dotted"))
    
    circle((8, -1.5), radius: 0.3, stroke: (paint: gray.lighten(1%), dash:"dotted"))
    content((8, -1.5), text(10pt, fill: gray.darken(1%))[$+$])
    line("z1.east", (7.7, -1.5), mark: (end: ">"), stroke: (paint: gray.lighten(1%), dash: "dotted"))
    line("dz2.south", (8.0, -1.2), mark: (end: ">"), stroke: (paint: gray.lighten(1%), dash: "dotted"))
    line((8.3, -1.5), (10, -1.5), mark: (end: ">"), stroke: (paint:gray.lighten(1%), dash: "dotted"))
    
    bezier((rel: (0.5, 0), to: "z1.east"), (rel: (0, -dh), to: "rnn1.west"), (5.3, 1), mark: (end: ">"), stroke: (paint: gray.lighten(1%), dash: "dotted"))
    line("h1.east", (rel: (0, dh), to: "rnn1.west"), stroke: (paint: gray.lighten(1%), dash: "dotted"), mark: (end: ">") )
  })

// Decoder Module
#let dec_fig = canvas({
    import draw: *
    let dec_c = olive
    // Latent input
    rect((-4, 0.25), (-3, 0.75), name: "z", stroke: 2pt + red)
    content("z.center", [$hat(bold(z))_t$])
    
    rect((-1.5, 0.0), (1.5, 1.0), name: "dec", stroke: 2pt + dec_c, fill: dec_c.lighten(90%))
    content("dec.center", [Decoder $d_theta_d$])
    
    line("z.east", "dec.west", mark: (end: ">"), stroke: 1pt)
    
    // Output
    rect((3, 0.25), (4, 0.75), name: "ehr", stroke: 2pt + blue)
    content("ehr.center", [$hat(bold(mu))_t$])
    
    line("dec.east", "ehr.west", mark: (end: ">"), stroke: 1pt)
    
  })

// Complete System Overview
#let ldm_fig = canvas({
    import draw: *

    let out_c = maroon
    
    // Input
    rect((0, 5), (2, 5.75), name: "ehr", stroke: 2pt + blue, fill: blue.lighten(95%))
    content("ehr.center", [EHR $bold(mu)_t$])
    
    // Infection module
    let inf_c = olive
    rect((-3.5, 3.0), (0.5, 4), name: "inf", stroke: 2pt + inf_c.lighten(20%), fill: inf_c.lighten(95%))
    content("inf.center", align(center, [Infection Module $f_theta_f$]))

    line((rel:(-1, 0.0), to:"inf.south"), (rel:(-1, -0.5), to:"inf.south"), (rel:(-2.5, -0.5), to:"inf.south"), (rel:(-2.5, 1.5), to:"inf.south"), (rel:(-1, 1.5), to:"inf.south"), (rel:(-1, 0), to:"inf.north"), mark: (end: ">"))
    content((rel: (-1.5, 0.9), to:"inf.north"), align(center, text(size: 10pt)[hidden state $bold(h)^f_(t-1)$]))
    
    line("ehr.south", (rel: (0, .5), to: "inf.north"), "inf.north", mark: (end: ">"), stroke: 1pt)
    
    rect((-3.5, 2.25), (0.5, 1.5), name: "iout", stroke: 1.5pt + out_c)
    content("iout.center", [$tilde(I)_t$])
    line("inf.south", "iout.north", mark: (end: ">"), stroke: 1pt)
    
    // SOFA module
    let sofa_c = olive
    rect((1.5, 3.0), (5.5, 4), name: "sofa", stroke: 2pt + sofa_c.lighten(20%), fill: sofa_c.lighten(95%))
    line((rel:(1, 0.0), to:"sofa.south"), (rel:(1, -0.5), to:"sofa.south"), (rel:(2.5, -0.5), to:"sofa.south"), (rel:(2.5, 1.5), to:"sofa.south"), (rel:(1, 1.5), to:"sofa.south"), (rel:(1, 0), to:"sofa.north"), mark: (end: ">"))
    content((rel: (2.5, 1.1), to:"sofa.north"), align(left, text(size: 10pt)[hidden state $bold(h)^g_(t-1)$\ and previous position $bold(z)^"raw"_(t-1)$]))
    
    content("sofa.center", align(center, [SOFA Module $g_theta_g$]))
    line("ehr.south", (rel: (0, .5), to: "sofa.north"), "sofa.north", mark: (end: ">"), stroke: 1pt)
    
    rect((1.5, 2.25), (5.5, 1.5), name: "zout", stroke: 1.5pt + out_c)
    content("zout.center", [$hat(bold(z))_t$])
    line("sofa.south", "zout.north", mark: (end: ">"), stroke: 1pt)

    rect((1.5, 1.0), (5.5, 0.25), name: "oout", stroke: 1.5pt + out_c)
    content("oout.center", [$hat(O) (bold(hat(z))_t)$])
    line("zout.south", "oout.north", mark: (end: ">"), stroke: 1pt)

    rect((1.5, -0.25), (5.5, -1), name: "aout", stroke: 1.5pt + out_c)
    content("aout.center", text(10pt)[$tilde(A)_t = o_(s,d)(hat(O)_(t-1), hat(O)_t)$])
    line("oout.south", "aout.north", mark: (end: ">"), stroke: 1pt)
    
    rect((6.5, -0.25), (7.5, -1), name: "oin", stroke: 1.5pt + out_c)
    content("oin.center", [$hat(O)_(t-1)$])
    line("oin.west", "aout.east", mark: (end: ">"), stroke: 1pt)

    // Dec
    let dec_c = olive
    rect((7, 2.5), (9.5, 1.25), name: "dec", stroke: 2pt + dec_c, fill: dec_c.lighten(90%))
    content("dec.center", [Decoder $d_theta_d$])
    
    line("zout.east", "dec.west", mark: (end: ">"), stroke: 1pt)
    rect((10, 1.5), (12, 2.25), name: "ehr", stroke: 2pt + blue)
    content("ehr.center", [$hat(bold(mu))_t$])
    
    line("dec.east", "ehr.west", mark: (end: ">"), stroke: 1pt)

    // Output
    rect((-0.25, -1.5), (2.75, -2.75), name: "risk", stroke: 3pt + red, fill: red.lighten(90%))
    content("risk.center", align(center, text(weight: "bold")[Sepsis Risk \ $tilde(S)_t = "CS"(tilde(A)_t) tilde(I)_t$]))

    line("iout.south", (rel: (-1.25, 0), to:"risk.west"), "risk.west", mark: (end: ">"), stroke: 1pt)
    line("aout.south", (rel: (0.75, 0), to:"risk.east"), "risk.east", mark: (end: ">"), stroke: 1pt)
  })

#figure(ldm_fig)
