#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/suiji:0.4.0": *
#import "helper.typ": *
#import draw: circle, content, line, rect

#set text(size: 8pt)

#set page(width: auto, height: auto, margin: 8pt)

#let sick = (0, 0 )
#let data = (3.5, 0)
#let doc = (6, 1.0)
#let diag = (8, 1.0)

#let d = 3.3
#let ol_ = (0, d)
#let or_ = (d, d)
#let ul_ = (0, 0)
#let ur_ = (d, 0)

#let high_fig = canvas({
  content(ul_,  align(center, "EHR") + $bold(mu)=(mu_1,...,mu_n)$, name:"ehr")
  content(ol_, $(cmbeta(beta), cmsigma(sigma))$, name:"coord")
  content(or_, $s^1$, name:"metric")
  content(ur_, [SOFA-Score], name:"sofa")
  
  line((rel:(0, 0.5), to: "ehr"), (rel: (0, -0.2), to: "coord"), name: "flow1", mark: emark)
  line((rel:(0, -0.2), to: "metric"), (rel: (0, 0.2), to: "sofa"), name: "flow2", mark: emark)
  line((rel:(0.5, 0), to: "coord"), (rel: (-0.5, 0), to: "metric"), name: "flow2", mark: emark)
  line((rel:(0.5, 0), to: "ehr"), (rel: (-0.8, 0), to: "sofa"), name: "flow3", mark: emark)
  // content(sick, image("sick.svg"), name: "sick")
  // content((rel: (0, .5), to: "sick"), [ICU Patient])
  
  // line((rel:(0.5, 0), to: sick), (rel: (-0.5, 0), to: data), name: "flow1", mark: emark)
  // content((rel: (0, -.3), to: "flow1"), [Take Measurements])
  
  // content(data, image("data.svg"), name: "data")
  // content((rel: (0, .6), to: "data"), "  " + $bold(mu)$ + "\nEHR")
  
  // content(doc, image("steto.svg"), name: "doc")
  // content((rel: (0, .5), to: "doc"), [Medical Expert])
  // line((rel:(0.5, 0), to: data), (rel: (-0.5, 0), to: doc), name: "flow2", mark: emark)
  // content((rel: (0, -.3), to: "flow2"), [Traditional], angle:37deg)

  
  // content(diag, image("diag.svg"), name: "diag")
  // content((rel: (0, .6), to: "diag"), "Diagnosis\nSepsis/No Sepsis")
})
#figure(high_fig)


