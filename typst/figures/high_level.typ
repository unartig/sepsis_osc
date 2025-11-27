#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/suiji:0.4.0": *
#import "helper.typ": *
#import draw: circle, content, line, rect, bezier, arc

// #set text(size: 8pt)

#set page(width: auto, height: auto, margin: 8pt)

#let d = 3.3
#let data = (0, 4)
#let coord = (2.5, 6)
#let zzz = (9, 6)
#let std = (14, 6)
#let ai = (18, 3)
#let aa = (18, 5)
#let sep = (21, 4)
#let mark-style = (symbol: "stealth", fill: black)
#let high_fig = canvas({
  content(data,  align(center, "Patient Data (EHR)\n" + $(mu_1,...,mu_n)_(t=0)$), name:"ehr")
  content(coord, align(center, "DNM-Parameter and\nTemporal Information\n" + $(z_(0,cmbeta(beta)), z_(0,cmsigma(sigma))), bold(h)_0$) + text("\n ", size: 2pt), name:"coord")
  content(zzz, align(center, "Latent Trajectory\n" + $bold(z)_(0:T)$), name:"zzz")
  content(std, align(center, "Synchronicity Measure/\nEstimated SOFA-Scores\n" + $s^1_(0:T) tilde hat(O)_(0:T)$ + "\n "), name:"metric")
  content(aa, align(center, $tilde(A)$))
  content(ai, align(center, $tilde(I)$))
  content(sep, align(center, "Sepsis-Prediction\n" + $tilde(S)$ + "\n "), name:"sofa")
  
  // line((rel:(0, 0.5), to: "ehr"), (rel: (0, 0.0), to: "coord"), name: "flow1", mark: emark)

  bezier((rel: (1.25, 0), to: data), (rel: (0, -0.65), to: coord), (2.5, 4), mark: emark)
  content((rel: (1.5, -0.6), to: midp(coord, data)), "Encoding\n" + $e_theta (bold(mu)_0)$, anchor:"west")
  
  line((rel: (1.5, -0.2), to: coord), (rel: (-0.7, -0.2), to: zzz), name:"auto", mark:emark)
  arc((rel: (0, 0.3), to: midp(coord, zzz)),
      start: 90deg,
      stop:(360+90) * 1deg,
      radius: .5,
      anchor: "origin",
      mark: (
        end: (
            (pos: 0%, ..mark-style),
          )
        )
      )
  content((rel: (-0.7, 1.5), to: midp(coord, zzz)), "Auto-Regressive Latent Prediction\n" +  $bold(z)_t, bold(h)_t = r_theta (bold(z)_(t-1), bold(h)_(t-1))$, anchor:"west")
  
  line((rel: (0.7, -0.2), to: zzz), (rel: (-1.5, -0.2), to: std), name:"int", mark:emark)
  content((rel: (-0.5, -0.7), to: midp(zzz, std)), "Integrate DNM-System", anchor:"north")

  bezier((rel: (2.5, 0), to: std), (rel: (0, 0.5), to:aa), (18, 6), mark: emark)
  content((rel: (4.5, 1.5), to: midp(std, aa)), "Estimate Organ Failure Risk\n" + $o_(s,d)(hat(O)_(0:T))$, anchor:"north")
  bezier((rel: (0, -0.5), to:aa), (rel: sep, to:(-1.5, 0)), (18, 4), mark: emark)

  // INF
  bezier((1.25, 4), (2.5, 2.99), (2.5, 4),name: "ehr-co")
  bezier((2.5, 3), (3.001, 2), (2.5, 2))
  line((3, 2), (15.5, 2), name: "flow1")
  bezier((15.5, 2), (rel: (0, -0.5), to:ai), (18, 2), mark: emark)
  content(midp((3, 2.7), (15.5, 2.7)), "Estimate Suspected-Infection Risk\n" + $f_theta (bold(mu)_0)$)
  bezier((rel: (0, +0.5), to:ai), (rel: sep, to:(-1.5, 0)), (18, 4), mark: emark)

  
})
#figure(high_fig)


