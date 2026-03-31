#import "@preview/cetz:0.4.2": canvas, draw, tree
#import "helper.typ": cmbeta, cmsigma

#set page(width: auto, height: auto, margin: 2pt, fill: none)

#import "@preview/cetz:0.4.0": canvas, draw

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  DESIGN TOKENS                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#let font-main  = "New Computer Modern"
#let sz-title   = 9.5pt
#let sz-body    = 8pt
#let sz-sub     = 7pt
#let sz-annot   = 6pt
#let sz-grp     = 8pt

#let page-w     = auto
#let page-h     = auto
#let margin-x   = 5mm
#let margin-y   = 4mm

#let c-ink      = rgb("#111111")
#let c-dim      = rgb("#555555")
#let c-faint    = rgb("#555555")
#let c-mod-bg   = rgb("#f4f4f4")
#let c-grp-bg   = rgb("#fafafa")
#let c-grp-str  = rgb("#c0c0c0")
#let c-risk-str = rgb("#922222")

#let sw-heavy   = 1.4pt
#let sw-grp     = 0.6pt
#let sw-mod     = 0.75pt
#let sw-out     = 0.55pt
#let sw-arr     = 0.75pt
#let sw-loop    = 0.5pt
#let sw-dec     = 0.55pt

// ── Layout ─────────────────────────────────────────────────────────────────
#let inf-x      = 4.7
#let org-x      = 10.3     // closer: gap = 6.6cm
#let cx         = (inf-x + org-x) / 2

#let ehr-y      = 9.6
#let module-y   = 7.5
#let z-y        = 6.6
#let tildes-y   = 4.40
#let a-y        = 3.60
#let risk-y     = 2.25

#let module-w   = 3.1
#let module-h   = 0.65
#let box-w      = 4.0
#let box-a-w    = 5.75
#let box-top    = 8.75
#let box-i-bot  = 6.00
#let box-o-bot  = 3.0
#let ow         = 2.55
#let oh         = 0.46
#let rx-box     = 0.07
#let rx-grp     = 0.20

#let dec-x      = 14.20
#let dec-y      = z-y
#let dec-w      = 1.55
#let dec-h      = 0.50

// ── Lookup inset: sits in the gap between ẑ and Ô, right of org branch ─────
// It lives in a column to the right, connected by straight horizontal elbow
#let lu-cx      = org-x + module-w/2   // centre of lookup column
#let lu-top     = z-y + oh / 2             // top: flush with ẑ box bottom
#let lu-bot     = tildes-y - oh / 2             // bottom: flush with Ô box top
#let lu-mid     = (lu-top + lu-bot) / 2
#let lu-h-cm    = (lu-top - lu-bot) * 1cm  // height as length
#let lu-w-cm    = 3.2cm                    // fixed width

// Page-space position of the lookup inset (for #place)
// canvas-h = page-h - 2*margin-y
#let cv-h       = 9.3cm - 2 * margin-y
#let lu-page-x  = margin-x + lu-cx * 1cm - lu-w-cm / 2
#let lu-page-y  = margin-y + cv-h - lu-top * 1cm   // top edge in page coords

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  FIGURE                                                                  ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#set page(width: page-w, height: page-h,
          margin: (x: margin-x, y: margin-y), fill: white)
#set text(font: font-main, size: sz-body, fill: c-ink)

// ── Main canvas ─────────────────────────────────────────────────────────────
#let ldm_fig = canvas({
  import draw: *

  let dash3 = (array: (3pt, 3pt), phase: 0pt)

  let arr(a, b, clr: c-ink) = line(a, b,
    stroke: sw-arr + clr,
    mark: (end: ">", size: .17, fi20ll: clr))

  let poly(pts, clr: c-ink) = {
    let n = pts.len()
    for i in range(n - 2) {
      line(pts.at(i), pts.at(i + 1), stroke: sw-arr + clr)
    }
    line(pts.at(n - 2), pts.at(n - 1),
      stroke: sw-arr + clr,
      mark: (end: ">", size: .17, fill: clr))
  }

  // ── Group brackets ─────────────────────────────────────────────────────
    rect((inf-x - box-w/2, box-i-bot), (inf-x + box-w/2, box-top),
      fill: c-grp-bg, stroke: sw-grp + c-grp-str, radius: rx-grp)
    rect((org-x - box-w/2, box-o-bot), (org-x + box-w/2, box-top),
      fill: c-grp-bg, stroke: sw-grp + c-grp-str, radius: rx-grp)
  content((inf-x, box-top - .35),
    text(size: sz-grp, weight: "semibold")[Infection Module], anchor: "south")
  content((org-x, box-top - .35),
    text(size: sz-grp, weight: "semibold")[Organ-Dysfunction Module], anchor: "south")

  // ── Lookup inset border + label + connector arrows (in cetz coords) ──────
  // Label
content((lu-cx -2.9, lu-top - 1.4), text(size: sz-annot, fill: c-dim)[Latent\ Lookup], anchor: "south")

  // FSQ dots drawn on top of phase map
  let fpad = 0.23
  let frad = 0.055
  let gx0 = lu-cx - 8.5 * fpad
  let gy0 = lu-mid - 1.5 * fpad
  for gxi in range(5) {
    for gyi in range(5) {
      circle((gx0 + gxi * fpad, gy0 + gyi * fpad),
        radius: frad,
        fill: gradient.linear(..color.map.viridis).sample(calc.sin(gxi - 1.5) * 50% + calc.sin(gyi - 1.5) * 50%),
        stroke: none)
    }
  }
  // ẑ point (slightly off-grid, red)
  let zhx = gx0 + 1 * fpad + 0.07
  let zhy = gy0 + 2 * fpad - 0.05
  // soft kernel glow
  // 3×3 neighbourhood box
  rect((gx0 + 0 * fpad - fpad * 0.55, gy0 + 1 * fpad - fpad * 0.55),
       (gx0 + 2 * fpad + fpad * 0.55, gy0 + 3 * fpad + fpad * 0.55),
    fill: none,
    stroke: (paint: rgb("#cc2200"), thickness: .7pt, dash: "dotted"),
    radius: 0.03)
  // ẑ dot
  circle((zhx, zhy), radius: frad + 0.01, fill: rgb("#cc2200"), stroke: none)

  // Connection: ẑ box → lookup inset (right side, horizontal then into box)
  line((org-x, z-y),
       (org-x, lu-top - (lu-top - lu-bot) * 0.25),
    stroke: sw-arr + c-ink,
    mark: (end: none, size: .17, fill: c-ink))

  // Connection: lookup inset → Ô box (out of box right into Ô)
  line((org-x, lu-bot + (lu-top - lu-bot) * 0.25 + 0.1),
       (org-x, tildes-y +0.23),
    stroke: sw-arr + c-ink,
    mark: (end: ">", size: .17))

  // ── EHR ──────────────────────────────────────────────────────────────────
  rect((cx - 1.25, ehr-y - .26), (cx + 1.25, ehr-y + .26),
    fill: white, stroke: sw-mod + c-ink, radius: rx-box)
  content((cx, ehr-y),
    text(weight: "semibold")[$bold(mu)_t$] +
    text(fill: c-dim)[#h(.5em) Observation])

  arr((cx, ehr-y - .26), (inf-x, box-top))
  arr((cx, ehr-y - .26), (org-x, box-top))

  // ── Infection module ──────────────────────────────────────────────────────
  rect((inf-x - module-w/2, module-y - module-h/2), (inf-x + module-w/2, module-y + module-h/2),
    fill: c-mod-bg,
    stroke: (paint: c-ink, thickness: sw-mod, dash: "dashed"), radius: rx-box)
  content((inf-x, module-y),
    text(fill: c-dim, size: sz-sub)[$f_theta$ #h(.4em) GRU])

  let lx = inf-x - module-w/2 - .38
  line((inf-x - .55, module-y - module-h/2),
       (inf-x - .75, module-y - module-h/2 - .22),
       (lx,          module-y - module-h/2 - .22),
       (lx,          module-y + module-h/2 + .22),
       (inf-x - .75, module-y + module-h/2 + .22),
       (inf-x - .55, module-y + module-h/2),
       stroke: sw-loop + c-faint,
       mark: (end: ">", size: .12, fill: c-faint))
  content((lx + .3, module-y + .6),
    align(top, text(size: sz-annot, fill: c-faint)[$bold(h)^f_(t-1)$]),
    anchor: "south")

  let iy = module-y - module-h/2 - .58
  rect((inf-x - ow/2, iy - oh/2), (inf-x + ow/2, iy + oh/2),
    fill: white, stroke: sw-out + c-ink, radius: rx-box)
  content((inf-x, iy), $p_theta(I_t | bold(mu)_(1:t))$)
  arr((inf-x, module-y - module-h/2), (inf-x, iy + oh/2))

  // ── Organ module ──────────────────────────────────────────────────────────
  rect((org-x - module-w/2, module-y - module-h/2), (org-x + module-w/2, module-y + module-h/2),
    fill: c-mod-bg,
    stroke: (paint: c-ink, thickness: sw-mod, dash: "dashed"), radius: rx-box)
  content((org-x, module-y),
    text(fill: c-dim, size: sz-sub)[$g_theta, q_theta$ #h(.4em) Encoder + GRU])

  let rrx = org-x + module-w/2 + .38
  line((org-x + .55, module-y - module-h/2),
       (org-x + .75, module-y - module-h/2 - .22),
       (rrx,         module-y - module-h/2 - .22),
       (rrx,         module-y + module-h/2 + .22),
       (org-x + .75, module-y + module-h/2 + .22),
       (org-x + .55, module-y + module-h/2),
       stroke: sw-loop + c-faint,
       mark: (end: ">", size: .12, fill: c-faint))
  content((rrx - .5, module-y+ .6),
    align(left, text(size: sz-annot, fill: c-faint)[$bold(h)^g_(t-1), bold(z)_(t-1)$]),
    anchor: "south")

  // ẑ_t
  rect((org-x - ow/2, z-y - oh/2), (org-x + ow/2, z-y + oh/2),
    fill: white, stroke: sw-out + c-ink, radius: rx-box)
  content((org-x, z-y), $(beta_t, sigma_t) in RR^2$)
  arr((org-x, module-y - module-h/2), (org-x, z-y + oh/2))

  // Ô_t
  rect((org-x - ow/2, tildes-y - oh/2), (org-x + ow/2, tildes-y + oh/2),
    fill: white, stroke: sw-out + c-ink, radius: rx-box)
  content((org-x, tildes-y), $tilde(s)^1(beta_t, sigma_t)$)
  // no direct arrow here — the lookup inset IS the connection

  // p(A_t)
  rect((org-x - ow/2, a-y - oh/2), (org-x + ow/2, a-y + oh/2),
    fill: white, stroke: sw-out + c-ink, radius: rx-box)
  content((org-x, a-y), $p_theta(A_t | I_t, bold(mu)_(1:t))$)
  arr((org-x, tildes-y - oh/2), (org-x, a-y + oh/2))

  // Ô_{t-1}
  let prevx = org-x - ow/2 - 1.0
  rect((prevx - .44, a-y - oh/2), (prevx + .44, a-y + oh/2),
    fill: white, stroke: sw-dec + c-faint, radius: rx-box)
  content((prevx, a-y), text(fill: c-dim)[$tilde(s)_(t-1)$])
  arr((prevx + .44, a-y), (org-x - ow/2, a-y), clr: c-faint)

  // ── Decoder ───────────────────────────────────────────────────────────────
  rect((dec-x - dec-w/2, dec-y - dec-h/2), (dec-x + dec-w/2, dec-y + dec-h/2),
    fill: white,
    stroke: (paint: c-faint, thickness: sw-dec, dash: dash3), radius: rx-box)
  content((dec-x, dec-y),
    text(size: sz-sub + .5pt, fill: c-dim)[Decoder $d_theta$])
  content((dec-x -1.4, dec-y - .25),
    text(size: sz-annot, fill: c-faint, style: "italic")[train only])

  let muy = dec-y + dec-h* 5/2 - .48
  rect((dec-x - 1.2, muy - .20), (dec-x + 1.2, muy + .20),
    fill: white, stroke: sw-dec + c-faint, radius: rx-box)
  content((dec-x, muy), text(fill: c-faint)[$hat(bold(mu))_t$ Reconstruction])

  // Route decoder elbow ABOVE the lookup inset to avoid crossing it
  let dec-elbow-y = lu-top + 0.25
  line((org-x + ow/2, z-y), (dec-x - dec-w/2, dec-y), mark: (end: ">",size: .13, fill: c-ink))

  line((dec-x, dec-y + dec-h/2), (dec-x, muy - .20),
    stroke: (paint: c-faint, thickness: sw-dec, dash: dash3),
    mark: (end: ">", size: .13, fill: c-faint))

  // ── Sepsis Risk ───────────────────────────────────────────────────────────
  rect((cx - 2.4, risk-y - .54), (cx + 2.4, risk-y + .54), stroke: sw-heavy + c-risk-str, radius: rx-box)
  content((cx, risk-y + .19),
    text(weight: "bold", size: sz-title, fill: c-risk-str)[Sepsis Risk])
  content((cx, risk-y - .19),
    $p_theta(S_t | bold(mu)_(1:t)) = p_theta(A_t | I_t, bold(mu)_(1:t)) dot p_theta(I_t | bold(mu)_(1:t))$)

  poly(((inf-x, iy - oh/2), (inf-x, risk-y), (cx - 2.4, risk-y)))
  poly(((org-x, a-y - oh/2), (org-x, risk-y), (cx + 2.4, risk-y)))
})

#figure(ldm_fig)
