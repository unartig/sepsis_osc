#import "@preview/cetz:0.4.2": canvas, draw
#import draw: bezier, circle, content, line
#import "helper.typ": *

#set page(width: auto, height: auto, margin: 8pt, fill: none)
// #set text(font: "poppins", size: 8pt)

#let node-radius = 0.4
#let enc_offset = 0.5
#let for_yoffset = 2.8
#let for_xoffset = 7
#let ode_scale = 0.3
#let ode_yoffset = 2
#let ode_xoffset = 16
#let sofa_xoffset = 9
#let title_offset = 4.5
#let l1pos = (ode_xoffset, 1.2 + ode_yoffset)
#let l2pos = (ode_xoffset, -1.2 + ode_yoffset)
#let nc = rgb("#eee")
#let arrow-style = (
  stroke: .5pt + black,
  mark: (
    start: "stealth",
    end: "stealth",
    fill: fuchsia,
    scale: .4,
    offset: 0.03,
  ),
)
#let descr_size = 14pt

#let mred(x) = text(fill: red, $#x$)

// ============================================================
// HELPER FUNCTIONS - ENCODER
// ============================================================

#let draw-encoder-network(show-labels: true, show-connections: true) = {
  let layers_enc = (
    // (x-pos, neuron-count, fill-color, label, y-offset, extra)
    (
      0,
      5,
      nc,
      ($mu_1$, " ... ", " ... ", " ... ", $mu_n$),
      3 + enc_offset,
      none,
    ),
    (
      2,
      4,
      nc,
      (" ... ", " ... ", " ... ", " ... "),
      2.7 + enc_offset,
      none,
    ),
    (
      4,
      2,
      nc,
      ($cmbeta(beta_0)$, $cmsigma(sigma_0)$),
      2.0 + enc_offset,
      none,
    ),
  )

  // Draw title
  if show-labels {
    content(
      (layers_enc.at(1).at(0), title_offset),
      align(
        text("Encoder\n" + $cal(N)^"Enc"_(theta_f)$, size: 16pt),
        center,
      ),
      anchor: "south",
      padding: .3em,
    )
  }

  // Draw connections between layers
  if show-connections {
    for idx in range(layers_enc.len() - 1) {
      let (x1, n1, _, _, y1, _) = layers_enc.at(idx)
      let (x2, n2, _, _, y2, _) = layers_enc.at(idx + 1)
      connect-layers(x1, n1, x2, n2, y1, y2)
    }
  }

  // Draw layer labels
  if show-labels {
    content(
      (layers_enc.at(0).at(0), layers_enc.at(0).at(-2) + .7),
      align(center,text(descr_size)[Input Layer])
    )
    content(
      (layers_enc.at(2).at(0) +.1, layers_enc.at(0).at(-2) + .4),
      align(center, text(size: descr_size)[Latent\ Representation])
    )
  }

  // Draw neurons
  draw-neurons(layers_enc)
}

// ============================================================
// HELPER FUNCTIONS - LATENT DYNAMICS
// ============================================================

#let draw-latent-predictor(show-labels: true, show-connections: true) = {
  let layers_for = (
    (
      0 + for_xoffset,
      2,
      nc,
      ($cmbeta(beta_t)$, $cmsigma(sigma_t)$),
      0 + for_yoffset,
      none,
    ),
    (
      1.5 + for_xoffset,
      3,
      nc,
      (" ... ", " ... ", " ... "),
      .5 + for_yoffset,
      none,
    ),
    (
      3 + for_xoffset,
      2,
      nc,
      ($cmbeta(beta_(t+1))$, $cmsigma(sigma_(t+1))$),
      0 + for_yoffset,
      true,
    ),
  )

  // Draw connections
  if show-connections {
    for idx in range(layers_for.len() - 1) {
      let (x1, n1, _, _, y1, _) = layers_for.at(idx)
      let (x2, n2, _, _, y2, _) = layers_for.at(idx + 1)
      connect-layers(x1, n1, x2, n2, y1, y2)
    }
  }

  // Draw neurons
  draw-neurons(layers_for)

  // Draw title
  if show-labels {
    content(
      (layers_for.at(1).at(0), title_offset),
      align(
        text("Latent Dynamics\n" + $cal(N)^"RNN"_(theta_g)$, size: 16pt),
        center,
      ),
      anchor: "south",
      padding: .3em,
    )
  }
}

// ============================================================
// HELPER FUNCTIONS - CONNECTIONS
// ============================================================

#let draw-encoder-predictor-connections() = {
  let layers_enc_ref = (
    (0, 5, nc, ($mu_1$, " ... ", " ... ", " ... ", $mu_n$), 3 + enc_offset, none),
    (2, 4, nc, (" ... ", " ... ", " ... ", " ... "), 2.7 + enc_offset, none),
    (4, 2, nc, ($cmbeta(beta_0)$, $cmsigma(sigma_0)$), 2.3 + enc_offset, none),
  )
  
  let layers_for_ref = (
    (0 + for_xoffset, 2, nc, ($cmbeta(beta_t)$, $cmsigma(sigma_t)$), 0 + for_yoffset, none),
    (1.5 + for_xoffset, 3, nc, (" ... ", " ... ", " ... "), 0 + for_yoffset, none),
    (3 + for_xoffset, 2, nc, ($cmbeta(beta_(t+1))$, $cmsigma(sigma_(t+1))$), 0 + for_yoffset, true),
  )

  // Line from encoder to predictor
  line(
    (layers_enc_ref.at(2).at(0) + .5, for_yoffset - .8),
    (layers_for_ref.at(0).at(0) - .5, ode_yoffset),
    mark: emark,
  )

  // Line to parametrize
  line(
    (layers_for_ref.at(2).at(0) + .5, for_yoffset - .8),
    (ode_xoffset - 3, ode_yoffset),
    mark: emark,
    name: "param",
  )

  content("param", text(descr_size)[Parametrize], anchor: "south", padding: .3em)

  // Auto-regressive feedback
  let midlt = midp(
    (layers_enc_ref.at(2).at(0) + .5, for_yoffset - .8),
    (layers_for_ref.at(0).at(0) - .5, ode_yoffset),
  )

  let midparam = midp(
    (layers_for_ref.at(2).at(0) + .5, for_yoffset - .8),
    (ode_xoffset - 3, ode_yoffset),
  )

  let midfor = (layers_for_ref.at(1).at(0), for_yoffset - 2.5)

  line(midfor, (midfor, "-|", midlt), midlt, mark: emark)
  line(midparam, (midparam, "|-", midfor), midfor)

  content(midfor, text(size: descr_size)[Auto regressive], anchor: "north", padding: .3em)
}

// ============================================================
// HELPER FUNCTIONS - DYNAMIC NETWORK MODEL
// ============================================================

#let draw-dynamic-network-model(show-labels: true,
                                show-annotations: true,
                                show-pkappa: true,
                                show-immune: true,
                                show-ikappa: true,
                                show-sigma: true,
                                ) = {
  // Parenchymal layer positions
  let p1 = cadd(l1pos, (2.5 * ode_scale, 2.7 * ode_scale))
  let p2 = cadd(l1pos, (8.0 * ode_scale, 0.5 * ode_scale))
  let p3 = cadd(l1pos, (4.0 * ode_scale, -2.2 * ode_scale))
  let p4 = cadd(l1pos, (-4.0 * ode_scale, -2.2 * ode_scale))
  let p5 = cadd(l1pos, (-8.0 * ode_scale, 0.5 * ode_scale))
  let p6 = cadd(l1pos, (-2.5 * ode_scale, 2.7 * ode_scale))

  // Immune layer positions
  let i1 = cadd(l2pos, (2.5 * ode_scale, 2.7 * ode_scale))
  let i2 = cadd(l2pos, (8.0 * ode_scale, 0.5 * ode_scale))
  let i3 = cadd(l2pos, (4.0 * ode_scale, -2.2 * ode_scale))
  let i4 = cadd(l2pos, (-4.0 * ode_scale, -2.2 * ode_scale))
  let i5 = cadd(l2pos, (-8.0 * ode_scale, 0.5 * ode_scale))
  let i6 = cadd(l2pos, (-2.5 * ode_scale, 2.7 * ode_scale))

  // Draw labels
  if show-labels {
    content(
      (midp(p1, p6).at(0), title_offset + 0.3),
      align(
        text("Dynamic Network Model (DNM)\n ", size: 16pt), 
        center,
      ),
      anchor: "south",
      padding: .3em,
    )

  // Draw neurons
    content(
      p1,
      text("Parenchymal Layer Cells", stroke: blue + .01pt),
      anchor: "west",
      padding: 2em,
    )
  }
  let layer1 = (p1, p2, p3, p4, p5, p6)
  // let layer1 = (p1, p2, p3)
  for pcoord in layer1 {
    neuron(pcoord, fill: blue, label: " ", s: none, st: none)
  }
  let curves = none
  lines-between(layer1, node-radius: node-radius, c: if show-pkappa {purple} else {black}, r: show-pkappa)
  if show-pkappa {
    if show-annotations {
      content(
        midp(p5, p6),
        text($cmp(kappa^1_(i j))$, size: 12pt),
        anchor: "south",
        padding: 0.5em,
      )
    }
  }

  if show-immune {
    if show-labels {
      content(
        i3,
        text("Immune Layer Cells", stroke: olive + .01pt),
        anchor: "west",
        padding: 2em,
      )
    }
    let layer2 = (i1, i2, i3, i4, i5, i6)
    for pcoord in layer2 {
      neuron(pcoord, fill: olive, label: " ", s: none, st: none)
    }

    lines-between(layer2, node-radius: node-radius, c: if show-ikappa {fuchsia} else {black}, r: show-ikappa)

    if show-ikappa {
      if show-annotations {
        content(
          midp(i3, i4),
          text($cmpp(kappa^2_(i j))$, size: 12pt),
          anchor: "north",
          padding: 0.5em,
        )
      }
    }
    if show-annotations {
      content(
        i5,
        text($cmg(phi^2_j)$, size: 12pt),
        anchor: "north-east",
        padding: 0.4em,
      )
    }
  }

  if show-sigma {
    // Draw dashed connections between layers
    dashed(p1, i1, node-radius: node-radius)
    dashed(p2, i2, node-radius: node-radius)
    dashed(p3, i3, node-radius: node-radius)
    dashed(p4, i4, node-radius: node-radius)
    dashed(p5, i5, name: "dash", node-radius: node-radius)
    dashed(p6, i6, node-radius: node-radius)
    if show-annotations {
      content(
        "dash",
        text($cmsigma(sigma)$, size: 12pt),
        anchor: "south-west",
        padding: 0.7em,
      )
    }
  }

  // Draw annotations
  if show-annotations {
    content(
        p5,
        text($cmb(phi^1_j)$, size: 12pt),
        anchor: "south-east",
        padding: 0.4em,
      )
  }
}

// ============================================================
// HELPER FUNCTIONS - DNM CONNECTIONS
// ============================================================

#let draw-dnm-connections() = {
}

// ============================================================
// HELPER FUNCTIONS - SOFA TRAJECTORY
// ============================================================

#let draw-sofa-trajectory(show-labels: true) = {
  let stdc = (ode_xoffset + sofa_xoffset, ode_yoffset)
  
  content(stdc, text($hat(O)(s^1_0), ..., hat(O)(s^1_T)$, size: descr_size))

  if show-labels {
    content(
      (stdc.at(0), title_offset + 0.3),
      align(
        text("SOFA-score Trajectory\n ", size: 16pt), 
        center,
      ),
      anchor: "south",
      padding: .3em,
    )
  }
}

// ============================================================
// HELPER FUNCTIONS - INTEGRATION CONNECTION
// ============================================================

#let draw-integration-connection() = {
  let stdc = (ode_xoffset + sofa_xoffset, ode_yoffset)
  
  line(
    (ode_xoffset + 3, ode_yoffset),
    cadd(stdc, (-2, 0)),
    mark: emark,
    name: "integrate",
  )

  content("integrate", text(descr_size)[Numerically Integrate], anchor: "south", padding: .3em)
}



#let create-poster-figure(
  // Component visibility toggles
  show-encoder: true,
  show-predictor: true,
  show-dnm: true,
  show-sofa: true,
  show-connections: true,
  show-labels: true,
  show-annotations: true,
  show-pkappa: true,
  show-immune: true,
  show-ikappa: true,
  show-sigma: true,
) = {
  canvas({
    // ============================================================
    // ENCODER SECTION
    // ============================================================
    if show-encoder {
      draw-encoder-network(
        show-labels: show-labels,
        show-connections: show-connections,
      )
    }

    // ============================================================
    // LATENT PREDICTOR SECTION
    // ============================================================
    if show-predictor {
      draw-latent-predictor(
        show-labels: show-labels,
        show-connections: show-connections,
      )
    }

    // ============================================================
    // CONNECTIONS BETWEEN ENCODER AND PREDICTOR
    // ============================================================
    if show-connections and show-encoder and show-predictor {
      draw-encoder-predictor-connections()
    }

    // ============================================================
    // DYNAMIC NETWORK MODEL (DNM)
    // ============================================================
    if show-dnm {
      draw-dynamic-network-model(
        show-labels: show-labels,
        show-annotations: show-annotations,
        show-pkappa: show-pkappa,
        show-immune: show-immune,
        show-ikappa: show-ikappa,
        show-sigma: show-sigma,
      )
    }

    // ============================================================
    // DNM TO SOFA CONNECTION
    // ============================================================
    if show-connections and show-dnm and show-predictor {
      draw-dnm-connections()
    }

    // ============================================================
    // SOFA-SCORE TRAJECTORY
    // ============================================================
    if show-sofa {
      draw-sofa-trajectory(
        show-labels: show-labels,
      )
    }

    // ============================================================
    // INTEGRATION CONNECTION
    // ============================================================
    if show-connections and show-dnm and show-sofa {
      draw-integration-connection()
    }
  })
}

// ============================================================
// USAGE EXAMPLES
// ============================================================

// Full poster with all components
#let poster_fig = create-poster-figure()

// Poster without SOFA trajectory
#let poster_fig_no_sofa = create-poster-figure(
  show-sofa: false,
)

// Poster with only encoder and predictor
#let poster_fig_encoder_predictor = create-poster-figure(
  show-dnm: false,
  show-sofa: false,
)

// Poster without labels (cleaner look)
#let poster_fig_no_labels = create-poster-figure(
  show-labels: false,
)

// Poster without annotations (mathematical symbols)
#let poster_fig_no_annotations = create-poster-figure(
  show-annotations: false,
)

// Minimal poster (only structure, no labels or annotations)
#let poster_fig_minimal = create-poster-figure(
  show-labels: false,
  show-annotations: false,
)

// ============================================================
// DEFAULT FIGURE
// ============================================================

#figure(poster_fig)
