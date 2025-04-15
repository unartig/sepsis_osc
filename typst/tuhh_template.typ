#import "@preview/acrostiche:0.4.0": *
#import "@preview/equate:0.3.1": equate
// Workaround for the lack of an `std` scope.
#let std-bibliography = bibliography

// This function formats the entire document as a thesis
#let thesis(
  // The title of the thesis.
  title: "Thesis Title",
  // The type of the thesis, e.g., "Bachelor Thesis".
  thesis_type: "Master Thesis",
  // The academic grade that is achieved with the thesis.
  academic_grade: "Master of Science",
  // The author of the thesis.
  author: "Juri Backes",
  // The institute where the thesis was written.
  institute: "Institute for Biomedical Imaging",
  // The university where the thesis was written.
  university: "Hamburg University of Technology",
  // The date of the thesis.
  date: datetime(year: 2024, month: 6, day: 17),
  // The first examiner of the thesis.
  first_examiner: "????",
  // The second examiner of the thesis.
  second_examiner: "????",
  // The supervisor of the thesis.
  supervisor: "????",
  // The logo of the university.
  university_logo: image("images/TUHH_logo-wortmarke_en_rgb.svg"),
  // The logo of the institute.
  institute_logo: image("images/inst_logo.svg"),
  // An abstract in English.
  abstract: [
    #lorem(100)
  ],
  // An abstract in German.
  abstract_de: [
    Dies ist die Kurzfassung.
  ],
  // An array of acronyms, in the style of acrostiche
  acronyms: (
    "WTP": ("Wonderful Typst Package","Wonderful Typst Packages"),
  ),
  // The bibliography of the proposal (result of a call to `bibliography`)
  bibliography: none,
  // Some acknowledgements at the end
  acknowledgements: none,
  // The content of this thesis
  body
) = {
  // Set document metadata
  set document(title: title, author: author)

  // set body font
  set text(font: "New Computer Modern", size: 11pt)

  // set page size to A4
  set page(paper: "a4")

  // ==========
  // Title Page
  // ==========
  stack(
    dir: ltr,
    1cm,
    {
      set image(height: 2cm)
      university_logo
    },
    1fr,
    {
      set image(height: 2cm)
      institute_logo
    },
    1cm,
  )
  v(4cm, weak: true)
  set align(center)
  text(25pt, fill: rgb(0, 0, 153), weight: "bold", title)
  v(1fr, weak: true)
  text(
    14pt,
    [
      #thesis_type\
      to achieve the academic grade\
      #academic_grade
    ],
  )
  v(1fr, weak: true)
  [
    #text(11pt, [submitted by])\
    #text(14pt, author)
  ]
  v(1fr, weak: true)
  text(
    14pt,
    [
      #institute\
      #university\
      #v(1em)
      #date.display("[month repr:long] [year]")
    ],
  )
  v(1.5fr, weak: true)

  set align(left)

  grid(
    columns: (1cm, 7em, 1fr),
    [],
    [
      1st Examiner:\
      2nd Examiner:\
      Supervisor:
    ],
    [
      #first_examiner\
      #second_examiner\
      #supervisor
    ],
  )

  // set alternating margins
  set page(margin: (
    inside: 3cm,
    outside: 2cm,
  ))

  // configure headings: text size, space above and below
  show heading: set block(below: 0.85em, above: 1.75em)
  show heading.where(level: 1): it => {
    counter(math.equation).update(0)
    set block(below: 1.85em, above: 1.75em)
    text(size: 20pt, it)
  }
  show heading.where(level: 2): it => {
    text(size: 16pt, it)
  }
  show heading.where(level: 3): it => {
    text(size: 12pt, it)
  }

  // configure how equations are displayed
  show: equate.with(breakable: true, sub-numbering: true)
  set math.equation(numbering: "(1.1)")

  // show math.equation: set block(spacing: 0.65em)
  // set math.equation(numbering: n => {
  //   let h1 = counter(heading).get().first()
  //   numbering("(1.1)", h1, n)
  // })

  // configure lists
  set enum(indent: 10pt, body-indent: 9pt)
  set list(indent: 10pt, body-indent: 9pt)

  // paragraphs are justified text with indented first line
  set par(justify: true, first-line-indent: 1em, spacing: 0.65em)

  // configure footer for prefix content
  set page(footer: context {
    line(length: 100%, stroke: black + 0.5pt)
    v(-0.7em)
    set align(if calc.even(here().page()) {
      left
    } else {
      right
    })
    counter(page).display("i")
  })

  // reset page counter
  counter(page).update(1)

  // ========
  // Abstract
  // ========
  pagebreak(to: "odd", weak: true)
  [
    #heading(level: 2, outlined: false, [Abstract])
    #abstract
    #v(1fr, weak: true)
    #heading(level: 2, outlined: false, [Kurzfassung])
    #abstract_de
    #v(1fr, weak: true)
  ]

  // =================
  // Table of Contents
  // =================
  show outline.entry.where(level: 1): it => {
    v(1.3em, weak: true)
    strong(it)
  }
  pagebreak(to: "odd", weak: true)
  outline(
    title: heading(level: 1, [Contents #v(1em)]),
    indent: auto,
  )

  // ==============
  // Acronyms Index
  // ==============
  init-acronyms(acronyms)
  pagebreak(to: "odd", weak: true)
  print-index()

  // configure headings to start on odd pages
  show heading.where(level: 1): it => {
    pagebreak(to: "odd", weak: true)
    it
  }
  set heading(numbering: "1.1")
  // configure header and footer for thesis content
  set page(
    header: context {
      let target = selector(heading.where(level: 1, outlined: true))
      let matches = query(target)
      let has_chapter_heading = matches.any(m => (
        m.location().page() == here().page()
      ))
      let headings = query(target.before(here()))
      if not has_chapter_heading and headings.len() != 0 {
        let last_heading = headings.last()
        set align(if calc.even(here().page()) {
          left
        } else {
          right
        })
        [ #counter(heading).at(
            last_heading.location(),
          ).at(0) #last_heading.body ]
        v(-0.7em)
        line(length: 100%, stroke: black + 0.5pt)
      }
    },
    footer: context {
      line(length: 100%, stroke: black + 0.5pt)
      v(-0.7em)
      set align(if calc.even(here().page()) {
        left
      } else {
        right
      })
      counter(page).display("1")
    },
  )
  // reset page counter, content begins...
  counter(page).update(1)

  // =======
  // Content
  // =======
  pagebreak(to: "odd", weak: true)
  body

  // no headers for appendix
  set page(header: none)

  // ============
  // Bibliography
  // ============
  if bibliography != none {
    pagebreak(to: "odd", weak: true)
    set std-bibliography(style: "ieee")
    bibliography
  }

  // ================
  // Acknowledgements
  // ================
  if acknowledgements != none {
    pagebreak(to: "odd", weak: true)
    heading(level: 1, numbering: none, [Acknowledgements])
    acknowledgements
  }

  // ===========
  // Declaration
  // ===========
  pagebreak(to: "odd", weak: true)
  [
    #heading(level: 1, numbering: none, [Declaration])

    Hereby, I declare that I produced the present work myself only with the help of the indicated aids and sources.
    The thesis in its current or a similar version has not been submitted to an auditing institution before.

    #v(2em)
    #grid(
      columns: (1fr, 1fr, 1fr),
      rows: (1em, 2pt, 1em),
      [#h(3pt) Hamburg, #date.display("[month repr:long] [day], [year]")],
      [],
      [],

      line(end: (13em, 0pt), stroke: black + 0.5pt),
      [],
      line(end: (13em, 0pt), stroke: black + 0.5pt),

      text(size: 10pt)[#h(3pt) Place, Date],
      [],
      text(size: 10pt)[#h(3pt) #author],
    )
  ]
}
