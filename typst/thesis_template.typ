#import "@preview/acrostiche:0.7.0": *
#import "@preview/equate:0.3.1": equate
#import "titlepage_template.typ": make-titlepage

#let in-outline = state("in-outline", false)
// #show outline: it => {
//   in-outline.update(true)
//   it
//   in-outline.update(false)
// }
#let flex-caption(short: none, long: none) = context if in-outline.get() { short } else { long }

// Workaround for the lack of an `std` scope.
#let std-bibliography = bibliography
#let std-smallcaps = smallcaps
#let std-upper = upper

// Overwrite the default `smallcaps` and `upper` functions with increased spacing between
// characters. Default tracking is 0pt.
#let smallcaps(body) = std-smallcaps(text(tracking: 0.6pt, body))
#let upper(body) = std-upper(text(tracking: 0.6pt, body))

// Colors used across the template.
#let stroke-color = luma(200)
#let fill-color = luma(250)

  

#show math.equation.where(block: false): box
// This function gets your whole document as its `body`.
#let thesis(
  // The title for your work.
  title: [Your Title],
  thesis-type: "Master",
  // Author's name.
  author: "Author",
  institute-logo: image("images/logo_tuhh_uke.svg"),
  program-name: "Example-Program",
  // The paper size to use.
  paper-size: "a4",
  font-size: 12pt,
  leading: .6em,
  // Language
  language: "en",
  //examiners
  first-examiner: none,
  second-examiner: none,
  first-supervisor: none,
  second-supervisor: none,
  // Date that will be displayed on cover page.
  // The value needs to be of the 'datetime' type.
  // More info: https://typst.app/docs/reference/foundations/datetime/
  // Example: datetime(year: 2024, month: 03, day: 17)
  date: datetime.today(),
  date-of-issue: none,
  date-of-submission: none,
  // Format in which the date will be displayed on cover page.
  // More info: https://typst.app/docs/reference/foundations/datetime/#format
  // The default format will display date as: MMMM DD, YYYY
  date-format: "[day].[month].[year repr:full]",
  // The contents for the summary pages. This will be displayed after the cover page. Can
  // be omitted if you don't have one.
  summary: none,
  // The contents for the notation pages. This will be displayed after the cover page. Can
  // be omitted if you don't have one.
  notation: none,
  // The result of a call to the `outline` function or `none`.
  // Set this to `none`, if you want to disable the table of contents.
  // More info: https://typst.app/docs/reference/model/outline/
  table-of-contents: outline(depth:2),
  table-of-figures: none,
  table-of-tables: none,
  acronyms: none,
  // Display an appendix after the body but before the bibliography.
  appendix-file: none,
  // The result of a call to the `bibliography` function or `none`.
  // Example: bibliography("refs.bib")
  // More info: https://typst.app/docs/reference/model/bibliography/
  bibliography: none,
  // Whether to start a chapter on a new page.
  chapter-pagebreak: true,
  // Whether to display a maroon circle next to external links.
  external-link-circle: true,
  head-font: "Nimbus Sans",
  body-font: "TeX Gyre Termes",
  raw-font: "Bitstream Vera Sans Mono",
  // The content of your work.
  body,
  acknowledgements: none,
) = {
  // Set the document's metadata.
  set document(title: title, author: author)

  make-titlepage(
  title: title, 
  thesis-type: thesis-type,
  subtitle: none,
  date: date,
  logo: scale(institute-logo, 100%),
  first-supervisor: first-supervisor,
  second-supervisor: second-supervisor,
  author: author
  )

  // Set the body font.
  set text(
    size: font-size, // default is 11pt
    font: body-font,
    lang: language,
    // spacing: 0.1em
  )

  // Set raw text font.
  show raw: set text(font: (raw-font), size: 9pt)

  // Set heading text font.
  show heading: set block(below: 1.5em, above:2.5em)
  show heading.where(level: 1): set text(font-size + 8pt, font: head-font)
  show heading.where(level: 1): set align(right)
  show heading.where(level: 1, supplement: none): set heading(supplement: [Chapter])
  show heading.where(level: 2): set text(font-size + 3pt, font: head-font)
  show heading.where(level: 3): set text(font-size + 1pt, font: head-font)

  // Configure page size and margins.
  set page(
    paper: paper-size,
    margin: (top: 3cm, bottom: 3cm, inside: 3cm, outside: 2.5cm),
  )

  set page(
    footer: context {
      // Get current page number.
      let i = counter(page).at(here()).first()
      let abs-page-number = here().page()

      // Align right for even pages and left for odd.
      let is-odd = calc.odd(abs-page-number)
      let aln = if is-odd {
        right
      } else {
        left
      }
      align(aln)[#counter(page).display("i")]
    },
  )

  // Customize figures of tables
  show figure.where(
    kind: table,
  ): set figure.caption(position: top)

  show figure.caption: it => {
    let supplement = it.supplement
    let number = it.counter.display(it.numbering)
    let label = text(weight: "bold")[#supplement #number:]
  
    pad(
      left: 0pt,
      block(width: 100%)[
        #grid(
          columns: (auto, 1fr),
          column-gutter: 0.5em,
          label,
          align(left, {
                                        set text(size: font-size - 1pt)
                                        it.body})
        )
      ]
    )
  }

  // Configure paragraph properties.
  set par(
    leading: leading,
    spacing: 1em, // TODO probably 1em is correct
    justify: true,
    linebreaks: "optimized",
    first-line-indent: 0pt,
  )

  // Display preface as the second page.
  page(footer: none)[]
  pagebreak(to: "odd")
  page({
    v(22.5%)
    [
      #heading([Eigenständigkeitserklärung], outlined: false)
      // #v(3cm)
      Hiermit erkläre ich, #author, an Eides statt, dass ich die vorliegende #{ thesis-type + "arbeit" } im Studiengang "#{program-name}" selbstständig verfasst und keine anderen als die angegebenen Hilfsmittel #sym.dash.en insbesondere keine im Quellenverzeichnis nicht benannten Internet-Quellen #sym.dash.en benutzt habe.
      Alle Stellen, die wörtlich oder sinngemäß aus Veröffentlichungen entnommen wurden, sind als solche kenntlich gemacht.
      Ich versichere weiterhin,  dass ich die Arbeit vorher nicht in einem anderen Prüfungsverfahren eingereicht habe.
      Sofern im Zuge der Erstellung der vorliegenden Abschlussarbeit generative Künstliche Intelligenz (gKI)-basierte elektronische Hilfsmittel verwendet wurden, versichere ich, dass meine eigene Leistung im Vordergrund stand und dass eine vollständige Dokumentation aller verwendeten Hilfsmittel gemäß der Guten Wissenschaftlichen Praxis vorliegt.
      Ich trage die Verantwortung für eventuell durch die gKI generierte fehlerhafte oder verzerrte Inhalte, fehlerhafte Referenzen, Verstöße gegen das Datenschutz- und Urheberrecht oder Plagiate.
      #v(1cm)

      Berlin, den #date.display(date-format)
      #v(2cm)
      (#author)
    ]
  })


  if acknowledgements != none {
    v(22.5%)
    heading([Acknowledgements], outlined: false)
    acknowledgements
    pagebreak()
  }

  // TABLE OF CONTENTS / FIGURES / TABLES
  set outline(indent: auto)
  init-acronyms(acronyms)
  if table-of-contents != none {
    // Display tables of contents.
    show outline.entry: set block(above: .5em)
    show outline.entry.where(level: 1): set block(above: 1.2em)
    show outline.entry: it => link(it.element.location(),
                                   it.indented(text(it.prefix(), 12pt),
                                                     text(it.body() + "   " + box(width:5fr, repeat([.], gap: 0.6em)) + "   " +  it.page(), 12pt)))
    show outline.entry.where(level: 1): it => link(it.element.location(),
                                                   it.indented(text(it.prefix(),font:head-font, 10pt, weight: "bold"),
                                                                     text(it.body() + box(width:1fr, "") +  "   " + it.page(), font:head-font, 10pt, weight: "bold")))
    show outline.entry: it => {
      show linebreak: none
      it
    }

    v(2.5%)
    // v(12.5%)
    table-of-contents
  }

  // set outline(indent: auto)
  pagebreak(weak:true)
  if table-of-tables != none {
    let tot = outline(title: [List of tables], target: figure.where(kind:table)) 
    in-outline.update(true)
    tot
    in-outline.update(false)
  }
  if table-of-figures != none {
    let tof = outline(title: [List of figures], target: figure.where(kind:image))
    in-outline.update(true)
    tof
    in-outline.update(false)
  }
  pagebreak(weak:true)
  print-index(row-gutter: 4pt)

  if notation != none {
    pagebreak(weak: true, to: "odd")
    v(12.5%)
    notation
  }
  pagebreak(weak: true, to: "odd")


  // Display the summary.
  if summary != none {
    pagebreak()
    // pagebreak(to: "odd")
    v(12.5%)
    heading(outlined: false, [Summary],)
    summary
    pagebreak(to: "odd", weak: true)
  }
  // PAGE
  set page(
    // FOOTER FOR FIRST PAGE IN CHAPTER
    footer: context {
      let i = counter(page).at(here()).first()
      let abs-page-number = here().page()
      let is-odd = calc.odd(abs-page-number)
      let chapter-page =  if query(heading.where(level: 1)).any(it => (it.location().page() == abs-page-number)) {true} else {false}
      if not chapter-page {return} else {align(right, text([#i], 14pt))}
    },
    // ELSE HEADER - ALTERNATING
    header: context {
      let i = counter(page).at(here()).first()
      let abs-page-number = here().page()
      let is-odd = calc.odd(abs-page-number)
      let chapter-page =  if query(heading.where(level: 1)).any(it => (it.location().page() == abs-page-number)) {true} else {false}
      if chapter-page {return} else {
        let before = query(heading.where(level:1).before(here()))
        if before.len() > 0 {
          let current = before.last()
          let chapter = {show linebreak: none; current.body}
          if current.numbering != none {
            if is-odd {
              text([#chapter], 14pt) + h(1fr) + text([#i], 14pt)
            } else {
              text([#i], 14pt) + h(1fr) + text([#chapter], 14pt)
            }
            v(-.2em)
            line(length: 100%, stroke: black)
            }
          }
      }
    }
  )

  // EQUATIONS
  show: equate.with(breakable: true, sub-numbering: true)
  // TODO can we have both?
  // set math.equation(numbering: "(1.1)")
  set math.equation(numbering: (n, ..) => {
    let h1 = counter(heading).get().first()
    numbering("(1.1)", h1, n)
  })
   // OTHER NUMBERINGS
   set figure(numbering: (n, ..) => {
    let h1 = counter(heading).get().first()
    numbering("1.1", h1, n)
  })

  // INLINE CODE
  show raw.where(block: false): box.with(
    fill: fill-color.darken(2%),
    inset: (x: 3pt, y: 0pt),
    outset: (y: 3pt),
    radius: 2pt,
  )

  // Display block code with padding.
  show raw.where(block: true): block.with(inset: (x: 5pt))

  // TABLE
  let tframe(stroke) = (x, y) => (
    left: if x > 0 { 0pt } else { stroke },
    right: stroke,
    top: if y < 2 { stroke } else { 0pt },
    bottom: stroke,
  )
  show figure.where(kind: table): set block(breakable: true)
  set table(
    // Increase the table cell's padding
    inset: 5pt, // default is 5pt
    stroke: tframe(1pt),
  )
  // Use smallcaps for table header row.
  show table.cell.where(y: 0): smallcaps

  // level 1 headings to Chapter
  show ref: it => {
    let el = it.element
    if el != none and el.func() == heading and el.level == 1 {
      link(
        it.target, 
        [Chapter #numbering(
          el.numbering,
          ..counter(heading).at(el.location())
        )]
      )
    } else {
      it
    }
  }

  // Wrap `body` in curly braces so that it has its own context. This way show/set rules
  // will only apply to body.
  // Configure heading numbering.
  set heading(numbering: "1.1")
  {
    show heading.where(level: 1): it => {
      // reset counter numbers
      counter(math.equation).update(0)
      counter(figure).update(0)
      counter(table).update(0)
      counter(figure.where(kind: image)).update(0)
      counter(figure.where(kind: table)).update(0)
      //
      if chapter-pagebreak {
        pagebreak(weak: true, to: "odd")
      }
      {
        set text(size: 90pt, font: "URW Bookman", fill:luma(150))
        counter(heading).display("1")
      }
      linebreak()
      it.body
      v(2.5%)
    }
    counter(page).update(1)
    body
  }
  // spacing
  {
    show heading: it => {
      it.body
      v(12em)
    }
  }

  // Display appendix before the bibliography.
  if appendix-file != none {
    pagebreak(to: "odd")
    counter(heading).update(0)
    counter(figure).update(0)  // Reset figure counter
    set heading(numbering: "A.1", supplement: [Appendix])
  
    // Update figure numbering to use letter prefix
    set figure(numbering: (n, ..) => {
      let h1 = counter(heading).get().first()
      numbering("A.1.", h1, n)
    })
  
    let toa = outline(title: text([Appendix], font-size + 8pt, font: head-font), target: heading.where(supplement: [Appendix]))
    toa
    include appendix-file
  }

  // Display bibliography.
  if bibliography != none {
    pagebreak()
    v(25%)
    show std-bibliography: set text(0.85em)
    // Use default paragraph properties for bibliography.
    show std-bibliography: set par(
      leading: 0.65em,
      justify: false,
      linebreaks: auto,
    )
    
    show link: set text(blue)
    show link: underline
    bibliography
  }

  // Display indices of figures, tables, and listings.
  let fig-t(kind) = figure.where(kind: kind)
  let has-fig(kind) = counter(fig-t(kind)).get().at(0) > 0
}

// This function formats its `body` (content) into a blockquote.
#let blockquote(body) = {
  block(
    width: 100%,
    fill: fill-color,
    inset: 2em,
    stroke: (y: 0.5pt + stroke-color),
    body,
  )
}
