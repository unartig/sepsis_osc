// #import "@preview/numberingx:0.0.1"
#import "@preview/acrostiche:0.7.0": *
#import "@preview/equate:0.3.1": equate

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
#let tof = outline(title: [List of figures], target: figure.where(kind:image))
#let tot = outline(title: [List of tables], target: figure.where(kind:table)) 
// This function gets your whole document as its `body`.
#let thesis(
  // The title for your work.
  title: [Your Title],
  thesis-type: "Master",
  // Author's name.
  author: "Author",
  institute-logo: image("images/inst_logo.svg"),
  // The paper size to use.
  paper-size: "a4",
  font-size: 11pt,
  leading: 0.8em,
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
  date: none,
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
  appendix: (
    enabled: false,
    heading-numbering-format: "A.1.1",
    body: none,
  ),
  // The result of a call to the `bibliography` function or `none`.
  // Example: bibliography("refs.bib")
  // More info: https://typst.app/docs/reference/model/bibliography/
  bibliography: none,
  // Whether to start a chapter on a new page.
  chapter-pagebreak: true,
  // Whether to display a maroon circle next to external links.
  external-link-circle: true,
  // Display an index of figures (images).
  figure-index: (
    enabled: false,
    title: "",
  ),
  // Display an index of tables
  table-index: (
    enabled: false,
    title: "",
  ),
  // Display an index of listings (code blocks).
  listing-index: (
    enabled: false,
    title: "",
  ),
  head-font: "New Computer Modern",
  body-font: "New Computer Modern",
  raw-font: "Bitstream Vera Sans Mono",
  // The content of your work.
  body,
) = {
  // Set the document's metadata.
  set document(title: title, author: author)

  // Set the body font.
  set text(
    size: font-size, // default is 11pt
    font: body-font,
    lang: language,
  )

  // Set raw text font.
  show raw: set text(font: (raw-font), size: 9pt)

  // Set heading text font.
  show heading.where(level: 1): set text(font-size + 8pt, font: head-font)
  show heading.where(level: 2): set text(font-size + 4pt, font: head-font)
  show heading.where(level: 3): set text(font-size + 2pt, font: head-font)

  // Configure page size and margins.
  set page(
    numbering: "i",
    paper: paper-size,
    margin: (bottom: 1.75cm, top: 3cm),
  )

  // Cover page.
  page(
    footer: "",
    grid(
      columns: (1fr, 1.5fr),
      gutter: 1cm,
      align: center + horizon,
      image("images/TUHH_logo-wortmarke_en_rgb.svg"), institute-logo,
    )
      + v(3cm)
      + align(
        center + top,
        block(width: 90%)[
          #text(2em, font: head-font)[*#title*]
          // #text(2.2em)[*#title*]


          #v(1fr)
          #text(1.5em)[
            #if language == "en" [#thesis-type's Thesis] else if (
              language == "de"
            ) { thesis-type + "arbeit" } else [= language not defined]

            #v(1em)
            #if language == "en" [of] else if (
              language == "de"
            ) [von] else [= language not defined]
            \ #author

            #v(1fr)
            #set text(14pt)
            #grid(
              columns: (1fr, 1fr),
              align: (right, left),
              gutter: 0.5em,
              if language == "en" [Date of issue:] else if language
                == "de" [Ausgabedatum:] else [anguage not defined],
              if date-of-issue != none { date-of-issue.display(date-format) },

              if language == "en" [Date of submission:] else if language
                == "de" [Abgabedatum:] else [language not defined],
              if date-of-submission != none {
                date-of-submission.display(date-format)
              },

              if language == "en" {
                if second-examiner != none [Examiners:] else [Examiner:]
              } else if language
                == "de" [Geprüft von:] else [language not defined],
              first-examiner + linebreak() + second-examiner,

              if language == "en" {
                if second-examiner != none [Supervisors:] else [Supervisor:]
              } else if language
                == "de" [Betreut von:] else [language not defined],
              first-supervisor + linebreak() + second-supervisor,
            )
            #v(2em)
          ]
        ],
      ),
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

  // Configure paragraph properties.
  set par(
    leading: leading,
    spacing: 1em, // TODO probably 1em is correct
    justify: true,
    linebreaks: "optimized",
    first-line-indent: 0pt,
  )

  // Add vertical space after headings.
  // show heading: it => {
  //   it
  //   v(3%, weak: true)
  // }
  // Do not hyphenate headings.
  // show heading: set text(hyphenate: false)

  // Show a small maroon circle next to external links.
  // show link: it => {
  //   it
  //   // Workaround for ctheorems package so that its labels keep the default link styling.
  //   if external-link-circle and type(it.dest) != label {
  //     sym.wj
  //     h(1.6pt)
  //     sym.wj
  //     super(box(height: 3.8pt, circle(
  //       radius: 1.2pt,
  //       stroke: 0.7pt + rgb("#993333"),
  //     )))
  //   }
  // }

  // Display preface as the second page.
  page(footer: none)[]
  pagebreak(to: "odd")
  page({
    v(12.5%)
    if language == "en" [
      = Statutory Declaration
      #v(3cm)
      I, #author, hereby affirm that the following #thesis-type's thesis has been elaborated solely by myself.
      No other means and sources except those stated, referenced and acknowledged have been used.
      #v(3cm)
      #if date != none { date.display(date-format) }
      #v(2cm)
      (#author)
    ] else if language == "de" [
      = Eigenständigkeitserklärung
      #v(3cm)
      Hiermit erkläre ich, #author, an Eides statt, dass ich die vorliegende #{ thesis-type + "arbeit" } selbstständig verfasst und keine anderen
      als die angegebenen Quellen und Hilfsmittel verwendet habe.
      #v(3cm)
      #date.display(date-format)
      #v(2cm)
      (#author)
    ] else [= language not defined]
  })

  // Display the summary.
  if summary != none {
    pagebreak(to: "odd")
    v(12.5%)
    if language == "en" [= Abstract] else if (
      language == "de"
    ) [= Kurzfassung] else [= language not defined]
    summary
    pagebreak(to: "odd", weak: true)
  }

  // Indent nested entires in the outline.
  set outline(indent: auto)

  init-acronyms(acronyms)
  // Display tables of contents.
  if table-of-contents != none {
    v(12.5%)
    table-of-contents
  }

  pagebreak(weak:true)
  if table-of-tables != none {
    tot
  }
  if table-of-figures != none {
    tof
  }
  pagebreak(weak:true)
  print-index(row-gutter: 4pt)

  if notation != none {
    pagebreak(weak: true, to: "odd")
    v(12.5%)
    notation
  }
  pagebreak(weak: true, to: "odd")

  // Configure page numbering and footer.
  set page(
    numbering: "1",
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
      align(aln)[#i]
    },
    header: context {
      // Get current page number.
      let abs-page-number = here().page()

      // Align right for even pages and left for odd.
      let is-odd = calc.odd(abs-page-number)
      let aln = if is-odd {
        right
      } else {
        left
      }

      // Are we on a page that starts a chapter?
      let target = if is-odd {
        heading.where(level: 2)
      } else {
        heading.where(level: 1)
      }
      if query(heading.where(level: 1)).any(it => (
        it.location().page() == abs-page-number
      )) {
        return align(aln)[]
      }

      // Find the chapter of the section we are currently in.
      let before = query(target.before(here()))
      if before.len() > 0 {
        let current = before.last()
        let gap = 1.75em
        let chapter = smallcaps(text(size: 0.68em, current.body))
        if current.numbering != none {
          if is-odd {
            align(aln)[#chapter]
          } else {
            align(aln)[#chapter]
          }
          line(length: 100%, stroke: black)
        }
      }
    },
  )
  // Configure equation numbering.
  show: equate.with(breakable: true, sub-numbering: true)
  set math.equation(numbering: "(1.1)")

  // Display inline code in a small box that retains the correct baseline.
  show raw.where(block: false): box.with(
    fill: fill-color.darken(2%),
    inset: (x: 3pt, y: 0pt),
    outset: (y: 3pt),
    radius: 2pt,
  )

  // Display block code with padding.
  show raw.where(block: true): block.with(inset: (x: 5pt))

  // Break large tables across pages.
  show figure.where(kind: table): set block(breakable: true)
  set table(
    // Increase the table cell's padding
    inset: 7pt, // default is 5pt
    stroke: (0.5pt + stroke-color),
  )
  // Use smallcaps for table header row.
  show table.cell.where(y: 0): smallcaps

  // Wrap `body` in curly braces so that it has its own context. This way show/set rules
  // will only apply to body.
  // Configure heading numbering.
  set heading(numbering: "1.1")
  // Start chapters on a new page.
  {
    show heading.where(level: 1): it => {
      if chapter-pagebreak {
        pagebreak(weak: true, to: "odd")
      }
      v(15%)
      counter(heading).display("1")
      h(0.8em)
      it.body
    }
    counter(page).update(1)
    body
  }

  // Display appendix before the bibliography.
  if appendix.enabled {
    pagebreak(to: "odd")
    set heading(numbering: appendix.heading-numbering-format)
    {
      counter(heading).update(0)
      show heading.where(level: 1): it => {
        if chapter-pagebreak {
          pagebreak(weak: true, to: "odd")
        }
        v(15%)
        counter(heading).display("A")
        h(0.8em)
        it.body
      }
      appendix.body
    }
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
    bibliography
  }

  // Display indices of figures, tables, and listings.
  let fig-t(kind) = figure.where(kind: kind)
  let has-fig(kind) = counter(fig-t(kind)).get().at(0) > 0
  if figure-index.enabled or table-index.enabled or listing-index.enabled {
    show outline: set heading(outlined: true)
    context {
      let imgs = figure-index.enabled and has-fig(image)
      let tbls = table-index.enabled and has-fig(table)
      let lsts = listing-index.enabled and has-fig(raw)
      if imgs or tbls or lsts {
        // Note that we pagebreak only once instead of each each individual index. This is
        // because for documents that only have a couple of figures, starting each index
        // on new page would result in superfluous whitespace.
        pagebreak()
        v(25%)
      }

      if imgs {
        outline(
          title: figure-index.at("title", default: "Index of Figures"),
          target: fig-t(image),
        )
      }
      if tbls {
        outline(
          title: table-index.at("title", default: "Index of Tables"),
          target: fig-t(table),
        )
      }
      if lsts {
        outline(
          title: listing-index.at("title", default: "Index of Listings"),
          target: fig-t(raw),
        )
      }
    }
  }
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
