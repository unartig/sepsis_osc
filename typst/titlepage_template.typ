#let make-titlepage(
  title: "A very long title of the actual work in big sized letters",
  thesis-type: "Bachelor",
  subtitle: none,
  author: "Max Mustermann",
  first-supervisor: "Albert Dreistein",
  second-supervisor: "Wolfgang St. Pauli",
  date: datetime.today(),
  logo: square(size: 5em, stroke: 2pt),
) = [
  // Start the title page
  #pagebreak(weak: true)
  #set page(numbering: none, header: none, footer: none)
  #set text(font: "Nimbus Sans")

  #grid(
    columns: (1.5fr, 1fr),
    gutter: 3cm,
    align: center + horizon,
    "", logo,
  )
  // #align(right, logo)
  
  #line(length: 100%, stroke: gray)
  #v(-0.3em)
  #align(left, text(thesis-type + " Thesis", size: 1.5em))
  #v(-0.7em)
  #line(length: 100%, stroke: gray)
  #v(1fr)


  #v(5em)
  #align(right, text(author, size: 1.5em))
  #v(1.5em)
  #align(right, text(title, size: 2em, weight: "bold"))
  #v(3em)
  #align(right, text(date.display("[month repr:long] [day], [year repr:full]"), size: 1.5em))
  #v(0.8em)

  // Subtitle
  #if subtitle != none [
    #text(subtitle, size: 1.2em)
    #v(2em)
  ]

  #v(13em)
  // Supervisor
  #line(length: 100%, stroke: gray)
  #v(-0.5em)
  #text("supervised by:\n", size:1.2em)#text(first-supervisor + "\n" + second-supervisor, size:1em)
  #v(-0.3em)
  #line(length: 100%, stroke: gray)


  #grid(
    columns: (1fr, 1.4fr),
    gutter: 4cm,
    align: left,
    text("Hamburg University of Technology\nInstitute for Biomedical Imaging\nAm Schwarzenberg-Campus 3\n 21073 Hamburg", size:0.83em),
    text("University Medical Center Hamburg-Eppendorf\nSection for Biomedical Imaging\nLottestraße 55\n22529 Hamburg", size:0.83em),
  )
  #v(1fr)

  // End of title page
  #pagebreak()
]
