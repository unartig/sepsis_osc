#import "tuhh_colors.typ": *

#let cell = rect.with(
  stroke: (thickness: 4pt, paint: colors.turquoise),
  height: 100%,
  width: 100%,
  inset: 0pt,
)
#let placeholder = rect.with(
  outset: -1em,
  fill: color.silver,
)
#let subheader_box_width = 645pt
#let subheader_height = 490pt

#let info-3-column-address(
  institute_name: "Institute (short)",
  tel: "T. +49 40 428 78-00 00",
  fax: "F. +49 40 428 78-00 00",
  room: "Room A0.123",
  email: "max.mustermann@tuhh.de",
  institute_phone: "T. +49 40 428 78-00 00",
) = {
  set text(fill: colors.turquoise)
  cell(
    inset: (x: 80pt),
    align(
      horizon,
      grid(
        columns: (5fr, 4fr, 3fr),
        rows: (1.5em, 1.5em, 1.5em),
        [*Technische Universität Hamburg*], tel, institute_name,
        [Am Schwarzenberg-Campus 1], fax, room,
        [21073 Hamburg], email, institute_phone,
      ),
    ),
  )
}

#let info-3-column-qr(
  institute_name: "Institute (short)",
  name: "Max Mustermann",
  email: "max.mustermann@tuhh.de",
  qr-image: placeholder(width: 8em, height: 8em),
) = {
  set text(fill: colors.turquoise)
  cell(
    inset: (x: 80pt),
    align(
      horizon,
      grid(
        columns: 2,
        block(
          inset: 1.5em,
          grid(
            columns: (5fr, 4fr),
            rows: (1.5em, 1.5em, 1.5em),
            [*Hamburg University of Technology*], institute_name,
            [Am Schwarzenberg-Campus 1], name,
            [21073 Hamburg], email,
          ),
        ),
        qr-image,
      ),
    ),
  )
}

#let tuhh-footer(
  info: info-3-column-address(),
  slogan: [Technisch ist das möglich.],
) = {
  let footer_circle_radius = 32pt
  let footer_circle_offset = 140pt
  cell(
    height: 455pt,
    grid(
      columns: (subheader_box_width, 1fr),
      cell(inset: 83pt, image("images/TUHH_logo-wortmarke_en_rgb.svg")),
      cell(
        grid(
          rows: (6fr, 7fr), cell(
            grid(
              columns: (610pt, 1fr),
              cell({
                place(
                  line(
                    start: (0%, 50%),
                    end: (100%, 50%),
                    stroke: (dash: (0pt, 10pt), cap: "round"),
                  ),
                )
                place(
                  dx: footer_circle_offset - footer_circle_radius,
                  dy: 50% - footer_circle_radius,
                  circle(
                    stroke: none,
                    fill: colors.coral,
                    radius: footer_circle_radius,
                  ),
                )
                place(
                  dx: footer_circle_offset - footer_circle_radius / 2.5,
                  dy: 50% - footer_circle_radius / 2.5,
                  rect(
                    fill: white,
                    width: footer_circle_radius * 4 / 5,
                    height: footer_circle_radius * 4 / 5,
                  ),
                )
              }),
              cell(
                align(
                  horizon + center,
                  text(
                    fill: colors.turquoise,
                    size: 60pt,
                    weight: "semibold",
                    slogan,
                  ),
                ),
              ),
            ),
          ), grid(
            columns: (1fr, 210pt),
            info, cell(),
          ),
        ),
      ),
    ),
  )
}


#let tuhh-poster(
  title: "Poster Title",
  header_image: none,
  institute_name: "Insitute Name",
  footer: tuhh-footer(),
  body,
) = {
  set line(stroke: (thickness: 4pt, paint: colors.turquoise))
  set circle(stroke: (thickness: 4pt, paint: colors.turquoise))

  let circle_radius = 58pt
  let small_circle_radius = circle_radius / 3

  let dot_radius = 5.5pt
  let center_dot_radius = dot_radius * 1.5

  set text(fill: colors.petrol, size: 25pt, font: "New Computer Modern")  // 25

  show heading.where(level: 1): set text(size: 37pt)
  show heading.where(level: 2): set text(
    size: 25pt,
    weight: "semibold",
    fill: colors.turquoise,
  )

  page(
    paper: "a0",
    margin: 88pt,
    grid(
      rows: (2 * circle_radius, subheader_height, 1fr, auto),
      cell({
        place(circle(radius: circle_radius))
        place(dx: 2 * circle_radius, circle(radius: circle_radius))
        place(dx: 4 * circle_radius, circle(radius: circle_radius))
        place(
          line(start: (2 * circle_radius, 0%), end: (2 * circle_radius, 100%)),
        )
        place(
          line(start: (4 * circle_radius, 0%), end: (4 * circle_radius, 100%)),
        )
        place(
          line(start: (6 * circle_radius, 0%), end: (6 * circle_radius, 100%)),
        )
        place(
          line(
            start: (5 * circle_radius, 50%),
            end: (subheader_box_width - small_circle_radius, 50%),
          ),
        )
        place(
          dx: subheader_box_width - small_circle_radius,
          dy: small_circle_radius * 2,
          circle(radius: small_circle_radius),
        )
        place(
          dx: 5 * circle_radius - center_dot_radius,
          dy: 50% - center_dot_radius,
          circle(radius: center_dot_radius, fill: colors.purple, stroke: none),
        )
        for i in range(8) {
          place(
            dx: 5 * circle_radius - dot_radius + calc.cos(i * 45deg) * circle_radius * 0.58,
            dy: 50% - dot_radius + calc.sin(i * 45deg) * circle_radius * 0.58,
            circle(radius: dot_radius, fill: colors.red, stroke: none),
          )
        }
        place(
          dx: subheader_box_width + 2 * circle_radius,
          box(
            height: 100%,
            align(
              horizon,
              text(size: 40pt, fill: colors.turquoise, institute_name),
            ),
          ),
        )
      }),
      cell(if header_image != none {
        grid(
          columns: (subheader_box_width, 1fr),
          header_image,
          cell(
            inset: 2 * circle_radius,
            align(horizon, text(size: 100pt, fill: colors.turquoise, title)),
          ),
        )
      } else {
        cell(
          inset: 2 * circle_radius,
          align(horizon, text(size: 100pt, fill: colors.turquoise, title)),
        )
      }),
      cell(body),
      footer,
    ),
  )
}

