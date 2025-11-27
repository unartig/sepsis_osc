#import "@preview/acrostiche:0.5.2": acr, acl
#import "@preview/drafting:0.2.2": inline-note, margin-note, note-outline, set-margin-note-defaults
#import "figures/helper.typ": cmalpha, cmbeta, cmred, cmsigma

#note-outline()
#let mean(f) = $chevron.l$ + f + $chevron.r$
#let ot = $1"/"2$


#let todo = margin-note
#let TODO = inline-note


// #let multicite(x) = "[" +
