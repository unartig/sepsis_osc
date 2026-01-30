#import "@preview/acrostiche:0.7.0": acr, acl, reset-acronym, acs
#import "@preview/drafting:0.2.2": inline-note, margin-note, note-outline, set-margin-note-defaults
#import "figures/helper.typ": cmalpha, cmbeta, cmred, cmsigma
#import "thesis_template.typ": flex-caption

#note-outline()
#let mean(f) = $chevron.l$ + f + $chevron.r$
#let ot = $1","2$

#let todo = margin-note
#let TODO = inline-note

#let multicite(x) = {x}



// Results
#let auroc = 0.84
#let aurocp = 84.69
#let auprc = 0.09
#let auprcp = 9.87

#let rocpeak = 0.847
#let prcpeak = 0.104

#let rocepo =  255
#let prcepo =  260
#let selepo = 258
#let stopepo = 290

#let totalstart = 493
#let totalendt = 82
#let totalendv = 94
