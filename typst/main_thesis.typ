#import "thesis_template.typ": thesis
#import "thesis_env.typ": *

#import "figures/tree.typ": tree_fig




#show: thesis.with(
  title: "Comprehensive Guidelines and Templates for Thesis Writing",
  summary: [
  ],
  // abstract_de: [
  // ],
  acronyms: (
    "ABX": "Antibiotics",
    "BCE": "Binary Cross Entropy",
    "DAMP": "Damage-Associated Molecular Patterns",
    "DL": "Deep Learning",
    "DNM": "Dynamic Network Model",
    "EHR": "Electronic Health Record",
    "FSQ": "Finite Scalar Quantization",
    // "GLM": "Generalized Linear Model",
    "GPU": "Graphics Processing Unit",
    "ICU": "Intensive Care Unit",
    "JIT": "Just In Time Compilation",
    "LDM": "Latent Dynamics Model",
    "ML": "Machine Learning",
    "MSE": "Mean Squared Error",
    "ODE": "Ordinary Differential Equation",
    "PAMP": "Pathogen-Associated Molecular Patterns",
    "PID": "Proportional-Integral-Derivative",
    "PINN": "Physics Informed Neural Networks",
    "PRR": "Pattern Recognition Receptors",
    "qSOFA": "Quick Sequential Organ Failure Assessment",
    "RAG": "Retrieval Augmented Generation",
    "SIRS": "Systemic Inflammatory Response Syndrome",
    "SI": "Suspected Infection",
    "SOFA": "Sequential Organ Failure Assessment",
    "TUHH": "Hamburg University of Technology",
    "YAIB": "Yet Another ICU Benchmark",
  ),
  bibliography: bibliography("bibliography.bib"),
  // TODO what about notation?
  table-of-figures: none,  // TODO and make pretty smh
  table-of-tables: true,
  // acknowledgements: [
  //   This thesis was written with the help of many people.
  //   I would like to thank all of them.
  // ],
)



#TODO[
  #list(
    [Sections to Chapters],
    [Styling],
    [Appendix to real Appendix],
    [Fix ACR separation],
    [Fix newline/linebreak after Headings],
    [Sources misc + link],
  )
]
#TODO[actual functional model
  what is learned
  connecting parts
]


#include "chapters/introduction.typ"

#include "chapters/sepsis.typ"

#include "chapters/problem_def.typ"

#include "chapters/dnm.typ"

#include "chapters/method.typ"

#include "chapters/experiment.typ"

#include "chapters/result.typ"

#include "chapters/discussion.typ"

#include "chapters/appendix.typ"



