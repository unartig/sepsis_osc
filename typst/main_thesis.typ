#import "thesis_template.typ": thesis
#import "thesis_env.typ": *

// #import "figures/tree.typ": tree_fig

#show: thesis.with(
  title: "Combining Machine-Learning and Dynamic Network Models to Improve Sepsis Prediction",
  summary: [],
  // abstract_de: [
  // ],
  acronyms: (
    "ABX": "Antibiotics",
    "AUPRC": "Area Under Precision Recall Curve",
    "AUROC": "Area Under Receiver Operationg Curve",
    "BCE": "Binary Cross Entropy",
    "DAMP": "Damage-Associated Molecular Patterns",
    "DL": "Deep Learning",
    "DNM": "Dynamic Network Model",
    "EHR": "Electronic Health Record",
    "FSQ": "Finite Scalar Quantization",
    // "GLM": "Generalized Linear Model",
    "GPU": "Graphics Processing Unit",
    "GRU": "Gated Recurrent Unit",
    "ICU": "Intensive Care Unit",
    "JIT": "Just In Time Compilation",
    "LDM": "Latent Dynamics Model",
    "LSTM": "Long Short-Term Memory",
    "MIMIC": "Medical Information Mart for Intensive Care",
    "ML": "Machine Learning",
    "MLP": "Multi Layer Perceptron",
    "MSE": "Mean Squared Error",
    "ODE": "Ordinary Differential Equation",
    "PAMP": "Pathogen-Associated Molecular Patterns",
    "PID": "Proportional-Integral-Derivative",
    "PINN": "Physics Informed Neural Networks",
    "PRR": "Pattern Recognition Receptors",
    "qSOFA": "Quick Sequential Organ Failure Assessment",
    "RAG": "Retrieval Augmented Generation",
    "RNN": "Recurrent Neural Networks",
    "SIRS": "Systemic Inflammatory Response Syndrome",
    "SI": "Suspected Infection",
    "SOFA": "Sequential Organ Failure Assessment",
    "SVM": "Support Vector Machine",
    "TUHH": "Hamburg University of Technology",
    "YAIB": "Yet Another ICU Benchmark",
  ),
  
  appendix-file: "chapters/appendix.typ",
  bibliography: bibliography("bibliography.bib"),
  // TODO what about notation?
  table-of-figures: true,  // TODO and make pretty smh
  table-of-tables: true,
  // acknowledgements: [
  //   This thesis was written with the help of many people.
  //   I would like to thank all of them.
  // ],
)



#TODO[
  #list(
    [Styling level 1 header],
    [Figure short captions],
    [fix Bibliography in TOC],
    [fix figure captions],
    [Sources misc + link],
    [Eq/fig numberings],
    [TUHH address wrong in template],
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

#include "chapters/soa.typ"

#include "chapters/method.typ"

#include "chapters/experiment.typ"

#include "chapters/result.typ"

#include "chapters/discussion.typ"

