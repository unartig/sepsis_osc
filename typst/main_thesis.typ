#import "thesis_template.typ": thesis
#import "thesis_env.typ": *

// #import "figures/tree.typ": tree_fig

#show: thesis.with(
  title: "Combining Machine-Learning and \n Dynamic Network Models \n to Improve Sepsis Prediction",
  summary: [
  As the most extreme course of an infectious disease, sepsis poses a very serious health threat, with a high mortality rate and frequent long-term consequences for survivors.
  Despite its enormous burden to global healthcare and ongoing research efforts, early sepsis prediction remains challenging due to the complex nature of its pathophysiology.
  Current approaches face a fundamental trade-off, data-driven machine learning models achieve strong performance but lack interpretability, while mechanistic models provide biological insight but have limited clinical validation.
  This thesis develops the _Latent Dynamics Model_, a hybrid approach that integrates a functional model of coupled oscillators representing organ- and immune-cell populations.
  // Rather than treating sepsis as a black-box classification task, the model decomposes prediction into two interpretable components, suspected infection likelihood and acute organ dysfunction.
  By projecting high-dimensional patient data into the low-dimensional parameter space of the functional model, machine-learned trajectories through this space enable detection and prediction of critical organ system states.
  As a proof-of-concept, the proposed method is trained and evaluated retrospectively on real intensive care patients, outperforming baseline methods.
  Qualitative analysis reveals that learned trajectories exhibit clinically plausible patterns of deterioration, recovery, and stability.
  This work demonstrates that embedding biologically-grounded structure can improve both predictive performance and interpretability in sepsis prediction.
  ],
  acronyms: (
    "ABX": "Antibiotics",
    "AUPRC": "Area Under Precision Recall Curve",
    "AUROC": "Area Under Receiver Operationg Curve",
    "BCE": "Binary Cross Entropy",
    "DAMP": "Damage-Associated Molecular Patterns",
    "DL": "Deep Learning",
    "DNM": "Dynamic Network Model",
    "EHR": "Electronic Health Record",
    "FN": "False Negatives",
    "FP": "False Positives",
    "FPR": "False Positive Rate",
    "FSQ": "Finite Scalar Quantization",
    // "GLM": "Generalized Linear Model",
    "GELU": "Gaussian Error Linear Unit",
    "GPU": "Graphics Processing Unit",
    "GRU": "Gated Recurrent Unit",
    "ICU": "Intensive Care Unit",
    "IQR": "Inter Quantile Range",
    "JIT": "Just In Time Compilation",
    "LDM": "Latent Dynamics Model",
    "LOS": "Length Of Stay",
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
    "ReLU": "Rectified Linear Unit",
    "RNN": "Recurrent Neural Networks",
    "SIRS": "Systemic Inflammatory Response Syndrome",
    "SI": "Suspected Infection",
    "SOFA": "Sequential Organ Failure Assessment",
    "SVM": "Support Vector Machine",
    "TN": "True Negatives",
    "TPR": "True Positive Rate",
    "TP": "True Positives",
    // "TUHH": "Hamburg University of Technology",
    "YAIB": "Yet Another ICU Benchmark",
  ),
  
  appendix-file: "chapters/appendix.typ",
  bibliography: bibliography("bibliography.bib"),
  // TODO what about notation?
  table-of-figures: true,
  table-of-tables: true,
  // acknowledgements: [
  //   This thesis was written with the help of many people.
  //   I would like to thank all of them.
  // ],
)


#include "chapters/introduction.typ"

#include "chapters/sepsis.typ"

#include "chapters/soa.typ"

#include "chapters/dnm.typ"

#include "chapters/method.typ"

#include "chapters/experiment.typ"

#include "chapters/discussion.typ"

#include "chapters/conclusion.typ"
