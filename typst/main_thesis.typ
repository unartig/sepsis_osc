#import "thesis_template.typ": thesis
#import "thesis_env.typ": *

// #import "figures/tree.typ": tree_fig

#show: thesis.with(
  title: "Combining Machine-Learning and \n Dynamic Network Models \n to Improve Sepsis Prediction",
  summary: [
  As the most extreme course of an infectious disease, sepsis poses a serious health threat, with a high mortality rate and frequent long-term consequences for survivors.
  Despite its enormous burden on global healthcare and ongoing research efforts, early sepsis prediction remains challenging due to the complex nature of its pathophysiology.
  Current approaches face a fundamental trade-off: data-driven machine learning models achieve strong performance but lack interpretability, while biologically inspired models provide mechanistic insights but have limited clinical validation.
  This thesis develops the _Latent Dynamics Model_, a hybrid machine learning approach that integrates a functional model of coupled oscillators representing organ- and immune-cell populations.
  // Rather than treating sepsis as a black-box classification task, the model decomposes prediction into two interpretable components, suspected infection likelihood and acute organ dysfunction.
  By projecting high-dimensional patient data into the low-dimensional parameter space of the functional model, machine-learned trajectories through this space enable detection and prediction of critical organ system states.
  The proposed method is trained and evaluated retrospectively on real intensive care patients, achieving state of the art performance.
  Qualitative analysis reveals that learned trajectories exhibit clinically plausible patterns of deterioration, recovery, and stability.
  This work demonstrates that embedding biologically grounded structure can improve both predictive performance and interpretability in sepsis prediction.
  ],
  acronyms: (
    // "ABX": "Antibiotics",
    "AUPRC": "Area Under Precision Recall Curve",
    "AUROC": "Area Under Receiver Operating Curve",
    "BCE": "Binary Cross Entropy",
    "DAMP": "Damage-Associated Molecular Patterns",
    "DL": "Deep Learning",
    "DNM": "Dynamic Network Model",
    "EHR": "Electronic Health Record",
    "FN": "False Negatives",
    "FNR": "False Negatives Rate",
    "FP": "False Positives",
    "FPR": "False Positive Rate",
    // "FSQ": "Finite Scalar Quantization",
    // "GLM": "Generalized Linear Model",
    "GELU": "Gaussian Error Linear Unit",
    // "GPU": "Graphics Processing Unit",
    "GRU": "Gated Recurrent Unit",
    "ICU": "Intensive Care Unit",
    // "IQR": "Interquantile Range",
    // "JIT": "Just In Time Compilation",
    "LDM": "Latent Dynamics Model",
    "LOS": "Length Of Stay",
    "LSTM": "Long Short-Term Memory",
    "MIMIC": "Medical Information Mart for Intensive Care",
    "ML": "Machine Learning",
    // "MLP": "Multi Layer Perceptron",
    "MSE": "Mean Squared Error",
    "ODE": "Ordinary Differential Equation",
    "PAMP": "Pathogen-Associated Molecular Patterns",
    // "PID": "Proportional-Integral-Derivative",
    // "PINN": "Physics Informed Neural Networks",
    // "PRR": "Pattern Recognition Receptors",
    "qSOFA": "Quick Sequential Organ Failure Assessment",
    // "RAG": "Retrieval Augmented Generation",
    "ReLU": "Rectified Linear Unit",
    "RNN": "Recurrent Neural Networks",
    // "SIRS": "Systemic Inflammatory Response Syndrome",
    "SI": "Suspected Infection",
    "SOFA": "Sequential Organ Failure Assessment",
    // "SVM": "Support Vector Machine",
    "TN": "True Negatives",
    "TPR": "True Positive Rate",
    "TP": "True Positives",
    // "TUHH": "Hamburg University of Technology",
    "YAIB": "Yet Another ICU Benchmark",
  ),
  
  appendix-file: "chapters/appendix.typ",
  author: "Juri Backes",
  bibliography: bibliography("bibliography.bib", style:"ieee_custom.csl"),
  // TODO what about notation?
  table-of-figures: true,
  table-of-tables: true,
  first-supervisor: "Prof. Dr. Tobias Knopp",
  second-supervisor: "M. Sc. Artyom Tsanda",
  program-name: "Informatik-Ingenieurwesen",
  acknowledgements: [
  I would like to thank Prof. Dr. Renz and Prof. Dr. Schöll for introducing me to this fascinating research topic, for the numerous discussions, and for their support throughout the creation of this thesis.
Furthermore, I would like to thank my supervisors Prof. Dr. Knopp and M. Sc. Tsanda for allowing me to work on this project, for the substantial discussions and inputs, and for creating the opportunity for me to attend an expert meeting on the topic.
Lastly, I would like to thank all my friends and family for their support throughout this journey.
  ],
)


#include "chapters/introduction.typ"

#include "chapters/sepsis.typ"

#include "chapters/soa.typ"

#include "chapters/dnm.typ"

#include "chapters/method.typ"

#include "chapters/experiment.typ"

#include "chapters/discussion.typ"

#include "chapters/conclusion.typ"
