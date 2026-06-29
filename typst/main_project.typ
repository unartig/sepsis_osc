#import "thesis_template.typ": thesis
#import "thesis_env.typ": *

#show: thesis.with(
  title: "Investigating the Latent Dynamics Model",
  summary: [
Early identification of sepsis in intensive care units remains a critical challenge due to its highly heterogeneous pathophysiology.
This work presents a comprehensive evaluation of the Latent Dynamics Model, a physics-informed deep learning architecture that integrates a biophysically grounded Physiological Network Model to predict online sepsis onset using electronic health records.
By decomposing the Sepsis-3 criterion into independent infection and organ-dysfunction modules, the Latent Dynamics Model projects patient trajectories into an interpretable two-dimensional parameter space governed by biological age/comorbidity and immune-organ coupling.

Through systematic ablation studies, model variations, and external validation across the MIMIC-IV and eICU databases, the performance and interpretability trade-offs inherent to this architecture are investigated and discussed.
The ablation analysis uncovers a clear hierarchy: while primary losses are load-bearing for task calibration and auxiliary geometric terms act purely as regularizers, consistent with their intended design goals.
Crucially, variation experiments isolate a performance bottleneck within the discrete latent lookup mechanism, where replacing the exact grid latent space with smooth, differentiable alternatives yields a significant increase in acute organ dysfunction detection performance.
External validation demonstrates competitive generalization performance, underscoring the model's ability to extract robust cross-cohort patient representations.
The findings establish the standard Latent Dynamics Model as a deliberate, design compromise that sacrificing downstream predictive performance in exchange for a structurally stable, decodable, and clinically meaningful latent space.
  ],
  acronyms: (
    "AUPRC": "Area Under Precision Recall Curve",
    "AUROC": "Area Under Receiver Operating Curve",
    "BCE": "Binary Cross Entropy",
    // "DAMP": "Damage-Associated Molecular Patterns",
    // "DL": "Deep Learning",
    "PNM": "Physiological Network Model",
    "eICU": "eICU Collaborative Research Database ",
    "EHR": "Electronic Health Record",
    // "FN": "False Negatives",
    // "FNR": "False Negatives Rate",
    // "FP": "False Positives",
    // "FPR": "False Positive Rate",
    // "GELU": "Gaussian Error Linear Unit",
    "GRU": "Gated Recurrent Unit",
    "ICU": "Intensive Care Unit",
    "LDM": "Latent Dynamics Model",
    // "LOS": "Length Of Stay",
    // "LSTM": "Long Short-Term Memory",
    "MIMIC": "Medical Information Mart for Intensive Care",
    // "ML": "Machine Learning",
    "MLP": "Multi-Layer-Perceptron",
    "MSE": "Mean Squared Error",
    // "ODE": "Ordinary Differential Equation",
    // "PAMP": "Pathogen-Associated Molecular Patterns",
    // "qSOFA": "Quick Sequential Organ Failure Assessment",
    "ReLU": "Rectified Linear Unit",
    // "RNN": "Recurrent Neural Networks",
    // "SI": "Suspected Infection",
    "SOFA": "Sequential Organ Failure Assessment",
    // "TN": "True Negatives",
    // "TPR": "True Positive Rate",
    // "TP": "True Positives",
    "YAIB": "Yet Another ICU Benchmark",
  ),
  
  appendix-file: "project_chapters/appendix.typ",
  author: "Juri Backes",
  thesis-type: "Master's Project",
  bibliography: bibliography("bibliography.bib", style:"ieee_custom.csl"),
  // TODO what about notation?
  table-of-figures: true,
  table-of-tables: true,
  first-supervisor: "Prof. Dr. Tobias Knopp",
  second-supervisor: "M. Sc. Artyom Tsanda",
  program-name: "Informatik-Ingenieurwesen",
  acknowledgements: none,
)


#include "project_chapters/introduction.typ"
#include "project_chapters/background.typ"
#include "project_chapters/experiments.typ"
#include "project_chapters/results.typ"
#include "project_chapters/discussion.typ"
#include "project_chapters/conclusion.typ"

