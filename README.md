# Chromatin State Prediction from DNA Sequences
This project utilizes Random Forest, Multilayer Perceptron Model (MLP), XGBoost, and CNN architectures to predict 18 distinct chromatin states based on genomic sequence data.

## The Challenge
Chromatin states provide a functional annotation of the genome (e.g., enhancers, promoters). Manually identifying these is resource-intensive. This project automates the classification using sequence-based features.

## Results & Performance
- **Dataset:** 280k+ genomic sequences.
- **Top Model:** MLP (Achieved X% Accuracy/F1-Score).
- **Key Insight:** Random Forest provided the best feature importance ranking for identifying regulatory motifs.

## Requirements
`pip install xgboost scikit-learn pandas opencv-python`
