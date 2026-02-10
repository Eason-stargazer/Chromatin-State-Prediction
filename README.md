This project predicts 18 chromatin states (such as promoters and enhancers) directly from DNA sequence data using machine learning models. Chromatin data is unavailable for many cell types, whereas sequence-based prediction provides a scalable way to annotate genomes and support human pangenome research.

## Models
-Logistic Regression
-Random Forest
-Gradient Boosting
-Multilayer Perceptron (MLP)
-Convolutional Neural Network (CNN)

## Dataset
-280k+ training sequences
-100k+ test sequences

## Results
- **Top Model:** MLP achieved 18% accuracy.
- **Context** The leading performance in this task is approximately 20% accuracy, placing our MLP models within the top tier of expected performance for this dataset.
- **Key Insight:** The primary bottleneck is the spatial dimensionality of the genome. Current models rely on 1D sequence data or localized markers, which do not fully capture the 3D chromatin architecture that determines state transitions.

## Requirements
```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn tensorflow
