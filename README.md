# Chromatin State Prediction
This project predicts 18 chromatin states (such as promoters and enhancers) directly from DNA sequence data using machine learning models. Chromatin data is unavailable for many cell types, whereas sequence-based prediction provides a scalable way to annotate genomes and support human pangenome research.

## Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)

## Dataset
- 280k+ training sequences
- 100k+ test sequences

## Results
- **Top Model:** MLP achieved 18% accuracy.
- **Context:** The leading performance in this task is approximately 20% accuracy, placing our MLP models within the top tier of expected performance for this dataset.
- **Key Insight:** The primary bottleneck is the spatial dimensionality of the genome. Current models rely on 1D sequence data or localized markers, which do not fully capture the 3D chromatin architecture that determines state transitions.

## Project Structure
**data/:** Directory containing the raw and processed genomic datasets.
- `kmer_train_counts1/2/3/4.csv`: Processed datasets containing raw k-mer counts used as the input features for the models.
- `testsequences.csv` & `trainlabels.csv`: The primary sequence data and their corresponding chromatin state labels.
- `kmer_count.py`: Script that parses raw DNA sequences to calculate the counts of specific k-length substrings.

**Model Implementation Scripts:**
- `rf_lr_gb.py`: Implementation and evaluation of traditional machine learning models: Random Forest, Logistic Regression, and Gradient Boosting.
- `mlp.py`: Implementation and evaluation of the Multi-Layer Perceptron (Neural Network) model.
- `cnn.py`: Implementation and evaluation of the one-hot encoding and CNN model.
- `predictions_result.csv`: The final output file containing the model's predicted chromatin states for the test dataset.

Due to GitHub's file size limitations, `kmer_train_counts3/4` and `trainsequences.csv` are hosted on Google Drive. You can download them [here](https://drive.google.com/drive/folders/1ho5tqd9btwwyApRLewMwwVa4awmCRQXH?)usp=sharing

## Requirements
```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn tensorflow
