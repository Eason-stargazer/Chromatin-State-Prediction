"""Chromatin State 18730 Demo """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

df_label = pd.read_csv("trainlabels.csv", names = ["labels"])
df1 = pd.read_csv("kmer_counts1.csv")
df2 = pd.read_csv("kmer_counts2.csv")
df3 = pd.read_csv("kmer_counts3.csv")
df4 = pd.read_csv("kmer_counts4.csv")

df = pd.concat([df4,df3,df2,df1], axis = 1)

df_seq = pd.read_csv("trainsequences.csv", names=["sequence"])

# List of motifs to count
motifs = [
# GC-rich / CpG / GC-repeat kmers
"CCGCGG",
"CGCGCG",
"GGGCGG",
"CGGGGC",
"GCCGCA",
"GGCGCC",
"CGCCCG",
"GGGGCGGGGC",
"GCGCGCGC",
"CCGCCCC",
"GGGGCGG",

# Restriction enzyme–like sites
"GAATTC",   # EcoRI
"GTCGAC",   # SalI
"CTGCAG",   # PstI
"CAGCTG",   # PvuII (palindromic)
"GGTGAG",
"CCAGGT",


# AT-rich / low-complexity kmers
"AAAAA",
"AAAAAA",
"TTTTT",
"TTTTTT",
"TATAAA",
"TATATA",
"ATATATAT",
"AAAATA",
"TATTTT",

# Transcription factor motifs
"TGACTCA",  # AP-1
"TGAGTCA",  # AP-1 variant
"CACGTG",   # E-box
"CCAAT",    # CCAAT box
"ACGTCA",
"AGGTCA",   # nuclear receptor half-site
"TCAGTC",
"TGTTTGT",
"CATTGTT",

# GC-balanced kmers
"GTCAAC",
"GTTAAC",
"GTTGAC",
"ATGCAAAT",
"GGGACTTTCC",
"TTGCGCAA",
"GATAAG",
"CACCTG",
"GCCAT",
"CCAGTT",
"CCTGG",

# Long GC-heavy motifs
"CCACCAGGGGGCG",
"CCGCCCCCTGGTG",

# Homopolymer GC kmers
"GGGGGG",
"CCCCCC",

# Reverse-complement related pairs
"AAAGGT",
"CCAGCC",
]

def compute_features(seq):
    L = len(seq)
    counts = []
    for m in motifs:
        k = len(m)
        # For single nucleotides, normalize by sequence length
        if k == 1:
            counts.append(seq.count(m) / L if L > 0 else 0)
        # For dinucleotides, normalize by L-1
        elif k == 2:
            counts.append(seq.count(m) / (L - 1) if L > 1 else 0)
        # For longer motifs, just count occurrences (or normalize if desired)
        else:
            counts.append(seq.count(m)/(L - k + 1))
    return counts

tf_features = np.array([compute_features(s) for s in df_seq["sequence"]], )

def compute_cg_features(seq):
    L = len(seq)
    return [
        seq.count("C") / L,
        seq.count("G") / L,
        seq.count("CG") / (L - 1),
        seq.count("GC") / (L - 1),
        (seq.count("G")-seq.count("C")) / (seq.count("C")+seq.count("G") + 1e-6),
        seq.count("A") / L,
        seq.count("T") / L,
        (seq.count("C") + seq.count("G"))/ L,
        (seq.count("A") + seq.count("C"))/(seq.count("G") + seq.count("T") + 1e-6),
        (seq.count("A") + seq.count("G"))/(seq.count("C") + seq.count("T") + 1e-6),
        seq.count("CG") * L / (seq.count("C") * seq.count("G") + 1e-6)
    ]

cg_features = np.array(
    [compute_cg_features(s) for s in df_seq["sequence"]]
)

cg_df = pd.DataFrame(
    cg_features,
    columns=["C_frac", "G_frac", "CG_frac", "GC_frac", "GC_skew", "A_frac",
             "T_frac", "C&G_frac", "AC_GT_ratio", "AG_CT_ratio", "CpG_ratio"]
)

import math
from collections import Counter
def compute_entropy(seq):
  if not seq:
    return 0

  counts = Counter(seq)
  seq_len = len(seq)

  entropy = 0
  for char in "ACGT":
    p = counts[char] / seq_len
    if p > 0:
      entropy -= p * math.log2(p)
  return entropy

entropy_features = np.array([compute_entropy(s) for s in df_seq["sequence"]])
entropy_df = pd.DataFrame(
    entropy_features,
    columns=["entropy"]
)

#the longest continuous nucleotide group eg. AAAAA
def compute_max_run(seq):
  max_run = 0
  current_run = 1
  for i in range(1, len(seq)):
      if seq[i] == seq[i-1]:
          current_run += 1
      else:
          max_run = max(max_run, current_run)
          current_run = 1
  max_run = max(max_run, current_run)
  return max_run

max_run_features = np.array([compute_max_run(s) for s in df_seq["sequence"]])
max_run_df = pd.DataFrame(
    max_run_features,
    columns=["max_run"]
  )

tf_df = pd.DataFrame(
    tf_features,
    columns=[
"CCGCGG",
"CGCGCG",
"GGGCGG",
"CGGGGC",
"GCCGCA",
"GGCGCC",
"CGCCCG",
"GGGGCGGGGC",
"GCGCGCGC",
"CCGCCCC",
"GGGGCGG",

"GAATTC",   
"GTCGAC",   
"CTGCAG",  
"CAGCTG",  
"GGTGAG",
"CCAGGT",

"AAAAA",
"AAAAAA",
"TTTTT",
"TTTTTT",
"TATAAA",
"TATATA",
"ATATATAT",
"AAAATA",
"TATTTT",

"TGACTCA",  
"TGAGTCA",  
"CACGTG",  
"CCAAT", 
"ACGTCA",
"AGGTCA",  
"TCAGTC",
"TGTTTGT",
"CATTGTT",

"GTCAAC",
"GTTAAC",
"GTTGAC",
"ATGCAAAT",
"GGGACTTTCC",
"TTGCGCAA",
"GATAAG",
"CACCTG",
"GCCAT",
"CCAGTT",
"CCTGG",

"CCACCAGGGGGCG",
"CCGCCCCCTGGTG",

"GGGGGG",
"CCCCCC",

"AAAGGT",
"CCAGCC",
])


df = pd.concat([df, max_run_df, entropy_df, cg_df, tf_df, df_label], axis=1)
df.head(20)

from sklearn.feature_selection import VarianceThreshold

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y  # stratify must match y
    )

    y_train_shifted = y_train - 1
    y_val_shifted = y_val - 1


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if oversample:
        ros = RandomOverSampler(random_state=42)
        X_train_scaled, y_train_shifted = ros.fit_resample(
            X_train_scaled,
            y_train_shifted
        )

    return X_train_scaled, X_val_scaled, y_train_shifted, y_val_shifted, scaler

X_train_scaled, X_val_scaled, y_train, y_val, scaler = scale_dataset(df, oversample=False)

## MLP Model ##
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight = dict(zip(classes, weights))

# 1. Determine the number of input features and output classes
input_shape = X_train_scaled.shape[1]
# Assuming 18 classes based on your previous data stratification
num_classes = int(len(np.unique(y_train)))

# 2. Build the Model Architecture

nn_model = models.Sequential([
    # Input layer + first hidden layer
    layers.Dense(512, input_shape=(input_shape,)),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.1), # Using LeakyReLU to prevent "dead neurons"
    layers.Dropout(0.4),

    # Second hidden layer
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.1),
    layers.Dropout(0.3),

    # Third hidden layer
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.1),

    # Output layer (Softmax is used for multi-class classification)
    layers.Dense(num_classes, activation='softmax')
])

# 3. Compile the Model
nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train the Model
print("Starting Neural Network training...")
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = nn_model.fit(
    X_train_scaled,
    y_train,
    epochs=13,       
    batch_size=64,   
    validation_data=(X_val_scaled, y_val),
    class_weight=class_weight,
    verbose=1
)

# 5. Evaluate
val_loss, val_acc = nn_model.evaluate(X_val_scaled, y_val, verbose=0)
print(f"\nNeural Network Validation Accuracy: {val_acc:.4f}")

# Generate a confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Generate predictions [cite: 115, 181]
y_val_pred = np.argmax(nn_model.predict(X_val_scaled), axis=1)

# 2. Compute confusion matrix [cite: 72]
cm = confusion_matrix(y_val, y_val_pred)

# 3. Create a professional plot
fig, ax = plt.subplots(figsize=(14, 12)) # Increase size to prevent overlap

# Normalize by row (True Label) to see percentages
disp = ConfusionMatrixDisplay.from_predictions(
    y_val,
    y_val_pred,
    display_labels=classes,
    cmap=plt.cm.Blues,
    xticks_rotation=45, 
    values_format='.2f', 
    normalize='true',
    ax=ax
)

plt.title('Normalized Confusion Matrix: Chromatin State Prediction', fontsize=16)
plt.show()

df_test_seq = pd.read_csv("testsequences.csv", names=["sequence"])

test_tf_features = np.array([
    compute_features(s) for s in df_test_seq["sequence"]
])

test_tf_df = pd.DataFrame(
    test_tf_features,
    columns=motifs
)

test_cg_features = np.array(
    [compute_cg_features(s) for s in df_test_seq["sequence"]]
)

test_cg_df = pd.DataFrame(
    test_cg_features,
    columns=["C_frac", "G_frac", "CG_frac", "GC_frac", "GC_skew", "A_frac",
             "T_frac", "C&G_frac", "AC_GT_ratio", "AG_CT_ratio", "CpG_ratio"]
)

test_entropy_features = np.array([compute_entropy(s) for s in df_test_seq["sequence"]])

test_entropy_df = pd.DataFrame(
    test_entropy_features,
    columns=["entropy"]
)

test_max_run_features = np.array([compute_max_run(s) for s in df_test_seq["sequence"]])

test_max_run_df = pd.DataFrame(
    test_max_run_features,
    columns=["max_run"])

test_df1 = pd.read_csv("kmer_test_counts1.csv")
test_df2 = pd.read_csv("kmer_test_counts2.csv")
test_df3 = pd.read_csv("kmer_test_counts3.csv")
test_df4 = pd.read_csv("kmer_test_counts4.csv")

X_test = pd.concat(
    [test_df4, test_df3, test_df2, test_df1, test_max_run_df, test_entropy_df, test_cg_df, test_tf_df],
    axis=1
).values

X_test_scaled = scaler.transform(X_test)

y_test_pred = np.argmax(nn_model.predict(X_test_scaled), axis=1)
y_test_pred_final = y_test_pred + 1

submission = pd.DataFrame({
    "predicted_state": y_test_pred_final
})

submission.to_csv(
    "predictions.csv",
    index=False,
    header=False
)
