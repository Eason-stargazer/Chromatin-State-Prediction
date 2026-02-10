# Generate files containing counts of 1-mer, 2-mer, 3-mer, and 4-mer sequences
import pandas as pd
from itertools import product

# Read sequences
df_seq = pd.read_csv("testsequences.csv", names=["sequence"])

k = 4

# Fixed list of all possible 2-mers
bases = ["A", "C", "G", "T"]
all_patterns = ["".join(p) for p in product(bases, repeat=k)]

# K-mer counting function
def count_kmers(sequence, patterns, k):
    counts = dict.fromkeys(patterns, 0)
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in counts:
            counts[kmer] += 1
    return counts

# Build k-mer count table
all_counts = []
for seq in df_seq["sequence"]:
    all_counts.append(count_kmers(seq, all_patterns, k))

out_df = pd.DataFrame(all_counts, columns=all_patterns)

# Save to CSV
out_df.to_csv("kmer_test_counts4.csv", index=False)
