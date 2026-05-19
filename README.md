# GeneScope AI
### Interpretable Deep Learning Pipeline for Genomic Variant Pathogenicity Prediction

A full end-to-end deep learning pipeline that predicts whether human genomic variants (SNVs) are pathogenic or benign, using local DNA sequence context and convolutional neural networks — with gradient-based saliency maps for interpretability.

Built independently as a self-directed research project exploring computational genomics and deep learning.

---

## Overview

Determining whether a genetic mutation is harmful or harmless is a central challenge in clinical genomics. This project tackles that problem using only the raw DNA sequence surrounding a mutation — no hand-crafted features, no external annotations.

The pipeline goes from raw ClinVar variant data and GRCh38 reference genome FASTA files all the way to a trained model with explainability analysis, including failure diagnosis and saliency-based interpretation of predictions.

---

## Pipeline Summary

```
ClinVar variant_summary.txt
        ↓
  Filtering & QC (SNVs, GRCh38, Pathogenic/Benign labels)
        ↓
  FASTA Integration (GRCh38 chromosome sequences)
        ↓
  101bp sequence window extraction + reference allele validation
        ↓
  Wild-type / Mutant sequence pair generation
        ↓
  One-hot encoding → 12-channel tensor representation
        ↓
  Two-branch CNN (context branch + mutation-difference branch)
        ↓
  Chromosome-aware train/val/test split
        ↓
  Evaluation + Gradient Saliency Map Analysis
```

---

## Dataset

- **Source:** ClinVar `variant_summary.txt` (public, NCBI) + GRCh38 reference genome (Ensembl)
- **Variant types:** Single nucleotide variants (SNVs) only
- **Labels:** Pathogenic (1) vs Benign (0)
- **Initial pool:** ~63,000 filtered SNVs → balanced 10,000 sample subset (5,000 per class)
- **After QC:** 6,256 high-confidence samples (reference allele validated, no ambiguous bases, bounds-checked)
- **Chromosomes used:** 1, 2, 3, 5, 7, 11, 16, 17, 19, X

---

## Sequence Representation

Each variant is represented as a **12-channel tensor** of shape `(101, 12)`:

| Channels | Description |
|----------|-------------|
| 0–3 | One-hot encoded wild-type sequence (A/C/G/T) |
| 4–7 | One-hot encoded mutant sequence |
| 8–11 | Difference channels (mutant − wild-type), scaled ×3 |

This lets the model simultaneously learn sequence context and explicitly focus on the mutation change itself.

---

## Model Architecture

A two-branch 1D CNN that processes context and mutation signal separately before merging:

```
Input (101, 12)
    ├── Context Branch  → channels 0–7 (wild-type + mutant)
    │       Conv1D(32, kernel=7) → BatchNorm → GlobalAveragePooling
    │
    └── Mutation Branch → channels 8–11 (difference)
            Conv1D(32, kernel=3) → BatchNorm → GlobalMaxPooling
                        ↓
                   Concatenate (64-dim)
                        ↓
              Dense(64, L2) → Dropout(0.25) → Dense(32, L2) → Dense(1, sigmoid)
```

**Total parameters:** 8,769

**Training details:**
- Chromosome-aware split: Train on chr 1,2,3,5,7,11 | Val on chr 16,17 | Test on chr 19,X
- Reverse-complement augmentation on training sequences
- Class weighting (1.6× for benign) to handle label imbalance
- Adam optimizer, binary crossentropy loss
- EarlyStopping + ReduceLROnPlateau callbacks

---

## Results

Evaluated on held-out chromosomes 19 and X (never seen during training):

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.6539 |
| PR-AUC | 0.7407 |
| Balanced Accuracy | 0.6317 |
| Accuracy | 0.6055 |
| Pathogenic F1 | 0.6134 |
| Benign F1 | 0.5974 |

> **Note on performance:** The model learns generalizable sequence-level signal across chromosomes it has never seen. The moderate accuracy reflects a deliberate design choice — the model uses *only* local 101bp sequence context, with no protein annotations, conservation scores, or functional features. This is intentionally a minimal baseline to isolate what raw sequence alone can capture.

---

## Explainability — Gradient Saliency Maps

Using TensorFlow `GradientTape`, the pipeline computes per-position input gradients to identify which nucleotides most influenced each prediction.

Saliency analysis was run across four prediction categories:
- Correct pathogenic predictions
- Correct benign predictions  
- False positives (benign variants predicted as pathogenic)
- False negatives (pathogenic variants predicted as benign)

The mutation site (position 50, center of the 101bp window) is marked in all visualizations. Saliency distributions reveal which flanking sequence regions the model weights most heavily — and where it fails.

---

## Key Design Decisions

**Why chromosome-aware splitting instead of random splitting?**  
Random splitting causes genomic data leakage — nearby variants on the same chromosome share sequence context, inflating test performance. Holding out entire chromosomes gives a more honest measure of generalization.

**Why the difference channel?**  
The mutation itself changes only one nucleotide out of 101. Without an explicit difference signal, the model has to discover the mutation position on its own. Scaling the difference channels (×3) makes the mutation site immediately salient.

**Why reverse-complement augmentation?**  
DNA is double-stranded and strand-symmetric. The same variant has the same biological effect regardless of which strand you read. Augmenting with reverse complements effectively doubles the training data and improves strand-invariant generalization.

---

## Limitations & Future Work

- Model uses only local 101bp sequence context — no splicing signals, conservation, or functional annotations
- Trained on ClinVar variants, which has known ascertainment bias toward clinically studied genes
- Extending to include features from tools like CADD, PhyloP, or protein consequence annotations would likely improve performance significantly
- Attention mechanisms or transformer-based architectures could better capture long-range sequence dependencies

---

## Tech Stack

- Python 3.12
- TensorFlow / Keras
- NumPy, Pandas
- scikit-learn
- Matplotlib
- pyfaidx, tqdm

---

## Project Structure

```
GeneScope-AI/
│
├── GeneAI_Final_Model.ipynb    # Complete pipeline (all phases)
└── README.md
```

> The full pipeline — preprocessing, model training, evaluation, and saliency analysis — is contained in a single self-documented notebook with clear phase headers.

---

## Data Sources

- **ClinVar:** https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
- **GRCh38 FASTA:** https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/

Both are publicly available. Data files are not included in this repository due to size (~400MB+). The notebook downloads them automatically.

---

*Built by Anusha Dalal — 2nd year Computer Engineering student exploring computational genomics and deep learning.*
