# GeneScope AI
### Chromosome-Aware CNN for Genomic Variant Pathogenicity Prediction

> *Can a neural network learn to distinguish disease-causing mutations from benign ones using only the DNA sequence surrounding the variant — with no evolutionary scores, no protein annotations, and no prior biological knowledge?*

---

## Overview

GeneScope AI is a deep learning pipeline that predicts whether a single nucleotide variant (SNV) is pathogenic or benign using only the 101 base-pair sequence window surrounding the mutation site. Trained on 40,000 ClinVar variants mapped to the GRCh38 human reference genome, the model uses a dual-branch 1D Convolutional Neural Network to learn sequence motifs that distinguish disease-causing from benign mutations — without access to evolutionary conservation scores, protein structure, or functional annotations.

A key design choice of this project is **chromosome-aware splitting**: training and test chromosomes are entirely separate, preventing any biological information leakage between variants from the same genomic region.

---

## Motivation

Most genomic variant interpretation tools rely on pre-computed conservation scores (CADD, SIFT, PolyPhen) that encode evolutionary and functional knowledge accumulated over decades. GeneScope asks a simpler question: how much signal lives in the raw sequence context alone? Exploring the predictive power of local sequence context alone is a prerequisite for knowing when additional annotations are actually necessary.

---

## Dataset

| Property | Value |
|---|---|
| Source | ClinVar variant_summary (NCBI FTP, GRCh38) |
| Total variants | 264,526 SNVs (Pathogenic + Benign, strict labels only) |
| Balanced sample | 20,000 Pathogenic + 20,000 Benign = 40,000 |
| Sequence window | 101 bp centred on variant position |
| Reference genome | GRCh38 (Ensembl chromosome FASTA files) |
| Chromosomes used | 1, 2, 3, 5, 7, 11, 16, 17, 19, X |

**Data quality filters applied:**
- Single nucleotide variants only (ref ≠ alt, both single ACGT bases)
- Strict ClinVar labels only — "Pathogenic" and "Benign" (no VUS, likely pathogenic, conflicting)
- Reference allele verified against GRCh38 at extracted position
- Ambiguous N bases excluded
- Out-of-bounds positions excluded
- Duplicates removed by (chromosome, position, ref, alt)

Final retained: **40,000 / 40,000 variants (100% pass rate)**

---

## Chromosome-Aware Splitting

To prevent genomic leakage, chromosome-level train/validation/test separation was enforced — ensuring no variants from test chromosomes (19 and X) were observed during training or threshold calibration. Variants from the same chromosomal region share local sequence context, regulatory elements, and repeat structures; random splitting would allow the model to memorise chromosomal patterns rather than learning generalisable sequence features.

| Split | Chromosomes | Variants |
|---|---|---|
| Train | 1, 2, 3, 5, 7, 11 | 24,820 × 2 (with augmentation) = 49,640 |
| Validation | 16, 17 | 7,626 |
| Test | 19, X | 7,554 |

No variant from chromosome 19 or X is seen during training or threshold calibration.

---

## Architecture

A dual-branch 1D CNN operating on a 12-channel input tensor:

```
Input: (101, 12)
├── Channels 0–3:   Wild-type sequence (one-hot encoded)
├── Channels 4–7:   Mutant sequence (one-hot encoded)
└── Channels 8–11:  Difference (mutant − wild-type, scaled ×3)
         │
         ├── Context Branch [channels 0–7]
         │   Conv1D(32, kernel=7, padding=same) → BatchNorm → GlobalAveragePooling1D
         │
         └── Mutation Branch [channels 8–11]
             Conv1D(32, kernel=3, padding=same) → BatchNorm → GlobalMaxPooling1D
                          │
                    Concatenate → Dense(64, L2) → Dropout(0.25) → Dense(32, L2) → Dense(1, sigmoid)
```

**Design rationale:**
- **Context branch** uses kernel size 7 to capture local sequence motifs spanning multiple nucleotides — splice site signals, regulatory elements, and trinucleotide context are typically 5–8bp wide
- **Mutation branch** uses kernel size 3 on the difference signal to focus specifically on the positional change introduced by the mutation
- **Difference channel scaled ×3** amplifies the mutation signal relative to the background sequence noise
- **GlobalAveragePooling + GlobalMaxPooling** combination captures both average sequence character and peak signal strength
- **L2 regularisation** on both Dense layers prevents overfitting on the relatively small training set

---

## Training

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam (lr=0.001) |
| Loss | Binary cross-entropy |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping | val_pr_auc, patience=10 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Class weights | 1.0 / 1.0 (balanced dataset) |

**Data augmentation:** Reverse complement of each training sequence added — doubles training set to 49,640. Reverse complement augmentation respects the double-stranded nature of DNA; a variant on the forward strand has an equivalent representation on the reverse strand. Not applied to validation or test sets.

**Threshold calibration:** Default sigmoid threshold of 0.5 produces excessive false positives on this imbalanced task. Threshold calibrated on validation chromosomes (16, 17) by optimising balanced accuracy (mean of sensitivity and specificity) across 200 threshold values. **Best threshold: 0.3510** — saved to file and applied to test evaluation without further tuning.

## Results

Evaluated on chromosomes 19 and X — completely withheld throughout training and calibration.

| Metric | Value |
|---|---|
| **PR-AUC** | **0.8349** |
| ROC-AUC | 0.8105 |
| Balanced Accuracy | 0.7363 |
| False Negatives | 1,061 |
| False Positives | 923 |
| Test Set Size | 7,554 variants |

**PR-AUC is the primary metric.** PR-AUC is reported as the primary metric because pathogenic variant detection is typically the clinically important class, and PR-AUC better reflects performance under class imbalance encountered in real-world variant databases.

### Key Result

Despite using only local DNA sequence context and no evolutionary or functional annotations, GeneScope achieved a PR-AUC of 0.8349 and ROC-AUC of 0.8105 on completely unseen chromosomes, demonstrating that pathogenic variants exhibit detectable local sequence signatures.

## Confusion Matrix

**Test Set (Chromosomes 19 and X)**

|                     | Predicted Benign | Predicted Pathogenic |
|---------------------|----------------:|---------------------:|
| **True Benign**     | 2449 | 923 |
| **True Pathogenic** | 1061 | 3121 |

### Derived Metrics

- Sensitivity (Recall): **74.63%**
- Specificity: **72.63%**
- Balanced Accuracy: **73.63%**

The confusion matrix is reported on chromosomes 19 and X, which were completely excluded from both training and validation. This provides a chromosome-level evaluation of model generalization and helps reduce genomic data leakage compared with random variant splitting.

## Example Saliency Maps

Representative examples of:

- Correct Pathogenic Prediction
- Correct Benign Prediction
- High Confidence False Positive
- High Confidence False Negative

These visualizations demonstrate that the model consistently attends to the mutation position and nearby sequence context.

---

## Explainability — Gradient Saliency

Gradient saliency maps are computed via GradientTape, identifying which nucleotide positions in the 101bp window most influence the model's prediction. For a correct pathogenic prediction, saliency is expected to concentrate at and immediately around the mutation site (position 50) — reflecting that the model has learned the local sequence context around the variant matters.

**Representative cases:**

| Case | True Label | Predicted Probability | Interpretation |
|---|---|---|---|
| Correct Pathogenic | 1 | 0.4671 | Sharp saliency peak at mutation site — model attending to local context |
Correct Benign | 0 | 0.2673 | Saliency includes the mutation site but lacks the strong localized activation observed in many pathogenic predictions
| High Confidence False Positive | 0 | 0.9409 | Mutation site in a sequence context that closely resembles known pathogenic regions — model fooled by context similarity |
| High Confidence False Negative | 1 | 0.0173 | Saliency concentrated at mutation site but probability very low — pathogenicity likely depends on protein structure or evolutionary context invisible to a 101bp window |

The false negative pattern is the most informative: the model is attending to the right position but cannot classify the variant correctly. This confirms the fundamental limitation of sequence-only prediction — some pathogenic variants require protein-level or conservation-level context that 101bp cannot encode.

---

## Limitations

1. **Sequence-only model.** No evolutionary conservation scores (CADD, PhyloP), protein structure, splice site predictions, or regulatory annotations. These are the features that state-of-the-art variant interpretation tools rely on most heavily.

2. **Local context only.** 101bp window cannot capture long-range regulatory interactions, enhancer-promoter loops, or trans-regulatory effects.

3. **ClinVar label quality.** ClinVar contains reclassified variants over time. "Benign" and "Pathogenic" labels reflect the state of evidence at download time — some may be reclassified in future database versions.

4. **Chromosome-aware split is conservative.** By witholding entire chromosomes, the test set may have a different class distribution and sequence composition than training chromosomes. This represents a harder evaluation than random splitting and may underestimate performance on same-chromosome generalisation.

5. **Predicts pathogenicity likelihood from sequence context, not biological mechanism.** The model cannot explain why a variant is pathogenic — only that its local sequence context resembles other pathogenic variants in ClinVar.

---

## Repository Structure

```
GeneScope-AI/
├── GeneScopeAI_Final_Model.ipynb    # Complete pipeline notebook
├── figures/
│   ├── test_case_correct_benign.png
│   ├── test_case_correct_pathogenic.png
│   ├── test_case_false_positive.png
│   └── test_case_false_negative.png
└── README.md
```

---

## Pipeline Summary

```
ClinVar variant_summary.txt.gz (NCBI FTP)
          │
          ▼
Filter: SNVs only, GRCh38, strict Pathogenic/Benign labels
          │
          ▼
Balance: 20,000 Pathogenic + 20,000 Benign
          │
          ▼
GRCh38 FASTA reference → extract 101bp windows
          │
          ▼
QC: verify ref allele, exclude N bases, bounds check
          │
          ▼
Chromosome-aware split: Train (1,2,3,5,7,11) | Val (16,17) | Test (19,X)
          │
          ▼
Encode: 12-channel tensor (WT + MT + Diff×3)
          │
          ▼
Augment: reverse complement (training only)
          │
          ▼
Dual-branch 1D CNN → threshold calibration on Val → evaluate on Test
          │
          ▼
Gradient saliency → per-case interpretability
```

---

## Technologies

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model architecture, training, GradientTape saliency |
| pandas | ClinVar data loading and filtering |
| numpy | Tensor construction and encoding |
| gzip / Python stdlib | GRCh38 FASTA loading via manual streaming parser |
| scikit-learn | Threshold calibration, confusion matrix |
| matplotlib | Saliency visualisation |

---

## Academic Context

This project is a proof-of-concept exploring the predictive power of local DNA sequence context for pathogenicity prediction using deep learning. For clinical variant interpretation, tools incorporating evolutionary conservation (CADD, SIFT, PolyPhen) and functional annotations are recommended.

**Data source:** Landrum MJ et al. ClinVar: improving access to variant interpretations and supporting evidence. Nucleic Acids Research. 2018.

**Reference genome:** Genome Reference Consortium Human Build 38 (GRCh38 / hg38).

---

*Built by Anusha Dalal — Computer Engineering, SPIT Mumbai (2024–2028)*
