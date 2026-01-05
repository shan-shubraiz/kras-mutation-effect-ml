# KRAS mutation effect prediction (Python + ML + Bioinformatics)

## Background
KRAS is a key oncogene; point mutations can alter signaling and cancer risk. This project builds a compact Python + machine learning pipeline to predict the functional impact (deleterious vs neutral) of KRAS mutations using sequence-derived features and visualizes high-impact sites on the KRAS structure.

## Methods
- Data: Starter CSV of KRAS mutations (expandable to ClinVar/cBioPortal).
- Features:
  - Physicochemical properties (hydrophobicity, charge, polarity, volume).
  - Substitution severity (Grantham distance, BLOSUM62 score).
  - Local sequence context (±2 window one-hot).
  - Position normalization.
- Model: Random Forest (scikit-learn) with cross-validated ROC-AUC.
- Visualization: Feature importance, predicted probabilities, PyMOL structure coloring.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn biopython matplotlib seaborn requests joblib
python src/train.py
python src/visualize.py
