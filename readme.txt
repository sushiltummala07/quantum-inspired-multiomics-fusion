# Quantum-Inspired Multi-Omics Fusion Framework

## Overview

This project presents an interpretable multimodal deep learning framework for disease classification using:

- Genomics
- Transcriptomics
- Proteomics

The architecture combines:
- quantum-inspired latent representations,
- attention-guided modality gating,
- multimodal fusion,
- and explainable AI techniques.

The framework dynamically learns modality importance while maintaining biological interpretability through SHAP analysis and permutation importance.

---

## Key Features

- Quantum-inspired feature embeddings
- Attention-guided modality weighting
- Multi-omics fusion
- SHAP explainability
- Permutation feature importance
- ROC / PR analysis
- Calibration analysis
- Stratified 5-fold cross-validation
- Baseline comparison with classical ML models

---

## Architecture

Pipeline:

```text
Genomics ─┐
          │
Transcriptomics ─► Modality Encoders ─► Quantum-Inspired Embeddings
          │
Proteomics ─┘
                    ↓
          Attention-Guided Gating
                    ↓
             Fusion Layer
                    ↓
          Classification Head
                    ↓
             Disease Prediction
## 📄 License

MIT License

---

## 👨‍💻 Author

Sushil Tummala
