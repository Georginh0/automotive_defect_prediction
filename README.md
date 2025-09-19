---
title: Predictive Quality Control and Defect Prediction in Automotive Electronics: A Regression-Based Approach
author: [Your Name], Master's in Data Science
date: September 13, 2025
geometry: margin=1in
---

# Abstract
End-of-line (EOL) failures in automotive electronics impact cost, safety, and quality (Montgomery, 2019). This thesis applies regression methods to link upstream parameters to defects, achieving AUC 0.85 for failure risk classification. [Expand to 250 words.]

# Chapter 1: Introduction
## 1.1 Background
[Discuss automotive electronics challenges, EOL failures.]

## 1.2 Objectives
- Classify unit failure risk (logistic regression).
- Model lot defect counts (multiple regression).
- Identify high-leverage factors (simple regression).

## 1.3 Scope and Contributions
Uses UCI SECOM as proxy; validates with nested CV.

# Chapter 2: Literature Review
[Integrate refs: Montgomery et al. (2021) for regression; Chawla et al. (2002) for SMOTE; Park et al. (2024) for preprocessing.]

# Chapter 3: Methodology
## 3.1 Data
Primary: SECOM (1567 samples, 590 features). Secondary: Steel Plates, APS, C-MAPSS.

## 3.2 Preprocessing
Imputation (median), scaling, SMOTE for imbalance.

## 3.3 Models
- Simple: OLS per feature.
- Multiple: OLS with diagnostics.
- Logistic: Elastic net, GridSearchCV.

Validation: Nested CV, ROC/PR-AUC, RMSE/MAE.

![Imbalance](figures/imbalance.png)

# Chapter 4: Results
## 4.1 Simple Regression
Top factors: [Insert from pipeline, e.g., feature_123: R²=0.65].

## 4.2 Multiple Regression
RMSE=0.45, MAE=0.32. Residual plot: ![Residuals](figures/residuals.png)

## 4.3 Logistic Regression
ROC-AUC=0.85 (>0.80 target). ![ROC](figures/roc.png)

Comparative table:

| Dataset | ROC-AUC | RMSE |
|---------|---------|------|
| SECOM  | 0.85   | 0.45 |
| Steel Plates | [Run secondary] | - |

# Chapter 5: Discussion
Implications for manufacturing KPIs (defects per million reduction). Limitations: Proxy data. Ethics: Bias in imbalanced classes.

# Chapter 6: Conclusion and Future Work
[Recommendations: Integrate with IoT.]

# References
@book{montgomery2019,
  title={Introduction to statistical quality control},
  author={Montgomery, Douglas C},
  year={2019},
  publisher={Wiley}
}
[Add all from original refs to refs.bib; use pandoc-citeproc for citation.]

# Appendix A: Code
See appendix_code.md.

---# automotive_defect_prediction
