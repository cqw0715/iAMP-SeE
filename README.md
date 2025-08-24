# iAMP-SeE: An Antimicrobial Peptide Recognition Model Based on ESM2 Feature Extraction and Hybrid Attention Mechanisms

## Dataset
<img width="2000" height="1530" alt="Figure2" src="https://github.com/user-attachments/assets/96b3fda9-61fd-4d9e-a47b-11132eb3bacc" />
In dataset 1, the positive samples were collected from five public antimicrobial peptide databases: RAMP, dbAMP, CAMPr-4, AMPfun, and ADAPBLE. The negative samples were sourced from the UniProt database.
Dataset 2 is sourced from the literature referenced below：
[1]Jun Zhao,Hangcheng Liu,Leyao Kang,Wanling Gao,Quan Lu,Yuan Rao & Zhenyu Yue.(2025).deep-AMPpred: A Deep Learning Method for Identifying Antimicrobial Peptides and Their Functional Activities..Journal of chemical information and modeling, 

## Core Dependencies
| Library          | Version  |
|------------------|----------|
| numpy            | ≥1.20.0  |
| pandas           | ≥1.3.0   |
| scikit-learn     | ≥1.0.0   |
| tensorflow       | ≥2.6.0   |
| matplotlib       | ≥3.5.0   |
| torch            | ≥1.10.0  |
| esm              | ≥0.5.0   |
| imbalanced-learn | ≥0.9.0   |

iAMP-SeEModel1.py is used for the binary classification task of AMPs, while iAMP-SeEModel2.py is utilized for the multi-class classification task of AMPs. The corresponding datasets are All32400.csv and All2_16200.csv, respectively.
