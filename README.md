# iAMP-SeE: A High-Accuracy Antimicrobial Peptide Recognition Model Based on ESM2 Feature Extraction and SeE Attention Mechanism

## Dataset Information
### Dataset source
<img width="2000" height="1530" alt="Figure2" src="https://github.com/user-attachments/assets/96b3fda9-61fd-4d9e-a47b-11132eb3bacc" />
In dataset 1, the positive samples were collected from five public antimicrobial peptide databases: RAMP, dbAMP, CAMPr-4, AMPfun, and ADAPBLE. The negative samples were sourced from the UniProt database.
Dataset 2 is sourced from the literature referenced below：
[1]Jun Zhao,Hangcheng Liu,Leyao Kang,Wanling Gao,Quan Lu,Yuan Rao & Zhenyu Yue.(2025).deep-AMPpred: A Deep Learning Method for Identifying Antimicrobial Peptides and Their Functional Activities..Journal of chemical information and modeling, 

pip install imbalanced-learn tqdm

### Key Features

- **Binary Classification (document1.py)**:
- ESM-2 feature extraction with caching
- Dual attention mechanism (SE + ECA)
- 10-fold cross validation
- Comprehensive binary metrics (Accuracy, Sensitivity, Specificity, MCC, F1, AUC)

- **Multi-class Classification (document2.py)**:
- Robust oversampling with fallback strategies
- Multi-class ESM feature extraction
- Dual attention mechanism (SE + ECA)
- Extended 10-fold cross validation
- Multi-class metrics (Micro/Macro F1, Hamming Loss, MAP, ROC AUC OVR/OVO)
- Per-class performance analysis

### Output Files

Both scripts will generate:
- `iAMP-SeE.csv`: Detailed cross-validation results
- `iAMP-SeE.png` (document1.py): ROC curve visualization
- `iAMP-SeE_results.txt` (document2.py): Summary metrics and per-class performance

### Notes

1. The ESM model requires significant GPU memory (recommended 12GB+)
2. First run will be slower due to ESM feature extraction (subsequent runs use cached features)
3. For multi-class classification, ensure labels are properly encoded (0-4 for 5 classes)
