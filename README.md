# iAMP-SeE: A High-Accuracy Antimicrobial Peptide Recognition Model Based on ESM2 Feature Extraction and SeE Attention Mechanism

## Dataset Information
### Dataset source
<img width="2000" height="1530" alt="Figure2" src="https://github.com/user-attachments/assets/96b3fda9-61fd-4d9e-a47b-11132eb3bacc" />
In dataset 1, the positive samples were collected from five public antimicrobial peptide databases: RAMP, dbAMP, CAMPr-4, AMPfun, and ADAPBLE. The negative samples were sourced from the UniProt database.
Dataset 2 is sourced from the literature referenced below：
[1]Jun Zhao,Hangcheng Liu,Leyao Kang,Wanling Gao,Quan Lu,Yuan Rao & Zhenyu Yue.(2025).deep-AMPpred: A Deep Learning Method for Identifying Antimicrobial Peptides and Their Functional Activities..Journal of chemical information and modeling, 

## Requirements

### Core Dependencies
| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Base programming language |
| TensorFlow | 2.8.0 | Deep learning framework |
| scikit-learn | 1.0.2 | Machine learning utilities |
| pandas | 1.4.2 | Data processing and analysis |
| numpy | 1.22.3 | Numerical computations |
| matplotlib | 3.5.1 | Visualization (for reproducing charts) |

### Optional Dependencies
| Library | Version | Purpose |
|---------|---------|---------|
| seaborn | 0.11.2 | Enhanced visualizations |
| imbalanced-learn | 0.8.1 | Handling class imbalance (if needed) |
| tqdm | 4.62.3 | Progress bars for long operations |

### Installation
1. **Base installation** (required):
