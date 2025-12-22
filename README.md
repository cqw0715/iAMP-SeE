# iAMP-SeE: An Antimicrobial Peptide Recognition Model Based on ESM2 Feature Extraction and Hybrid Attention Mechanisms

## Dataset
<img width="800" height="500" alt="Figure1" src="https://github.com/user-attachments/assets/1ca00947-b244-4952-b5d2-b73b79f04b67" />
<br>
In dataset 1, the positive samples were collected from five public antimicrobial peptide databases: RAMP, dbAMP, CAMPr-4, AMPfun, and ADAPBLE. The negative samples were sourced from the UniProt database.
<br>
Dataset 2 is sourced from the literature referenced below：<br>
[1]Jun Zhao,Hangcheng Liu,Leyao Kang,Wanling Gao,Quan Lu,Yuan Rao & Zhenyu Yue.(2025).deep-AMPpred: A Deep Learning Method for Identifying Antimicrobial Peptides and Their Functional Activities..Journal of chemical information and modeling, 

## Core Dependencies
The main environment for the model operation is as follows：<br>
| Library          | Version  |
|------------------|----------|
| tensorflow       | 2.13.0   |
| torch            | 2.9.1    |
| fair-esm         | 2.0.0    |
| numpy            | 1.23.5   |<br>

I have uploaded the complete environment for running the model locally to the requirements.txt file. You can refer to this file for installation.

## Model execution
### 1. Dataset Preparation
Please use a single CSV file to store the dataset for model execution. The file must include the following two columns:<br>
1. 'sequence': for storing protein sequences.<br>
2. 'label': for recording the class label of the protein sequence. Labels should start from 0.<br>
<br>
For binary classification of AMPs, the labels should be limited to 0 or 1. Please download the model code: iAMP-SeE_Model_1.py, and replace "data1.csv" on line 316 with the name of your dataset.
<br>
For multiclass classification of AMPs, labels should be 0-n. Please download the model code: iAMP-SeE_Model_2.py, replace "data2.csv" on line 323 with the name of your dataset, and update the parameters accordingly: change 'num_classes=5' to 'num_classes=n' on lines 200 and 254, and replace 'depth=5' with 'depth=n' on lines 281, 285, 290, and 300.

iAMP-SeEModel1.py is used for the binary classification task of AMPs, while iAMP-SeEModel2.py is utilized for the multi-class classification task of AMPs. The corresponding datasets are All32400.csv and All2_16200.csv, respectively.
