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
For multiclass classification of AMPs, labels should be 0-n. Please download the model code: iAMP-SeE_Model_2.py, replace "data2.csv" on line 323 with the name of your dataset, and update the parameters accordingly: change 'num_classes=5' to 'num_classes=n+1' on lines 200 and 254, and replace 'depth=5' with 'depth=n+1' on lines 281, 285, 290, and 300.

### 2. Feature Extraction with ESM-2
In the feature extraction method, we primarily employ the ESM-2 approach, specifically using the version: esm2_t33_650M_UR50D. The corresponding version can be downloaded via the following link: https://zenodo.org/records/7566741.<br>
If your GPU has sufficient memory, the sequence features in the dataset can be extracted quickly. <br>
If GPU memory is insufficient, the following two approaches can be applied:<br>
1)To reduce the batch size for feature extraction, the minimum value can be set to 'batch_size=1'. The corresponding modifications should be made at: line 25 in 'iAMP-SeE_Model_1.py' and line 91 in 'iAMP-SeE_Model_2.py'.
2) Use the CPU for feature extraction: when GPU memory is limited, the model will automatically switch to the CPU to extract sequence features, though this will require more time;<br>
3) Perform feature extraction on another device, ensuring that the numpy versions on both devices are consistent; otherwise, the extracted features will be unusable.
<br>
The initially extracted ESM-2 features are named "esm_features.pkl". The naming can be adjusted according to your requirements, provided that the local pkl file and the filename input to the model remain consistent. The corresponding locations for modification are: line 25 in 'iAMP-SeE_Model_1.py' and line 91 in 'iAMP-SeE_Model_2.py'.
### 3. Model Training
The model can now proceed to run normally. Please wait patiently until the execution is complete.
