# Radiogenomic Classification of Glioblastoma by Multimodal 3D MRI

This project was done as part of B.Tech Capstone project CP302/CP303 at IIT Ropar (session 2022-23)

Authors: [Deepan Maitra](https://www.linkedin.com/in/deepan-maitra-71810b1b4/) (B.Tech CSE'23) and [Shivani Kumari](https://www.linkedin.com/in/shivani-kumari-577392193/) (B.Tech CSE'23)

<p align="center">
    <img width="200" src="latex_report_files/report_images/methylated.png">
</p>

## A note about the directory structure:
- [data](data) contains the MRI scan files and the classification labels. Some pre-trained models are also saved in [saved_models](data/saved_models)
- [code](code) contains all the source code files, arranged in folders. The explanations of the code files are in [code_readme](code/README.md)
- [models](models) contains .py files returning all the segmentation, MTL and classification models. The explanations are in [models_readme](models/README.md)
- [Latex](latex_report_files) contains the Latex project report directory. [report.tex](latex_report_files/report.tex) is the main build file. Building the latex report is explained in [latex_readme](latex_report_files/README.md)
- [Results](results) contains all the implemented results (tabular and pictorial format) along with [code assistance](results/README.md)


## Project Abstract
O6-Methylguanine-DNA methyltransferase (MGMT) promoter methylation
status is an important genetic characteristic of glioblastoma and
is crucial for it’s diagnosis and chemotherapy efficacy. Multimodal
MRI imaging techniques can contribute towards monitored automation
of invasive surgical approaches. In this body of work, we propose
an end-to-end pipeline of tumour segmentation and subsequent radiogenomic
classification to classify the given MRI scans into being
methylated or non-methylated. We develop a novel Multitask-learning
based model (adapted from U-Net) to simultaneously perform segmentation
and classification. Further, we utilise the segmentation
results to cascade lightweight classification models based on several
MRI slice sampling techniques to output the final classification scores.
Our resultant pipeline performs well on each of the MRI axes, and several
ensembles are tried out to arrive at suitable improvements. Based
on 5-fold-cross-validation, we are able to surpass the current Kaggle
leaderboard AUCs of the past BraTS RSNA-MICCAI Brain tumour
challenge of 2021.


## Problem source
This project uses the [BraTS 2021 Kaggle contest](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/overview/description) as the problem statement and utilises the same datasets as provided to competitors. 3D MRI files were sourced from the auxillary contest hosted in [Synapse](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571). 



