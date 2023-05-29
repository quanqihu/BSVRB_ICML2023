### Introduction

This folder contains the code and scripts to reproduce the Hyperparameter Optimization experimental results in the manuscript. We implement our proposed algorithms BSVRBv1 and BSVRBv2, and the baseline algorithm RSVRB from existing work.


### Datasets

We use binary classification datasets a8a and w8a from LIBSVM Data. The datasets are contained in the /datasets folder. Both datasets are accessible at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.

### Model training and evaluation
We implement three methods including BSVRBv1, BSVRBv2 and RSVRB in main.py. To train the model, you may run the provided example bash files `run_BSVRBv1.sh`, `run_BSVRBv2.sh`, or `run_RSVRB.sh`.
