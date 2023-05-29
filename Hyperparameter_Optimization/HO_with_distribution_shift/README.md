### Introduction
This folder contains the code and scripts to reproduce the experimental results of the Hyperparameter Optimization with imbalanced data and noisy labels in the manuscript. We implement three methods, i) logistic regression with the standard logistic loss as the baseline, ii) BSVRBv1 for solving the bilevel formulation of HO with only one lower level problem ($m=1$ and using the standard logistic loss), and iii) BSVRBv1 for solving multi-block bilevel formulation of HO with $m=100$ blocks corresponding to 100 settings of the scaling factor $\tau_i$ in the logistic loss.

### Dataset Prepration
We use binary classification dataset a8a from LIBSVM Data. The dataset files are contained in the /datasets folder. The dataset is accessible at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/. To construct data distribution shift, run `python datasets_generator.py`.

### Model training and evaluation
Run example bash files `run_BSVRB.sh` or `run_log_reg.sh` to train the model using BSVRB or logistic regression respectively.
