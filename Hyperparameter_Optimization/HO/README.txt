Introduction

The project contains the code and scripts to reproduce the Hyperparameter Optimization experimental results in the manuscript. We implement our proposed algorithms BSVRBv1 and BSVRBv2, and the baseline algorithm RSVRB from existing work.


Instructions

1. Datasets. We use binary classification datasets a8a and w8a from LIBSVM Data. The datasets are contained in the /datasets folder. Both datasets are accessible at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.

2. Methods. BSVRBv1, BSVRBv2 and RSVRB are implemented in main.py. 

3. Training. To train the model, you may run the provided example bash files run_BSVRBv1.sh, run_BSVRBv2.sh, or run_RSVRB.sh.