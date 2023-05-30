## Data
The data for reproducing the results is available at https://drive.google.com/drive/folders/1hZ1T24hFCUmkHRLTdR4GxX9BoD3LFNBm. 

## Organization
After downloading the data, please organize the folder structure as follows:  
    ---|---data  (put the downloaded data here).   
    &nbsp;&nbsp;&nbsp;&nbsp; |---src   (put the code and scripts in this repository here).    
    &nbsp;&nbsp;&nbsp;&nbsp; |---log   (training logs will be store here).   
    &nbsp;&nbsp;&nbsp;&nbsp; |---model (checkpoint models will be here).   
       
## Warm-up
You can run **pretrain_ml.sh** and **pretrian_ml.sh** to get warm-up model checkpoints

## Run algorithms
To run BSRVB algorithm, you can run **run_ml_bsvrb.sh** and **run_nf_bsvrb.sh**.   
   The code framework will save the training log and checkpoint models automatically.  

   Other algorithm baselines: RankNet, ListNet, ListMLE, NeuralNDCG, ApproxNDCG, LambdaRank, SONG/K-SONG. 

## Test
To test the pre-trained models, please run **ml_test_all.sh** and **nf_test_all.sh**.
