# NFBS_skullstrip


## Run Preprocess-Resize to 192 192 192 - run Kfold - Train with Densenet3d
matlab trainingNFBS192.m 

Input datapath: inputNFBS192.json

## Create Test data with kfold indices, Run Segmentation on Test Data 
matlab testNFBS192.m

## Calculate DSC 
matlab dsc.m

