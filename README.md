# NFBS_skullstrip


## Run Preprocess-Resize to 192 192 192 - run Kfold - Train with Densenet3d
matlab trainingNFBSWithC3d.m 

Input datapath: inputNFBS192.json

## Helper functions for run training: 
matRead.m
upsample3dLayer.m
dicePixelClassification3dLayer.m

## Create Test data with kfold indices & Run Segmentation on Test Data 
matlab testNFBSWithC3d.m

## Calculate DSC 

