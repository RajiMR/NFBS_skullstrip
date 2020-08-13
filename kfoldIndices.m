% Clear workspace
clear; close all; clc;

% Images Datapath % define reader
volLoc = '/rsrch1/ip/rmuthusivarajan/imaging/NFBS/nfbsOutput/nfbsPreprocessedDataset/imagesMain';
volReader = @(x) matRead(x);
imds = imageDatastore(volLoc, ...
    'FileExtensions','.mat','LabelSource','foldernames','ReadFcn',volReader);

lblLoc = '/rsrch1/ip/rmuthusivarajan/imaging/NFBS/nfbsOutput/nfbsPreprocessedDataset/labelsMain';
classNames = ["background","tumor"];
pixelLabelID = [0 1];
pxds = pixelLabelDatastore(lblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',volReader);

num_images = length(imds.Labels); %number of obervations for kfold

%kfold partition
c1 = cvpartition(num_images,'kfold',5);
err = zeros(c1.NumTestSets,1);

for i = 1:c1.NumTestSets

testIdx{i} = test(c1,i); %logical indices for first fold test set
imdstest{i} = subset(imds,testIdx{i}); %first fold test imagedatastore
pxdsTest{i} = subset(pxds,testIdx{i}); % first fold test pixelimagedatastore

holdIdx{i} = training(c1,i); %logical indices for first fold training set-holdout partition
imdshold{i} = subset(imds,holdIdx{i}); %imds for holdout partition
pxdshold{i} = subset(pxds,holdIdx{i}); %imds for holdout partition

num_imdshold{i} = length(imdshold{i}.Labels); %number of obervations for holdout partition
c2 = cvpartition(num_imdshold{i},'holdout',0.25);

valIdx{i} = test(c2); %logical indices for first fold val. set
imdsVal{i} = subset(imdshold{i},valIdx{i}); % first fold val. imagedatastore
pxdsVal{i} = subset(pxdshold{i},valIdx{i}); % first fold val. pixelimagedatastore

trainIdx{i} = training(c2); %logical indices for first fold training set
imdsTrain{i} = subset(imdshold{i},trainIdx{i}); % first fold training imagedatastore
pxdsTrain{i} = subset(pxdshold{i},trainIdx{i}); % first fold training pixelimagedatastore

end
