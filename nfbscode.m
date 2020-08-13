%% 3-D Brain Extraction from MRI
%    Train and cross validate a 3-D U-net for brain extraction on T1 image
%          load nifti data from /rsrch1/ip/egates1/NFBS\ Skull\ Strip/NFBSFilepaths.csv 
% Setting up the code: fresh start
clc
clear all
close all

fname = 'brainTest.json';
jsonText = fileread(fname);
jsonData = jsondecode(jsonText);

% Read file pathways into table
fullFileName = jsonData.fullFileName;
    % enter: = /rsrch1/ip/egates1/NFBS Skull Strip/NFBSFilepaths.csv

delimiter = jsonData.delimiter;
    % enter: ,

T = readtable(fullFileName, 'Delimiter', delimiter);
A = table2array(T);
volCol = jsonData.volCol;
    % enter: 4
    
lblCol = jsonData.lblCol;
    % enter: 5

volLoc = A(:,volCol);
lblLoc = A(:,lblCol);

stoFoldername = jsonData.stoFoldername;
% for user-defined: destination = input("Please enter the file pathway for folder to store training, validation, and test sets: ", 's')
destination = fullfile(tempdir,stoFoldername, 'preprocessedDataset');

 % define readers
 maskReader = @(x) (niftiread(x)>0);
 volReader = @(x) niftiread(x);
 
 %read data into datastores
 volds = imageDatastore(volLoc, ...
     'FileExtensions','.gz','ReadFcn',volReader);
 
 classNames = ["background","brain"];
  pixelLabelID = [0 1];
 
 % read data intp pixelLabeldatastore
 pxds = pixelLabelDatastore(lblLoc,classNames, pixelLabelID, ...
        'FileExtensions','.gz','ReadFcn',maskReader);
  reset(volds);
  reset(pxds);      
 

   % create directories to store data sets
        mkdir(fullfile(destination,'imagesMain'));
        mkdir(fullfile(destination,'labelsMain'));
        
        imDir = fullfile(destination, 'imagesMain', stoFoldername);
        labelDir = fullfile(destination, 'labelsMain', stoFoldername);
    

       
   %% Crop relevant region
    NumFiles = length(pxds.Files);
    id = 1;
    


    
    while hasdata(pxds)
        outL = readNumeric(pxds);
        outV = read(volds);
        temp = outL>0;
        sz = size(outL);
        reg = regionprops3(temp,'BoundingBox');
        tol = 64;
        ROI = ceil(reg.BoundingBox(1,:));
        ROIst = ROI(1:3) - tol;
        ROIend = ROI(1:3) + ROI(4:6) + tol;

        ROIst(ROIst<1)=1;
        ROIend(ROIend>sz)=sz(ROIend>sz);

        tumorRows = ROIst(2):ROIend(2);
        tumorCols = ROIst(1):ROIend(1);
        tumorPlanes = ROIst(3):ROIend(3);

        tcropVol = outV(tumorRows,tumorCols, tumorPlanes);
        tcropLabel = outL(tumorRows,tumorCols, tumorPlanes);


% Data set with a valid size for 3-D U-Net (multiple of 8)
        ind = floor(size(tcropVol)/8)*8;
        incropVol = tcropVol(1:ind(1),1:ind(2),1:ind(3));
        mask = incropVol == 0;

%%%%%%%% channelWisePreProcess
        % As input has 4 channels (modalities), remove the mean and divide by the
        % standard deviation of each modality independently.
        incropVol1=single(incropVol);

        chn_Mean = mean(incropVol1,[1 2 3]);
        chn_Std = std(incropVol1,0,[1 2 3]);
        cropVol1 = (incropVol1 - chn_Mean)./chn_Std;

        rangeMin = -5;
        rangeMax = 5;
        % Remove outliers
        cropVol1(cropVol1 > rangeMax) = rangeMax;
        cropVol1(cropVol1 < rangeMin) = rangeMin;

        % Rescale the data to the range [0, 1]
        cropVol1 = (cropVol1 - rangeMin) / (rangeMax - rangeMin);

%%%%%%%%        
        % Set the nonbrain region to 0
        cropVol1(mask) = 0;
        cropLabel = tcropLabel(1:ind(1),1:ind(2),1:ind(3));

        
        % save preprocessed data to folders
         save([imDir num2str(id,'%.3d') '.mat'],'cropVol1');
         save([labelDir num2str(id,'%.3d') '.mat'],'cropLabel');
         
         %outDim{id} = size(cropVol1);
        
         id=id+1;

   end  
    


%% create datastores for processed labels and images

% procvolds stores processed T1 volumetric data
procvolReader = @(x) matRead(x);
procvolLoc = fullfile(destination,'imagesMain');
procvolds = imageDatastore(procvolLoc, ...
    'FileExtensions','.mat','ReadFcn',procvolReader);

labelDirProc = fullfile(destination, 'labelsMain');

% proclblfs stores processed mask file info
proclblfs = matlab.io.datastore.DsFileSet(labelDirProc);

%%%%%%%load 3d unet %%%%%%%%%%%%
% before starting, need to define "n" which is the number of channels.
n = 1;
lgraph = layerGraph();

tempLayers = [
    image3dInputLayer([64 64 64 n],"Name","input","Normalization","none")
    convolution3dLayer([3 3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module1_Level1")
    reluLayer("Name","relu_Module1_Level1")
    convolution3dLayer([3 3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level2")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2 2])
    convolution3dLayer([3 3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module2_Level1")
    reluLayer("Name","relu_Module2_Level1")
    convolution3dLayer([3 3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level2")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2 2])
    convolution3dLayer([3 3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module3_Level1")
    reluLayer("Name","relu_Module3_Level1")
    convolution3dLayer([3 3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level2")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2 2])
    convolution3dLayer([3 3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level1")
    convolution3dLayer([3 3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level2")
    transposedConv3dLayer([2 2 2],512,"Name","transConv_Module4","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    concatenationLayer(4,2,"Name","concat3")
    convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level1")
    convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level2")
    transposedConv3dLayer([2 2 2],256,"Name","transConv_Module5","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    concatenationLayer(4,2,"Name","concat2")
    convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level1")
    convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level2")
    transposedConv3dLayer([2 2 2],128,"Name","transConv_Module6","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    concatenationLayer(4,2,"Name","concat1")
    convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level1","Padding","same")
    reluLayer("Name","relu_Module7_Level1")
    convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level2","Padding","same")
    reluLayer("Name","relu_Module7_Level2")
    convolution3dLayer([1 1 1],2,"Name","ConvLast_Module7")
    softmaxLayer("Name","softmax")
    dicePixelClassification3dLayer("output")];

%     helperDicePixelClassification3dLayer("output",1e-08,categorical(["background";"tumor"]));

lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
lgraph = connectLayers(lgraph,"relu_Module1_Level2","maxpool_Module1");
lgraph = connectLayers(lgraph,"relu_Module1_Level2","concat1/in1");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","maxpool_Module2");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","concat2/in1");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","maxpool_Module3");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","concat3/in1");
lgraph = connectLayers(lgraph,"transConv_Module4","concat3/in2");
lgraph = connectLayers(lgraph,"transConv_Module5","concat2/in2");
lgraph = connectLayers(lgraph,"transConv_Module6","concat1/in2");

plot(lgraph);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do the k-fold partition

patients = A(:,1);% Extract the patient ids in the filepaths table
partition = cvpartition(patients,'k',5);
err = zeros(partition.NumTestSets,1);


for i = 1:partition.NumTestSets
    trIdx = partition.training(i);
    teIdx = partition.test(i);
    trData = subset(procvolds, trIdx);
    trMask = subset(proclblfs, trIdx);
    
    tvSplit = cvpartition(numpartitions(trData),'HoldOut',0.125);

    % Training, validation, and test data for each fold
    trainData = subset(trData, tvSplit.training);
    trainMask = subset(trMask, tvSplit.training);
    valData = subset(trData, tvSplit.test);
    valMask = subset(trMask, tvSplit.test);
    testData = subset(procvolds, teIdx);
    testMask = subset(proclblds, teIdx);
    
    % write file pathways of mask sets from dsfileset to table
    trmaskinfo = resolve(trainMask);
    valmaskinfo = resolve(valMask);
    testmaskinfo = resolve(testMask);
    
    % convert tables to arrays
    trmaskfullArr = table2array(trmaskinfo);
    valmaskfullArr = table2array(valmaskinfo);
    testmaskfullArr = table2array(testmaskinfo);
    
    % read file pathways into string arrays
    trmaskArr = trmaskfullArr(:,1);
    valmaskArr = valmaskfullArr(:,1);
    testmaskArr = testmaskfullArr(:,1);
    
    % convert string to char arrays
    trmaskChar = convertStringsToChars(trmaskArr);
    valmaskChar = convertStringsToChars(valmaskArr);
    testmaskChar = convertStringsToChars(testmaskArr);
    
    % read these into pixellabeldatastores
    proclblReader = @(x) matRead(x);
    classNames = ["background","tumor"];
    pixelLabelID = [0 1];
    trmaskpxds = pixelLabelDatastore(trmaskChar,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',procvolReader);

    valmaskpxds = pixelLabelDatastore(valmaskChar,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',procvolReader);

    tsmaskpxds = pixelLabelDatastore(testmaskChar,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',procvolReader);


    % Compute Dice(general concept, might be a more code-friendly way to do it)
    %{
    p = networkPrediction.*correctPrediction
    s = 2*sum(p, 'all')
    err(i) = s/(sum(networkPrediction,'all')+sum(correctPrediction, 'all'))
    %}
    
    % Need Random Patch Extraction on testing and validation Data
    patchSize = [64 64 64];
    patchPerImage = 16;
    miniBatchSize = 8;
  %training patch datastore
  trpatchds = randomPatchExtractionDatastore(trainData,trmaskpxds,patchSize, ...
    'PatchesPerImage',patchPerImage);
  trpatchds.MiniBatchSize = miniBatchSize;
  %validation patch datastore
  dsVal = randomPatchExtractionDatastore(valData,valmaskpxds,patchSize, ...
    'PatchesPerImage',patchPerImage);
  dsVal.MiniBatchSize = miniBatchSize;

%% do the training %%
options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'InitialLearnRate',5e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.95, ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',400, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'MiniBatchSize',miniBatchSize);


    modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
    [net,info] = trainNetwork(trpatchds,lgraph,options);
    save(['trained3DUNet-' modelDateTime '-Epoch-' num2str(options.MaxEpochs) '.mat'],'net');
    infotable = struct2table(info);
    writetable(infotable, ['2DUNetinfo-' modelDateTime '-Epoch-' num2str(options.MaxEpochs) '.txt']);

end

% Average Loss Function Error for all folds
%cvErr = sum(err)/sum(partition.TestSize);

%% evaluate the average dice similarity
%% train the model on the training set for each fold in the k-fold

%% evaluate the average dice similarity
%   manu - output nifti files of the test set predictions
%   priya - visualize differences in itksnap

%% compare to MONSTR
%   @egates1 - do you have MONSTR predictions on this dataset ? are there any papers that have already done this ?
