%% Segmentation on Test Data

% Clear workspace
clear; close all; clc;

destination = '/rsrch1/ip/rmuthusivarajan/imaging/NFBS/192densenet3d';

% Images Datapath % define reader
procVolReader = @(x) niftiread(x);
procVolLoc = fullfile(destination,'preprocess192','images');
procVolDs = imageDatastore(procVolLoc, ...
    'FileExtensions','.nii','LabelSource','foldernames','ReadFcn',procVolReader);

% Labels Datapath % define reader
procLblReader =  @(x) niftiread(x);
procLblLoc = fullfile(destination,'preprocess192','labels');
classNames = ["background","tumor"];
pixelLabelID = [0 1];
procLblDs = pixelLabelDatastore(procLblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.nii','ReadFcn',procLblReader);

%%Load test indices 
s = load('idxTest.mat');
c = struct2cell(s);
idxTest = cat(1,c{:});

%%Load patient id
P = load('PId.mat');
P1  = P.PId;
patientId = cellstr(P1);


for kfold = 5
    
    disp(['Processing K-fold-' num2str(kfold)]);
    
    voldsTest = subset(procVolDs,idxTest{1,kfold}); %ground truth test images
    pxdsTest = subset(procLblDs,idxTest{1,kfold}); %ground truth labels
    
    trainedNetName = ['fold_' num2str(kfold) '-trainedDensenet3d192.mat'];
    load(fullfile(destination, trainedNetName));
          
    t = idxTest{1,kfold};
    testPatientId = patientId(t);%create test patientid set
    
    id = 1;

    while hasdata(voldsTest)
    
    vol{id} = read(voldsTest);
    lbl{id} = readNumeric(pxdsTest);
    
    patientId = testPatientId{id};
    
    predictedLabel = semanticseg(vol{id},net,'ExecutionEnvironment','cpu');
    groundTruthLabel = lbl{id};
       
    predLblName = ['predictedLbl_', patientId];
    grdLblName = ['groundTruthLbl_',patientId];
     
    %create directories to store labels 
    mkdir(fullfile(destination,['predictedLabel-fold' num2str(kfold)]));
    mkdir(fullfile(destination,['groundTruthLabel-fold' num2str(kfold)]));

    predDir = fullfile(destination,['predictedLabel-fold' num2str(kfold)],predLblName);
    groundDir = fullfile(destination,['groundTruthLabel-fold' num2str(kfold)],grdLblName);
        
    % save preprocessed data to folders
    niftiwrite(uint8(predictedLabel),predDir);
    niftiwrite(uint8(groundTruthLabel),groundDir);
                               
    id = id + 1;
    end
end


