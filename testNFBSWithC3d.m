%% Segmentation on Test Data

% Clear workspace
clear; close all; clc;

destination = '/rsrch1/ip/rmuthusivarajan/imaging/NFBS/192withc3d';

% Images Datapath % define reader
procVolReader = @(x) niftiread(x);
procVolLoc = fullfile(destination,'preprocess','imgResized');
procVolDs = imageDatastore(procVolLoc, ...
    'FileExtensions','.gz','LabelSource','foldernames','ReadFcn',procVolReader);

% Labels Datapath % define reader
procLblReader =  @(x) uint8(niftiread(x));
procLblLoc = fullfile(destination,'preprocess','lblResized');
classNames = ["background","tumor"];
pixelLabelID = [0 1];
procLblDs = pixelLabelDatastore(procLblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.gz','ReadFcn',procLblReader);

%%Load test indices 
s = load('idxTest.mat');
c = struct2cell(s);
idxTest = cat(1,c{:});

%%Load patient id
P = load('PId.mat');
PId  = P.idLoc;

C = cell(25,5);
testPatientId = deal(C);

for kfold = 1:5
    
    disp(['Processing K-fold-' num2str(kfold)]);
    
    voldsTest = subset(procVolDs,idxTest{1,kfold}); %ground truth test images
    pxdsTest = subset(procLblDs,idxTest{1,kfold}); %ground truth labels
    
    trainedNetName = ['fold_' num2str(kfold) '-trainedDensenet3d.mat'];
    load(fullfile(destination, trainedNetName));
          
    testSet = idxTest{1,kfold};
    testPatientId(:,kfold) =  PId(testSet);%create test patientid set
    save('testPatientId.mat','testPatientId');
   
    %create directories to store labels 
        mkdir(fullfile(destination,['predictedLabel-fold' num2str(kfold)]));
        mkdir(fullfile(destination,['groundTruthLabel-fold' num2str(kfold)]));
    
    id = 1;

    while hasdata(voldsTest)
        
        C = cell(1,25);
        [vol,lbl] = deal(C);
    
        vol{id} = read(voldsTest);
        lbl{id} = readNumeric(pxdsTest);
    
        patientId = testPatientId{id};
    
        predictedLabel = semanticseg(vol{id},net,'ExecutionEnvironment','cpu');
        groundTruthLabel = lbl{id};
       
        predLblName = ['predictedLbl_', patientId];
        grdLblName = ['groundTruthLbl_',patientId];
     
        predDir = fullfile(destination,['predictedLabel-fold' num2str(kfold)],predLblName);
        groundDir = fullfile(destination,['groundTruthLabel-fold' num2str(kfold)],grdLblName);
        
        % save preprocessed data to folders
        niftiwrite(uint8(predictedLabel),predDir);
        niftiwrite(uint8(groundTruthLabel),groundDir);
                               
        id = id + 1;
    end
end