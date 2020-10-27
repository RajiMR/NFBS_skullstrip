% Segmentation on Test Data

% Clear workspace
clear; close all; clc;

destination = '/rsrch1/ip/rmuthusivarajan/imaging/NFBS/192withc3d';

imgResizedDir = dir(fullfile(destination, 'preprocess', 'imgResized','*.gz'));
imgFile = {imgResizedDir.name}';
imgFolder = {imgResizedDir.folder}';

lblResizedDir = dir(fullfile(destination, 'preprocess', 'lblResized','*.gz'));
lblFile = {lblResizedDir.name}';
lblFolder = {lblResizedDir.folder}';

%%Load test indices 
s = load('idxTest.mat');
c = struct2cell(s);
idxTest = cat(1,c{:});

%%Load patient id
P = load('PId.mat');
PId  = P.idLoc;
%patientId = char(idLoc(id,:));

C = cell(25,5);
[testPatientId, imgFileTest, imgFolderTest, lblFileTest, lblFolderTest] = deal(C);

for kfold = 1:5
    
    disp(['Processing K-fold-' num2str(kfold)]);
    
    trainedNetName = ['fold_' num2str(kfold) '-trainedDensenet3d.mat'];
    load(fullfile(destination, trainedNetName));
          
    testSet = idxTest{1,kfold};
    testPatientId(:,kfold) =  PId(testSet);%create test patientid set
    save('testPatientId.mat','testPatientId');
    
    imgFileTest(:,kfold) = imgFile(testSet);
    imgFolderTest(:,kfold) = imgFolder(testSet);
    
    lblFileTest(:,kfold) = lblFile(testSet);
    lblFolderTest(:,kfold) = lblFolder(testSet);
   
    %create directories to store labels 
        mkdir(fullfile(destination,['predictedLabel-fold' num2str(kfold)]));
        mkdir(fullfile(destination,['groundTruthLabel-fold' num2str(kfold)]));
        
    for id = 1:length(imgFileTest)
        
        imgLoc = fullfile(imgFolderTest(id,kfold),imgFileTest(id,kfold));
        imgName = niftiread(char(imgLoc));
        imginfo = niftiinfo(char(imgLoc));
               
        lblLoc = fullfile(lblFolderTest(id,kfold),lblFileTest(id,kfold));
        lblName = niftiread(char(lblLoc));
        lblinfo = niftiinfo(char(lblLoc));
            
        patientId = char(testPatientId(id,kfold));
               
        predLblName = ['predictedLbl_', patientId];
        grdLblName = ['groundTruthLbl_',patientId];
        
        predDir = fullfile(destination,['predictedLabel-fold' num2str(kfold)],predLblName);
        groundDir = fullfile(destination,['groundTruthLabel-fold' num2str(kfold)],grdLblName);
        
        groundTruthLabel = lblName;
        predictedLabel = semanticseg(imgName,net,'ExecutionEnvironment','cpu');
        
        % save preprocessed data to folders
        niftiwrite(single(predictedLabel),predDir,imginfo);
        niftiwrite(groundTruthLabel,groundDir,lblinfo);
                               
        id = id + 1;
    end
end