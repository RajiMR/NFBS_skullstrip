%%Calculate dice similarity coefficient

% Clear workspace
clear; close all; clc;

destination = '/rsrch1/ip/rmuthusivarajan/imaging/NFBS/192densenet3d';

% grdLbl Datapath % define reader
grdLblReader = @(x) niftiread(x);
grdLblLoc = fullfile(destination,'groundTruthLabel-fold5');
classNames = ["background","tumor"];
pixelLabelID = [0 1];
grdLblDs = pixelLabelDatastore(grdLblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.nii','ReadFcn',grdLblReader);

% predLbl Datapath % define reader
predLblReader =  @(x) niftiread(x);
predLblLoc = fullfile(destination,'predictedLabel-fold5');
classNames = ["background","tumor"];
pixelLabelID = [0 1];
predLblDs = pixelLabelDatastore(predLblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.nii','ReadFcn',predLblReader);

diceResult = zeros(length(grdLblDs.Files),2);

for j = 1:length(grdLblDs.Files)
    temppredictedLabel = read(predLblDs);
    predictedLabel{j} = temppredictedLabel{1};
    
    tempgroundTruthLabel = read(grdLblDs);
    groundTruthLabel{j} = tempgroundTruthLabel{1};
    
    diceResult(j,:) = dice(groundTruthLabel{j},predictedLabel{j});
end

%Calculate the average Dice score across the set of test volumes.
meanDiceBackground = mean(diceResult(:,1));
disp(['Average Dice score of background across ',num2str(j), ...
    ' test volumes = ',num2str(meanDiceBackground)])

meanDiceBrain = mean(diceResult(:,2));
disp(['Average Dice score of tumor across ',num2str(j), ...
    ' test volumes = ',num2str(meanDiceBrain)])

createBarplot = true;
if createBarplot
    figure
    bar(diceResult(:,2))
    title('Test Set Dice Accuracy')
    xlabel('Brain')
    ylabel('Dice Coefficient')
end