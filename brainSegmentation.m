% Segmentation on Test Data

function brainSegmentation(inputFilePath, trainedNetName, outputFilePath)

disp( ['inputFilePath  = ''',inputFilePath ,''';']);
disp( ['trainedNetName = ''',trainedNetName ,''';']);
disp( ['outputFilePath = ''',outputFilePath ,''';']);

trainedNetwork = load(trainedNetName);
          
imginfo = niftiinfo(inputFilePath);
imgvol = niftiread(imginfo);

predictedLabel = semanticseg(imgvol,trainedNetwork.net,'ExecutionEnvironment','cpu');;
        
niftiwrite(single(predictedLabel),outputFilePath,imginfo);
