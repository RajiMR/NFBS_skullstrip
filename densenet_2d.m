%% Create Layer Graph
% Create the layer graph variable to contain the network layers.
%define n as number of channels
n = 1;
lgraph = layerGraph();
%% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.

tempLayers = [
    imageInputLayer([64 64 n],"Name","input","Normalization","none")
    batchNormalizationLayer("Name","BN_Module1_Level1")
    convolution2dLayer([3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module1_Level2")
    convolution2dLayer([3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","depthcat_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","BN_Module2_Level1")
    convolution2dLayer([3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module2_Level2")
    convolution2dLayer([3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","depthcat_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","BN_Module3_Level1")
    convolution2dLayer([3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module3_Level2")
    convolution2dLayer([3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","depthcat_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","BN_Module4_Level1")
    convolution2dLayer([3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module4_Level2")
    convolution2dLayer([3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_5")
    upsample2dLayer([2 2],512,"Name","upsample_Module4","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_4")
    batchNormalizationLayer("Name","BN_Module5_Level1")
    convolution2dLayer([3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module5_Level2")
    convolution2dLayer([3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_7")
    upsample2dLayer([2 2],256,"Name","upsample_Module5","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_6")
    batchNormalizationLayer("Name","BN_Module6_Level1")
    convolution2dLayer([3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module6_Level2")
    convolution2dLayer([3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_8")
    upsample2dLayer([2 2],128,"Name","upsample_Module6","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_9")
    batchNormalizationLayer("Name","BN_Module7_Level1")
    convolution2dLayer([3 3],64,"Name","conv_Module7_Level1","Padding","same")
    reluLayer("Name","relu_Module7_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module7_Level2")
    convolution2dLayer([3 3],64,"Name","conv_Module7_Level2","Padding","same")
    reluLayer("Name","relu_Module7_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_10")
    batchNormalizationLayer("Name","BN_Module7_Level3")
    convolution2dLayer([1 1],2,"Name","ConvLast_Module7")
    softmaxLayer("Name","softmax")
    dicePixelClassificationLayer("Name","output")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

lgraph = connectLayers(lgraph,"relu_Module1_Level1","BN_Module1_Level2");
lgraph = connectLayers(lgraph,"relu_Module1_Level1","depthcat_1/in1");
lgraph = connectLayers(lgraph,"relu_Module1_Level2","depthcat_1/in2");
lgraph = connectLayers(lgraph,"depthcat_1","maxpool_Module1");
lgraph = connectLayers(lgraph,"depthcat_1","depthcat_9/in1");
lgraph = connectLayers(lgraph,"relu_Module2_Level1","BN_Module2_Level2");
lgraph = connectLayers(lgraph,"relu_Module2_Level1","depthcat_2/in1");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","depthcat_2/in2");
lgraph = connectLayers(lgraph,"depthcat_2","maxpool_Module2");
lgraph = connectLayers(lgraph,"depthcat_2","depthcat_6/in2");
lgraph = connectLayers(lgraph,"relu_Module3_Level1","BN_Module3_Level2");
lgraph = connectLayers(lgraph,"relu_Module3_Level1","depthcat_3/in1");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","depthcat_3/in2");
lgraph = connectLayers(lgraph,"depthcat_3","maxpool_Module3");
lgraph = connectLayers(lgraph,"depthcat_3","depthcat_4/in1");
lgraph = connectLayers(lgraph,"relu_Module4_Level1","BN_Module4_Level2");
lgraph = connectLayers(lgraph,"relu_Module4_Level1","depthcat_5/in2");
lgraph = connectLayers(lgraph,"relu_Module4_Level2","depthcat_5/in1");
lgraph = connectLayers(lgraph,"upsample_Module4","depthcat_4/in2");
lgraph = connectLayers(lgraph,"relu_Module5_Level1","BN_Module5_Level2");
lgraph = connectLayers(lgraph,"relu_Module5_Level1","depthcat_7/in1");
lgraph = connectLayers(lgraph,"relu_Module5_Level2","depthcat_7/in2");
lgraph = connectLayers(lgraph,"upsample_Module5","depthcat_6/in1");
lgraph = connectLayers(lgraph,"relu_Module6_Level1","BN_Module6_Level2");
lgraph = connectLayers(lgraph,"relu_Module6_Level1","depthcat_8/in1");
lgraph = connectLayers(lgraph,"relu_Module6_Level2","depthcat_8/in2");
lgraph = connectLayers(lgraph,"upsample_Module6","depthcat_9/in2");
lgraph = connectLayers(lgraph,"relu_Module7_Level1","BN_Module7_Level2");
lgraph = connectLayers(lgraph,"relu_Module7_Level1","depthcat_10/in1");
lgraph = connectLayers(lgraph,"relu_Module7_Level2","depthcat_10/in2");
%% Plot Layers

plot(lgraph);
