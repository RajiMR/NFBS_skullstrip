function layer = upsample3dLayer( varargin )
% transposedConv3dLayer 3D transposed convolution layer
%
%   layer = transposedConv3dLayer(filterSize, numFilters) creates a
%   transposed 3D convolution layer. This layer is used to upsample feature
%   maps. filterSize specifies the height, width and depth of the filters. 
%   It can be a scalar, in which case the filters will have the same height,
%   width and depth, or a vector [h w d] where h, w and d specify the 
%   height, width and depth of the filters. numFilters specifies the number
%   of filters, which determines the number of channels in the output 
%   feature map.
%
%   layer = transposedConv3dLayer(filterSize, numFilters, 'PARAM1', VAL1, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%       'Stride'                  - The up-sampling factor of the input.
%                                   When used with Cropping 'same', the
%                                   output size equals inputSize .* stride.
%                                   The value of Stride can be a scalar, in
%                                   which case the same value is used for
%                                   all dimensions, or it can be a vector
%                                   [u v w] where u, v and w are the vertical
%                                   horizontal and the depth stride. The
%                                   default is [1 1 1] and no upsampling is
%                                   performed.
%
%       'Cropping'                - Amount to trim the edges of the full
%                                   transposed convolution, specified as
%                                   one of the following: 
%                                   - 'same'. Cropping is set so that the 
%                                     output size equals inputSize .*
%                                     stride, where inputSize is the height
%                                     width and depth of the input.
%                                   - a scalar, in which case the same
%                                     amount of data is trimmed from the
%                                     all vertical, horizontal and depth edges.
%                                   - a vector [a b c] where a is the
%                                     amount to trim from the top and
%                                     bottom, b is the amount to trim from
%                                     the left and right and c is the
%                                     amount to trim from front and back.
%                                   - a 2x3 matrix [t l f; b r bk] where t is the
%                                     cropping applied to the top, b is the
%                                     cropping applied to the bottom, l is
%                                     the cropping applied to the left,
%                                     r is the cropping applied to the
%                                     right, f is the cropping applied to 
%                                     the front and bk to back.
%                                   The default is 0.
%
%       'NumChannels'             - The number of channels for each filter.
%                                   If a value of 'auto' is passed in, the
%                                   correct value for this parameter will
%                                   be inferred at training time. The
%                                   default is 'auto'.
%
%       'Weights'                 - Layer weights, specified as a
%                                   filterSize-by-numFilters-by-numChannels 
%                                   array or []. The default is []. 
%
%       'Bias'                    - Layer biases, specified as a
%                                   1-by-1-by-1-by-numFilters array or [].
%                                   The default is [].
%
%       'WeightLearnRateFactor'   - A number that specifies multiplier for
%                                   the learning rate of the weights. The
%                                   default is 1.
%
%       'BiasLearnRateFactor'     - A number that specifies a multiplier
%                                   for the learning rate for the biases.
%                                   The default is 1.
%
%       'WeightL2Factor'          - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   weights. The default is 1.
%
%       'BiasL2Factor'            - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   biases. The default is 0.
%
%       'WeightsInitializer'      - The function to initialize the weights,
%                                   specified as 'glorot', 'he',
%                                   'narrow-normal', 'zeros', 'ones' or a
%                                   function handle. The default is
%                                   'glorot'.
%
%       'BiasInitializer'         - The function to initialize the bias,
%                                   specified as 'narrow-normal', 'zeros',
%                                   'ones' or a function handle. The
%                                   default is 'zeros'.
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   Example:
%       % Create a transposed convolutional layer with 32 filters that have a
%       % height of 4, width of 3 and depth of 2, and that upsamples the 
%       % input by a factor of 2.
%
%       layer = transposedConv3dLayer([4 3 2], 32, 'Cropping', 'same', ...
%           'Stride', 2);
%
%   See also nnet.cnn.layer.TransposedConvolution3DLayer, convolution3dLayer, transposedConv2dLayer.
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2018 The MathWorks, Inc.

% Parse the input arguments.

args = iParseInputArguments(varargin{:});

% Create an internal representation of a convolutional layer.
internalLayer = nnet.internal.cnn.layer.TransposedConvolution3D(args);

internalLayer.Weights.L2Factor = args.WeightL2Factor;
internalLayer.Weights.LearnRateFactor = args.WeightLearnRateFactor;

internalLayer.Bias.L2Factor = args.BiasL2Factor;
internalLayer.Bias.LearnRateFactor = args.BiasLearnRateFactor;

% Pass the internal layer to external layer constructor to construct a user visible
% transposed convolutional 3D layer.
layer = nnet.cnn.layer.TransposedConvolution3DLayer(internalLayer);
layer.WeightsInitializer = args.WeightsInitializer;
layer.BiasInitializer = args.BiasInitializer;
layer.Weights = args.Weights;
layer.Bias = args.Bias;
end

function inputArguments = iParseInputArguments(varargin)
varargin = nnet.internal.cnn.layer.util.gatherParametersToCPU(varargin);
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser);
end

function p = iCreateParser()
p = inputParser;

defaultStride = 2;
defaultCropping = 0;
defaultNumChannels = 'auto';
defaultWeightLearnRateFactor = 0;
defaultBiasLearnRateFactor = 0;
defaultWeightL2Factor = 1;
defaultBiasL2Factor = 0;
defaultName = '';
defaultLearnable = [];
defaultWeightsInitializer = 'ones';
defaultBiasInitializer = 'zeros';

p.addRequired('FilterSize', @iAssertValidFilterSize);
p.addRequired('NumFilters', @iAssertValidNumFilters);
p.addParameter('Stride', defaultStride, @iAssertValidStride);
p.addParameter('Cropping', defaultCropping, @iAssertValidCropping);
p.addParameter('NumChannels', defaultNumChannels, @iAssertValidNumChannels);
p.addParameter('WeightLearnRateFactor', defaultWeightLearnRateFactor, @iAssertValidFactor);
p.addParameter('BiasLearnRateFactor', defaultBiasLearnRateFactor, @iAssertValidFactor);
p.addParameter('WeightL2Factor', defaultWeightL2Factor, @iAssertValidFactor);
p.addParameter('BiasL2Factor', defaultBiasL2Factor, @iAssertValidFactor);
p.addParameter('WeightsInitializer', defaultWeightsInitializer);
p.addParameter('BiasInitializer', defaultBiasInitializer);
p.addParameter('Name', defaultName, @iAssertValidLayerName);
p.addParameter('Weights', defaultLearnable);
p.addParameter('Bias', defaultLearnable);
end

function inputArguments = iConvertToCanonicalForm(p)
% Make sure integral values are converted to double and strings to char vectors
inputArguments = struct;
inputArguments.FilterSize = double( iMakeIntoRowVectorOfThree(p.Results.FilterSize) );
inputArguments.NumFilters = double( p.Results.NumFilters );
inputArguments.Stride = double( iMakeIntoRowVectorOfThree(p.Results.Stride) );
inputArguments.CroppingMode = iCalculateCroppingMode(p.Results.Cropping);
inputArguments.CroppingSize = double( iCalculateCroppingSize(p.Results.Cropping) );
inputArguments.NumChannels = double( iConvertToEmptyIfAuto(p.Results.NumChannels) );
inputArguments.WeightLearnRateFactor = p.Results.WeightLearnRateFactor;
inputArguments.BiasLearnRateFactor = p.Results.BiasLearnRateFactor;
inputArguments.WeightL2Factor = p.Results.WeightL2Factor;
inputArguments.BiasL2Factor = p.Results.BiasL2Factor;
inputArguments.Weights = p.Results.Weights;
inputArguments.Bias = p.Results.Bias;
inputArguments.WeightsInitializer = p.Results.WeightsInitializer;
inputArguments.BiasInitializer = p.Results.BiasInitializer;
inputArguments.Name = char(p.Results.Name);
end

function iAssertValidFilterSize(value)
validateattributes(value, {'numeric'}, ...
    {'positive', 'real', 'integer', 'nonempty'});
iAssertScalarOrRowVectorOfThree(value,'FilterSize');
end

function iAssertValidNumFilters(value)
validateattributes(value, {'numeric'}, ...
    {'scalar','integer','positive'});
end

function iAssertValidStride(value)
nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'Stride', 3);
end

function iAssertValidCropping(value)
nnet.internal.cnn.layer.paramvalidation.validatePadding(value,3,'Cropping');
end

function iAssertValidNumChannels(value)
if(ischar(value) || isstring(value))
    validatestring(value,{'auto'});
else
    validateattributes(value, {'numeric'}, ...
        {'scalar','integer','positive'});
end
end

function iAssertValidFactor(value)
nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function rowVectorOfThree = iMakeIntoRowVectorOfThree(scalarOrRowVectorOfThree)
if(iIsRowVectorOfThree(scalarOrRowVectorOfThree))
    rowVectorOfThree = scalarOrRowVectorOfThree;
else
    rowVectorOfThree = [scalarOrRowVectorOfThree scalarOrRowVectorOfThree scalarOrRowVectorOfThree];
end
end

function iAssertScalarOrRowVectorOfThree(value,name)
if ~(isscalar(value) || iIsRowVectorOfThree(value))
    exception = MException(message('nnet_cnn:layer:Layer:ParamMustBeScalarOrTriple',name));
    throwAsCaller(exception);
end
end

function tf = iIsRowVectorOfThree(x)
tf = isrow(x) && numel(x)==3;
end

function y = iConvertToEmptyIfAuto(x)
if(iIsAutoString(x))
    y = [];
else
    y = x;
end
end

function tf = iIsAutoString(x)
tf = strcmp(x, 'auto');
end

function croppingMode = iCalculateCroppingMode(value)
croppingMode = nnet.internal.cnn.layer.padding.calculatePaddingMode(value);
end

function croppingSize = iCalculateCroppingSize(value)
croppingSize = nnet.internal.cnn.layer.padding.calculatePaddingSize(value,3);
end