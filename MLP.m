%% Persian Digit Classification With MLP


% Read Images from DataSet and Compute Means Of Block Images
numberOfImages = 1699; % Number Of Images*
numberOfClasses = 10; % Digits 0~9* -> Number of Classes

% Block Mean Feature Parameters
withOfWindow = 128;   % With Of Window
lengthOfWindow = 128; % Length Of Window
WinSize = 8; % Length Of Squre For Avaraging

% Extracted Feature
Samples = zeros(((lengthOfWindow/WinSize)*(withOfWindow/WinSize))+1,numberOfImages);
Targets = zeros(numberOfClasses,numberOfImages);

%% Feature Extraction:
for a = 0 : numberOfImages-1
    for b = 0 : numberOfClasses-1
		%% Images Path:
        Adress = ['Digits\', num2str(b), '_', num2str(a), '.bmp'];
        if(exist(Adress,'file')) ~= 0
            I = imread(Adress); % Load image  
            FeatureVec = BlockMean(I,lengthOfWindow,withOfWindow,WinSize); % Get BlokMean features
            Samples(:,a+1) = FeatureVec; % Make Matrix of features whre Evry Column represent a Feature vector* 
            Targets(b+1,a+1) = 1; % Make Target Matrix in which evry column has One True Value*
        end
    end
end

numberOfTrainingSamples = round(.7*numberOfImages); % Number of Training Samples
numberOfTestSamples = numberOfImages - numberOfTrainingSamples;

trainingSamples = Samples(:, 1:numberOfTrainingSamples); % Training samples
trainingTargets = Targets(:, 1:numberOfTrainingSamples); % Training targets
testSamples = Samples(:, numberOfTrainingSamples:numberOfImages); % Test samples
testTargets = Targets(:, numberOfTrainingSamples:numberOfImages); % Test targets

%% Training Section
% Initializing Weights
W1 = (rand(((lengthOfWindow/WinSize)*(withOfWindow/WinSize))+1,NH)-.5)*.4; % Initializing Weights between Inputs And Hidden Layer
W2 = (rand(NH+1,numberOfClasses)-.5)*.4; % Initializing Weights between Hidden Layer And OutPut Layer
	
	
Delta2N1 = zeros(NH+1,numberOfClasses); % For Save Previuse Delta (Generalize Delta Rule)
Delta1N1 = zeros(((lengthOfWindow/WinSize)*(withOfWindow/WinSize))+1,NH); % For Save Previuse Delta (Generalize Delta Rule)
% Initializing parameters:
learningRate = 0.015; % Learning Rate 
NH = 80; % Number Of Neuron For Hidden Layer
Epoch = 40; % Number of Epochs

%% Train Network:
[trainedNetworkWeigthL1, trainedNetworkWeightL2, Err] = trainNetwork(trainingSamples, trainingTargets, W1, W2, Delta1N1, Delta2N1, numberOfTrainingSamples, numberOfImages, learningRate, Epoch);

%% Test Network:
numberOfCorrectedClassified = testNetwork(testSamples,testTargets, numberOfTestSamples, trainedNetworkWeigthL1, trainedNetworkWeightL2);

%% Accuration and Error:
Accuration = (numberOfCorrectedClassified/(numberOfImages-numberOfTrainingSamples))*100
plot(Err);title('rms ERROR')