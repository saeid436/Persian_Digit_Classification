%% Training the Network:

function [trainedNetworkWeigthL1, trainedNetworkWeightL2, Err] = trainNetwork(trainingSamples, trainingTargets, W1, W2, Delta1N1, Delta2N1, numberOfTrainingSamples, numberOfImages, learningRate, Epoch)

	
	U = .9; % For Generalize Delta Rule
	ERROR = zeros(Epoch,1);
	EW = zeros(numberOfImages,1);

	for epoch = 1:Epoch
		for r = 1:numberOfTrainingSamples
        
			U = U/epoch; % Change U In Term (1/K)
			iSample = trainingSamples(:,r); % getting every feature
			iTarget = trainingTargets(:,r); % getting target 
			NET1 = iSample'*W1; % NET Of Hidden Layer
			NET1 = NET1';
			Out1 = 1./(1+exp(-NET1)); % Output Of Hidden Layer
			X2 = [1;Out1]; % Input Of OutPut Layer
			Net2 = X2'*W2; % NET Of OutPut Layer
			Out2 = 1./(1+exp(-Net2)); % Output Of OutPut Layer
			Z = Out2*(1-Out2')*(iTarget - Out2'); % DeltaK (For Updating Wkj)
			Delta12 = X2 * Z';
			Delta2 = learningRate.*Delta12 + U.*Delta2N1; % DeltaWkj
			Delta2N1 = Delta2; % Save Delta For Next Updating
			W2 = W2 + Delta2; % Wkj Correction 
			W21 = W2(2:end,1:end); % Weights Between Hidden Layer & OutPut Layer
			DeltaZ = W21 * Z;
			Z2  = iSample*Out1'*(1-Out1)*DeltaZ'; 
			Delta1 = learningRate.*Z2 + U*Delta1N1; % DeltaWji
			Delta1N1 = Delta1; % Save Delta For Next Updating
			W1 = W1 + Delta1; % Wji Correction
			EW(r) = ((sum((iTarget' - Out2).^2))/2);
		end
		
		ERROR(epoch) = sum(EW)/numberOfImages;
	end
	trainedNetworkWeigthL1 = W1;
	trainedNetworkWeightL2 = W2;
	Err = ERROR;
 end