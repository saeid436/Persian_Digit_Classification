%% Test the Network:

function numberOfCorrectedClassified = testNetwork(testSamples,testTargets, numberOfTestSamples, trainedNetworkWeightL1, trainedNetworkWeightL2)
	
	correctedClassified = 0;
	for i = 1 : numberOfTestSamples
		
		OSample = testSamples(:,i); % getting test samples
		OTarget = testTargets(:,i); % getting the target
		NET1 = OSample'*trainedNetworkWeightL1;
		NET1 = NET1';
		OUT1 = 1./(1+exp(-NET1));
		X2 = [1;OUT1];
		NET2 = X2'*trainedNetworkWeightL2;
		NET2 = NET2';
		OUT2 = 1./(1+exp(-NET2)); % Output of the trained network
		[V, Index] = max(OUT2); % getting the index of max value of output vector
		
		if(OTarget(Index) == 1)
			correctedClassified = correctedClassified + 1;
		end
	end
	numberOfCorrectedClassified = correctedClassified;
end