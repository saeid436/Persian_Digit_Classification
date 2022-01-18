%% Block Meant Feature Extraction Function:

function [FeatureVec] = BlockMean(Image,lengthOfWindow,withOfWindow,WinSize)
	ImageResize = imresize(Image,[lengthOfWindow withOfWindow]); % Resize Input Image TO Desired Scale
	FeatureVec = zeros(((lengthOfWindow/WinSize)*(withOfWindow/WinSize))+1,1); % A Vector For Save Features
	FeatureVec(1) = 1; % Bias
	K = 2;
	for i = 1 : WinSize : lengthOfWindow
		for j = 1 : WinSize : withOfWindow
			BlockAvarage = sum(sum(ImageResize(i : i+WinSize-1,j : j+WinSize-1)))/(WinSize^2); % Compute Values Of Means For Every Block
			FeatureVec(K) = BlockAvarage;
			K = K+1;
		end
	end
end