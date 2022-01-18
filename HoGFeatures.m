%% Histogram Of Gradient(HOG) feature extraction function:

function HoG = HoGFeatures(Image,lengthOfWindow,withOfWindow)
	Image = imresize(Image,[lengthOfWindow,withOfWindow]);
	D1 = edge(Image,'sobel',vertical); % 
	D2 = edge(Image,'sobel',horizontal);
	Gx = conv2(Image,D2);
	Gy = conv2(Image,D1);
	AmpG = sqrt(Gx.^2 + Gy.^2);
	Theta = arctan(Gy./Gx);
	
	if(Theta >=0 && Theta < 180)
		Theta = Theta;
	else
		Theta = Theta - 180;
	end
	Histogram = zeros(1,181);
	
	for i = 1:lengthOfWindow
		for j = 1:withOfWindow
			Angle = Theta(i,j);
			A = AmpG(i,j);
			Histogram(Angle+1) = Histogram(Angle+1) + A;
		end
	end
	
	MaxHist = max(Histogram);
	MinHist = min(Histogram);
	HoG = Histogram - MinHist/(MaxHist-MinHist);
end