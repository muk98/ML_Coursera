function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
       % mean(double(predictions ~= yval))
%
values1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% values2 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];


com = zeros(64,2);
k=1;
for i=1:8
	for j=i:8
		if(i==j)
			com(k,1) = values1(1,i);
			com(k,2) = values1(1,j);
			k++;
			
		else
			com(k,1) = values1(1,i);
			com(k,2) = values1(1,j);
			k++;
			com(k,1) = values1(1,j);
			com(k,2) = values1(1,i);
			k++;
		
		end
	end
end



% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
predError = Inf;


for i=1:size(com,1)
	model= svmTrain(X, y, com(i,1), @(x1, x2) gaussianKernel(x1, x2, com(i,2))); 
	predictions = svmPredict(model, Xval);
	pError = mean(double(predictions ~= yval));
	if(pError < predError)
		predError = pError;
		C = com(i,1);
		sigma = com(i,2);
	end
end

% C = 1;
% sigma = 0.3;


% =========================================================================

end
