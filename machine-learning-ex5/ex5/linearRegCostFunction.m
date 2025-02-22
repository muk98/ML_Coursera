function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_theta = X*theta;
z= theta'*X';

J = (1/(2*m))*(sum((h_theta - y)'*(h_theta - y))) + (lambda/(2*m))*sum(theta(2:end,1)'*theta(2:end,1));

temp1 = 0;
temp2 = 0;

% grad = (1/m).*(X'*X*theta - X'*y) + (lambda/m).*(theta);
for i = 1:size(theta,1)
	temp1 = 0;
	for j = 1:size(z,2)
		temp1 = temp1 + (h_theta(j) - y(j))*X(j,i);
	endfor
	if (i == 1)
		grad(i) = (1/m)*temp1 ;
	else 
		grad(i) = (1/m)*temp1 + (lambda/m)*theta(i);
	endif
endfor





% =========================================================================

grad = grad(:);

end
