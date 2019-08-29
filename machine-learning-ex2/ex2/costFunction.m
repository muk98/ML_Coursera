function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
temp = 0;

% temp = zeros(size(theta));
z= theta'*X';
g = sigmoid(z);
for i = 1:size(z,2)
	temp = temp + (-(y(i)*log(g(i))) - (1-(y(i)))*log(1-g(i)) );
endfor	
J = (1/m)*(temp);


% grad(0) = (1/m)*(g(i)-y(i));
for i = 1:size(theta)
	temp = 0;
	for j = 1:size(z,2)
		temp = temp + (g(j) - y(j))*X(j,i);
	endfor
	grad(i) = (1/m)*temp;
endfor

% =============================================================

end
