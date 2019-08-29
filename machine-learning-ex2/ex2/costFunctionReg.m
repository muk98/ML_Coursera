function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
temp1 = 0;
temp2 = 0;
% temp = zeros(size(theta));
z= theta'*X';
g = sigmoid(z);
for i = 1:size(z,2)
	temp1 = temp1 + (-(y(i)*log(g(i))) - (1-(y(i)))*log(1-g(i)) );
endfor	

for i = 2:size(theta)
	temp2 = temp2 + theta(i)**2;
endfor


J = (1/m)*(temp1) + (lambda/(2*m))*temp2;


for i = 1:size(theta)
	temp1 = 0;
	for j = 1:size(z,2)
		temp1 = temp1 + (g(j) - y(j))*X(j,i);
	endfor
	if (i == 1)
		grad(i) = (1/m)*temp1 ;
	else 
		grad(i) = (1/m)*temp1 + (lambda/m)*theta(i);
	endif
endfor


% =============================================================

end
