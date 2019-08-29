function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1) X];

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%

% size(X) 
% size(Theta2) %= 10 , 26
% size(Theta1) %= 25 401

y1 = zeros(size(y,1),num_labels);
for i = 1:size(y,1)
	y1(i,y(i,:)) = 1;
end

Y = y1;

a2 = sigmoid(X*Theta1');
a2 = [ones(m, 1) a2];
a3 = sigmoid(a2*Theta2');

temp1= 0;
temp2 = 0;
temp3 = 0;

for i=1:m
	for j=1:num_labels
		temp1 = temp1 + (-y1(i,j))*log(a3(i,j)) - (1-y1(i,j))*log(1-a3(i,j));
	end
end



for i=1:hidden_layer_size
	for j= 2:input_layer_size+1
		temp2 = temp2 + Theta1(i,j)**2;
	end
end

for i=1:num_labels
	for j=2:hidden_layer_size+1
		temp3 = temp3 + Theta2(i,j)**2;
	end
end


J = (1/m)*temp1 + (lambda/(2*m))*(temp2 + temp3);


% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

	
% % for i = 1:m
% 	d3 = a3 - y1;
% 	d2 = (d3*Theta2).*sigmoidGradient(a2);
% 	d2 = d2(:,2:end);
% 	delta2 = zeros(size(d3'*a2));
% 	delta1 = zeros(size(d2'*X));
% 	delta2 = delta2 + d3'*a2; 
% 	delta1 = delta1 + d2'*X;

% % 	size(Theta1_grad) 
% % size(Theta2_grad)
% % size(Theta2) %= 10 , 26
% % size(Theta1) %= 25 401
% % size(d3)  %= 5000 10
% size(d2) %= 5000 25
% % size(a2) %= 5000 26

% %  size(delta1) %= 26 401
% %  size(delta2) %= 10 26
 


% 	% Theta2_grad(1,:) = (1/m).*delta2(1,:);
% 	% Theta2_grad(2:end,:) = (1/m).*delta2(2:end,:) + (lambda/m).*Theta2(2:end,:);
% 	% % Theta1_grad = (1/m).*delta1;
% 	% Theta1_grad(1,:) = (1/m).*delta1(1,:);
% 	% Theta1_grad(2:end,:) = (1/m).*delta1(2:end,:) + (lambda/m).*Theta1(2:end,:);

% 	for i = 1:size(delta2,1)
% 		for j = 1:size(delta2,2)
% 			if(j==1) Theta2_grad(i,j) = (1/m)*delta2(i,j) ;
% 			else Theta2_grad(i,j) = (1/m)*delta2(i,j) + (lambda/m)*Theta2(i,j);
% 			endif;
% 		end
% 	end

% 	for i = 1:size(delta1,1)
% 		for j = 1:size(delta1,2)
% 			if(j==1) Theta1_grad(i,j) = (1/m)*delta1(i,j) ;
% 			else Theta1_grad(i,j) = (1/m)*delta1(i,j) + (lambda/m)*Theta1(i,j);
% 			endif;
% 		end
% 	end



temp1 = Theta1(:, 2:end);
temp2 = Theta2(:, 2:end);
Delta1 = 0;
Delta2 = 0;

for t = 1:m
	a1 = X(t, :)';
	a2 = [1; sigmoid(Theta1 * a1)]; 
	a3 = sigmoid(Theta2 * a2);
	d3 = a3 - Y(t, :)';
	
	d2 = (temp2' * d3) .* sigmoidGradient(Theta1 * a1);
	Delta2 += (d3 * a2');
	Delta1 += (d2 * a1');
endfor

Theta2_grad = (1 / m) * Delta2;
Theta1_grad = (1 / m) * Delta1;

Theta2_grad(:, 2:end) += ((lambda / m) *temp2);
Theta1_grad(:, 2:end) += ((lambda / m) *temp1);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
