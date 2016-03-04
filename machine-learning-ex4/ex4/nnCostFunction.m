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
eye_matrix=eye(num_labels);
y_matrix=eye_matrix(y,:);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
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

a1=[ones(m,1),X];             %5000x401
z2=Theta1*a1';                %25x401*401x5000
a2=sigmoid(z2);               %25x5000
a2=a2';                       %5000x25
a2_with_1=[ones(m,1),a2];     %5000x26
z3=Theta2*a2_with_1';         %10x26*26x5000
a3=sigmoid(z3);               %10x5000
a3=a3';                       %5000x10
h=a3;

first=y_matrix.*log(h);       %5000x10 .* 5000x10 = 5000x10
second=(1-y_matrix).*log(1-h);      
J=-first-second;              %5000x10
J=sum(sum(J));                %1x1
J=J/m;

%Reg'd section
tempTheta1=Theta1(:,2:(input_layer_size+1));
tempTheta2=Theta2(:,2:(hidden_layer_size+1));

regdJ=((sum(sum(tempTheta1.^2)))+(sum(sum(tempTheta2.^2))))*lambda/(2*m);

J=J+regdJ;


delta_3=a3-y_matrix;                      %5000x10
delta_2=delta_3*Theta2;                   %5000x10 * 10x26 = 5000x26
delta_2=delta_2(:,2:end);                 %5000x25
delta_2=delta_2.*sigmoidGradient(z2');    %5000x25


D1=delta_2'*a1;                           %25x5000 * 5000x401 = 25x401            
D2=delta_3'*a2_with_1;                    %10x5000 * 5000x26 = 10x26
Theta1_grad=D1/m;
Theta2_grad=D2/m;

regd_grad1=Theta1(:,2:end);
regd_grad1=(lambda/m)*regd_grad1;
regd_grad1=[zeros(hidden_layer_size,1),regd_grad1];
regd_grad2=Theta2(:,2:end);
regd_grad2=(lambda/m)*regd_grad2;
regd_grad2=[zeros(num_labels,1),regd_grad2];

Theta1_grad=Theta1_grad+regd_grad1;
Theta2_grad=Theta2_grad+regd_grad2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
