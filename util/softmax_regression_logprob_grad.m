function [g] = softmax_regression_logprob_grad(w, x, y, L)
% [g] = softmax_regression_logprob_grad(w, x, y, L)
% Gradient of average log probability under softmax regression.
% INPUTS
%       w    D*L weight vector
%       x    DxN data matrix
%       y    LxN binary label matrix, 1-out-of-L columns
%       L    number of classes
% OUTPUTS
%       p    gradient of 1/N * logP(y|x,w) wrt w
%
% George Papamakarios, Nov 2014

N = size(x, 2);

p = y - exp(logsoftmax(w, x, L));
g = x * p';
g = g(:) / N;
