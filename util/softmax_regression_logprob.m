function [p] = softmax_regression_logprob(w, x, y, L)
% [p] = softmax_regression_logprob(w, x, y, L)
% Average log probability under softmax regression.
% INPUTS
%       w    D*L weight vector
%       x    DxN data matrix
%       y    LxN binary label matrix, 1-out-of-L columns
%       L    number of classes
% OUTPUTS
%       p    average log probability 1/N * logP(y|x,w)
%
% George Papamakarios, Nov 2014

y = logical(y);

lp = logsoftmax(w, x, L);
p = mean(lp(y));
