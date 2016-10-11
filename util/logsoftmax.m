function [y] = logsoftmax(w, x, L)
% [y] = logsoftmax(w, x, L)
% Log softmax function.
% INPUTS
%       w    D*L weight vector
%       x    DxN data matrix
%       L    number of classes
% OUTPUTS
%       y    LxN matrix, the exp of which columnwise sums to one
%
% George Papamakarios, Nov 2014

D = size(x, 1);
w = reshape(w, [D L]);

y = w'*x;
y = y - ones(L, 1) * logsumexp(y);
