function [y] = sigm(x)
% [y] = sigm(x)
% Logistic sigmoid function.
%
% George Papamakarios, Nov 2014

y = 1 ./ (1 + exp(-x));

% handle underflow
ii = y == 0;
y(ii) = y(ii) + eps;

ii = y == 1;
y(ii) = y(ii) - eps;
