function [y] = logsumexp(x)
% [y] = logsumexp(x)
% Returns the log of the sum of the exps of the columns in x.
%
% George Papamakarios, Nov 2014

N = size(x, 1);
xmax = max(x);
x = x - ones(N, 1) * xmax;
y = xmax + log(sum(exp(x)));
