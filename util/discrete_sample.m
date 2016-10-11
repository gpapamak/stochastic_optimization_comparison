function [x] = discrete_sample(p)
% [x] = discrete_sample(p)
% Returns a sample from a discrete distribution.
% INPUTS
%       p    discrete distribution of length N
% OUTPUTS
%       x    a sample in {1,2,...,N}
%
% George Papamakarios, Nov 2014

p = p(:);
c = cumsum(p);

assert(abs(c(end) - 1) < 1.0e-12, 'Probabilities must sum to one.');
assert(all(p >= 0), 'Probabilities must be positive.');

x = sum(rand > c) + 1;
