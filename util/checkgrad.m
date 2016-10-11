function [err] = checkgrad(x, f, df, times, eps)
% [err] = checkgrad(x, f, df, times, eps)
% Checks gradients using finite differences.
% INPUTS
%       x       column vectors where to check
%       f       handle of the function
%       df      handle of the gradient
%       times   number of times to check (optional)
%       eps     factor to multiply dx by (optional)
% OUTPUTS
%       err     maximum error encountered
%
% George Papamakarios, Nov 2014

if nargin < 5
    eps = 1.0e-9;
end
if nargin < 4
    times = 10;
end

[D, N] = size(x);
err = zeros(times, N);

for n = 1:N
    for t = 1:times

        dx = eps * randn(D, 1);
        err(t, n) = f(x(:,n) + dx) - f(x(:,n)) - dx' * df(x(:,n));

    end
end

err = max(abs(err(:)));
