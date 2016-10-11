function [x, info] = gd(x, f, df, f_star, options)
% [x, info] = gd(x, f, df, f_star, options)
% Minimises a function using gradient descent.
% INPUTS
%       x         decision variable initialisation
%       f         handle of the function
%       df        handle of the gradient
%       f_star    optimal function value
%       options   a struct of options (optional)
% OUTPUTS
%       x         minimiser
%       info      a struct with execution information
%
% George Papamakarios, Nov 2014

x = x(:);

% options
if nargin < 5
    options = struct;
end
if ~isfield(options, 'step')
    options.step = 0.1;
end
if ~isfield(options, 'tol')
    options.tol = 1.0e-8;
end
if ~isfield(options, 'max_epoch')
    options.max_epoch = inf;
end
if ~isfield(options, 'verbose')
    options.verbose = false;
end
if ~isfield(options, 'store_x')
    options.store_x = false;
end

% initialise
epoch = 0;
err = inf;
store_info = nargout > 1;
if store_info
    info.epoch = epoch;
    info.err = err;
    if options.store_x
        info.x = x;
    end
end

% iterate
while err > options.tol && epoch < options.max_epoch
    
    x = x - options.step * df(x);
    epoch = epoch + 1;
    err = f(x) - f_star;
    
    % print info
    if options.verbose
        fprintf('Epoch %d, error = %g \n', epoch, err);
    end
    
    % store info
    if store_info
        info.epoch = [info.epoch epoch];
        info.err = [info.err err];
        if options.store_x
            info.x = [info.x x];
        end
    end
    
end
