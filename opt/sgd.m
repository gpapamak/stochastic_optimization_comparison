function [x, info] = sgd(x, f, df, f_star, options)
% [x, info] = sgd(x, f, df, f_star, options)
% Minimises a function using stochastic gradient descent.
% INPUTS
%       x         decision variable initialisation
%       f         handle of the function
%       df        array of handles of the gradients
%       f_star    optimal function value
%       options   a struct of options (optional)
% OUTPUTS
%       x         minimiser
%       info      a struct with execution information
%
% George Papamakarios, Nov 2014

x = x(:);
df = df(:);
N = length(df);

% options
if nargin < 5
    options = struct;
end
if ~isfield(options, 'step')
    options.step = 0.1;
end
if ~isfield(options, 'lambda')
    options.lambda = 0.1;
end
if ~isfield(options, 'tol')
    options.tol = 1.0e-8;
end
if ~isfield(options, 'batch_size')
    options.batch_size = 1;
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
iter = 0;
epoch = 0;
grad_eval = 0;
err = inf;
num_batches = floor(N / options.batch_size);
store_info = nargout > 1;
if store_info
    info.grad_eval = grad_eval;
    info.err = err;
    if options.store_x
        info.x = x;
    end
end

% iterate
while err > options.tol && epoch < options.max_epoch
    
    % randomly permute gradients
    df = df(randperm(N));
    
    % for each batch
    for i = 1:num_batches
        
        dfi = 0;
        for n = (i-1)*options.batch_size + 1 : i*options.batch_size
            dfi = dfi + df{n}(x);
        end
        dfi = dfi / options.batch_size;
        grad_eval = grad_eval + options.batch_size;
        
        step = options.step / (1 + options.step * options.lambda * iter);
        x = x - step * dfi;
        iter = iter + 1;
        
    end
    
    epoch = epoch + 1;
    err = f(x) - f_star;
    
    % store info
    if store_info
        info.grad_eval = [info.grad_eval grad_eval];
        info.err = [info.err err];
        if options.store_x
            info.x = [info.x x];
        end
    end
    
    % print info
    if options.verbose
        fprintf('Epoch = %d, error = %g \n', epoch, err);
    end
    
end
