function [x, info] = s2gd(x, f, df, f_star, options)
% [x, info] = s2gd(x, f, df, f_star, options)
% Minimises a function using semi-stochastic gradient descent.
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
% Based on the paper:
% J. Konecny and P. Richtarik, "Semi-Stochastic Gradient Descent Methods,"
% arXiv:1312.1666, 2013.
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

if ~isfield(options, 'max_inner_iter')
    options.max_inner_iter = num_batches;
end

% distribution of inner loop iterations
p_inner_iter = (1 - options.lambda * options.step) .^ (options.max_inner_iter-1: -1 : 0);
p_inner_iter = p_inner_iter / sum(p_inner_iter);

% iterate
while err > options.tol && epoch < options.max_epoch
    
    % randomly permute gradients
    df = df(randperm(N));
    
    % compute full gradient
    full_df = 0;
    for n = 1:N
        full_df = full_df + df{n}(x);
    end
    full_df = full_df / N;
    x0 = x;
    grad_eval = grad_eval + N;
    
    % for each inner iteration
    for j = 1:discrete_sample(p_inner_iter)
        
        i = mod(j, num_batches) + 1;

        dfi = 0;
        dfi0 = 0;
        for n = (i-1)*options.batch_size + 1 : i*options.batch_size
            dfi = dfi + df{n}(x);
            dfi0 = dfi0 + df{n}(x0);
        end
        dfi = dfi / options.batch_size;
        dfi0 = dfi0 / options.batch_size;
        grad_eval = grad_eval + options.batch_size;
        
        x = x - options.step * (full_df + dfi - dfi0);
        
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
