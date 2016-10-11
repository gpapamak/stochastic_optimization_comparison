% Generates a synthetic dataset for logistic regression.
%
% Generation is done as follows:
% (1) Input locations are drawn from a gaussian distribution, and are 
%     labelled according to a logistic likelihood.
% (2) A regularized logistic regression model is fitted to the generated 
%     dataset using bach gradient descent, which is run for a very long
%     time in order to achieve high accuracy.
% (3) Finally, both the dataset and the logistic regression weights are
%     saved to a mat file.
%
% George Papamakarios, Nov 2014

clear;

% parameters
N = 100;
D = 100;
w_true = 0.5 * ones(D, 1);
lambda = 0.1;

% generate data (input locations and corresponding labels)
x = 20 * randn(D, N);
y = rand(1, N) < sigm(w_true' * x);
y = 2*y - 1;
assert(sum(y == 1) + sum(y == -1) == N);

% negative log likelihood and its derivative of a regularized logistic
% regression model
f = @(w) -mean(log(sigm(y.*(w'*x)))) + lambda * (w'*w) / 2;
df = @(w) -mean((ones(D,1) * (sigm(-y.*(w'*x)) .* y)) .* x, 2) + lambda * w;
assert(checkgrad(5 * randn(D, 10), f, df) < 1.0e-5);

% find optimum using batch gradient descent
options.step = 0.05; %1 / (sum(sum(x.^2)) / (4*N) + lambda);
options.max_epoch = 1e+5;
w_star = gd(w_true, f, df, -inf, options);

% save data and weights
savefile = fullfile('data', 'synthetic', 'data_100d_100.mat');
save(savefile, 'x', 'y', 'N', 'D', 'w_true', 'w_star', 'lambda');
