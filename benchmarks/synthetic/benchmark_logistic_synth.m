% Benchmarks GD, SGD, S2GD and SAG on synthetic data.
%
% The task is to fit a regularized logistic regression model on the
% synthetic dataset.
%
% Each optimization algorithm is benchmarked in turn, and the results for
% each are saved in a corresponding folder. The three stochastic algorithms
% are benchmarked for various minibatch sizes.
%
% George Papamakarios, Nov 2014

clear;
close all;

% load synthetic data
load(fullfile('data', 'synthetic', 'data_100d_100.mat'));
outdir = fullfile('results', 'synthetic', 'data_100d_100');
mkdir(outdir);

% negative log likelihood and its gradient
f = @(w) -mean(log(sigm(y.*(w'*x)))) + lambda * (w'*w) / 2;
df = @(w) -mean((ones(D,1) * (sigm(-y.*(w'*x)) .* y)) .* x, 2) + lambda * w;
assert(checkgrad(5 * randn(D, 10), f, df) < 1.0e-5);

fn = cell(1, N);
dfn = cell(1, N);
for n = 1:N
    fn{n} = @(w) -log(sigm(y(n)*(w'*x(:,n)))) + lambda * (w'*w) / 2;
    dfn{n} = @(w) -sigm(-y(n)*(w'*x(:,n))) * y(n) * x(:,n) + lambda * w;
    assert(checkgrad(5 * randn(D, 1), fn{n}, dfn{n}, 1) < 1.0e-6);
end

% general options for optimization algorithms
w_init = zeros(D, 1);
options.tol = -inf;
options.max_epoch = 2000;
options.verbose = true;
options.lambda = lambda;
batch_size = [1, 10, 100];

%% gd
options.step = 0.05;
[~, info_gd] = gd(w_init, f, df, f(w_star), options); %#ok<ASGLU>
save(fullfile(outdir, 'gd.mat'), 'info_gd');
clear info_gd;

%% sgd
info_sgd = cell(size(batch_size));

for i = 1:length(batch_size)
    options.batch_size = batch_size(i);
    options.step = 0.1 * options.batch_size;
    [~, info_sgd{i}] = sgd(w_init, f, dfn, f(w_star), options);
end
save(fullfile(outdir, 'sgd.mat'), 'info_sgd');
clear info_sgd;

%% s2gd
info_s2gd = cell(size(batch_size));

for i = 1:length(batch_size)
    options.batch_size = batch_size(i);
    options.step = 0.0001 * options.batch_size;
    [~, info_s2gd{i}] = s2gd(w_init, f, dfn, f(w_star), options);
end
save(fullfile(outdir, 's2gd.mat'), 'info_s2gd');
clear info_s2gd;

%% sag
info_sag = cell(size(batch_size));

for i = 1:length(batch_size)
    options.batch_size = batch_size(i);
    options.step = 0.00005 * options.batch_size;
    [~, info_sag{i}] = sag(w_init, f, dfn, f(w_star), options);
end
save(fullfile(outdir, 'sag.mat'), 'info_sag');
clear info_sag;

%% plot all results
close all;

% options for the plots
fontsize = 16;
markersize = 5;
linewidth = 3;

% load results
load(fullfile(outdir, 'gd.mat'));
load(fullfile(outdir, 'sgd.mat'));
load(fullfile(outdir, 's2gd.mat'));
load(fullfile(outdir, 'sag.mat'));

% plot them all
for i = 1:length(batch_size)
    figure;
    semilogy(info_gd.epoch * N, info_gd.err, 'ks:', 'MarkerSize', markersize, 'Linewidth', linewidth); hold on;
    semilogy(info_sgd{i}.grad_eval, info_sgd{i}.err, 'r>:', 'MarkerSize', markersize, 'Linewidth', linewidth);
    semilogy(info_s2gd{i}.grad_eval, info_s2gd{i}.err, 'b^:', 'MarkerSize', markersize, 'Linewidth', linewidth);
    semilogy(info_sag{i}.grad_eval, info_sag{i}.err, 'go:', 'MarkerSize', markersize, 'Linewidth', linewidth);
    xlabel('Number of gradient evaluations', 'FontSize', fontsize);
    ylabel('Optimality gap', 'FontSize', fontsize);
    legend('GD', 'SGD', 'S2GD', 'SAG', 'Location', 'NorthEast');
    set(gca, 'FontSize', fontsize);
    xlim([0 info_gd.epoch(end) * N]);
    ylim(10.^[-15 5]);
end
