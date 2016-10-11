% Benchmarks GD, SGD, S2GD and SAG on the MNIST dataset of handwritten
% digits.
%
% The task is to fit a regularized softmax regression model on MNIST. We
% use all MNIST classes.
%
% Each optimization algorithm is benchmarked in turn, and the results for
% each are saved in a corresponding folder. The three stochastic algorithms
% are benchmarked for various minibatch sizes.
%
% Apart from the training progress of each algorithm, as this is measured
% by the objective function during training, we also calculate and plot the
% accuracy and mean log probability on the MNIST test set, as these changed
% during training. The point of doing this is to see whether improved
% convergence leads to improvements in generalization preformance or not.
%
% George Papamakarios, Nov 2014

clear;
close all;

% load data
load ../../data/mnist/all_data_0.001.mat
outdir = 'results_all_data_0.001';
mkdir(outdir);

% negative log likelihood and gradient of regularized softmax regression
f = @(w) -softmax_regression_logprob(w, x_trn, y_trn, L) + lambda * w'*w / 2;
df = @(w) -softmax_regression_logprob_grad(w, x_trn, y_trn, L) + lambda * w;
assert(checkgrad(5 * randn(D*L, 5), f, df) < 1.0e-5);

fn = cell(1, N);
dfn = cell(1, N);
for n = 1:N
    fn{n} = @(w) -softmax_regression_logprob(w, x_trn(:,n), y_trn(:,n), L) + lambda * w'*w / 2;
    dfn{n} = @(w) -softmax_regression_logprob_grad(w, x_trn(:,n), y_trn(:,n), L) + lambda * w;
    assert(checkgrad(5 * randn(D*L, 1), fn{n}, dfn{n}, 1) < 1.0e-6);
end

% general options for optimization algorithms
w_init = zeros(D*L, 1);
options.tol = -inf;
options.max_epoch = 30;
options.verbose = true;
options.lambda = lambda;
options.store_x = true;
batch_size = [1, 10, 100];

%% gd
options.step = 0.5;
[~, info_gd] = gd(w_init, f, df, f(w_star), options);
save(fullfile(outdir, 'gd.mat'), 'info_gd');
clear info_gd;

%% sgd
info_sgd = cell(size(batch_size));

for i = 1:length(batch_size)
    options.batch_size = batch_size(i);
    options.step = 1 * options.batch_size;
    [~, info_sgd{i}] = sgd(w_init, f, dfn, f(w_star), options);
end
save(fullfile(outdir, 'sgd.mat'), 'info_sgd');
clear info_sgd;

%% s2gd
info_s2gd = cell(size(batch_size));

for i = 1:length(batch_size)
    options.batch_size = batch_size(i);
    options.step = 0.01 * options.batch_size;
    [~, info_s2gd{i}] = s2gd(w_init, f, dfn, f(w_star), options);
end
save(fullfile(outdir, 's2gd.mat'), 'info_s2gd');
clear info_s2gd;

%% sag
info_sag = cell(size(batch_size));

for i = 1:length(batch_size)
    options.batch_size = batch_size(i);
    options.step = 0.001 * options.batch_size;
    [~, info_sag{i}] = sag(w_init, f, dfn, f(w_star), options);
end
save(fullfile(outdir, 'sag.mat'), 'info_sag');
clear info_sag;

%% plot all results
close all;

% options for the plots
fontsize = 16;
markersize = 6;
linewidth = 3;

% load results
load(fullfile(outdir, 'gd.mat'));
load(fullfile(outdir, 'sgd.mat'));
load(fullfile(outdir, 's2gd.mat'));
load(fullfile(outdir, 'sag.mat'));

% plot training progress
for i = 1:length(batch_size)
    figure;
    semilogy(info_gd.epoch * N, info_gd.err, 'ks:', 'MarkerSize', markersize, 'Linewidth', linewidth); hold on;
    semilogy(info_sgd{i}.grad_eval, info_sgd{i}.err, 'r>:', 'MarkerSize', markersize, 'Linewidth', linewidth);
    semilogy(info_s2gd{i}.grad_eval, info_s2gd{i}.err, 'b^:', 'MarkerSize', markersize, 'Linewidth', linewidth);
    semilogy(info_sag{i}.grad_eval, info_sag{i}.err, 'go:', 'MarkerSize', markersize, 'Linewidth', linewidth);
    xlabel('Number of gradient evaluations', 'FontSize', fontsize);
    ylabel('Optimality gap', 'FontSize', fontsize);
    legend('GD', 'SGD', 'S2GD', 'SAG', 'Location', 'SouthWest');
    set(gca, 'FontSize', fontsize);
    xlim([0 info_gd.epoch(end) * N]);
    ylim(10.^[-15 2]);
end

% calculate and accuracy and mean log probability on the test set, as these
% progressed during training

% gd
epochs = length(info_gd.epoch);
acc_trn_gd = zeros(1, epochs);
mlp_trn_gd = zeros(1, epochs);
acc_tst_gd = zeros(1, epochs);
mlp_tst_gd = zeros(1, epochs);
for i = 1:epochs
    [acc_trn_gd(i), ~, mlp_trn_gd(i), ~] = eval_pred_softmax(info_gd.x(:,i), x_trn, y_trn);
    [acc_tst_gd(i), ~, mlp_tst_gd(i), ~] = eval_pred_softmax(info_gd.x(:,i), x_tst, y_tst);
end

% sgd
epochs = length(info_sgd{1}.grad_eval);
acc_tst_sgd = zeros(1, epochs);
mlp_tst_sgd = zeros(1, epochs);
for i = 1:epochs
    [acc_tst_sgd(i), ~, mlp_tst_sgd(i), ~] = eval_pred_softmax(info_sgd{1}.x(:,i), x_tst, y_tst);
end

% s2gd
epochs = length(info_s2gd{1}.grad_eval);
acc_tst_s2gd = zeros(1, epochs);
mlp_tst_s2gd = zeros(1, epochs);
for i = 1:epochs
    [acc_tst_s2gd(i), ~, mlp_tst_s2gd(i), ~] = eval_pred_softmax(info_s2gd{1}.x(:,i), x_tst, y_tst);
end

% sag
epochs = length(info_sag{1}.grad_eval);
acc_tst_sag = zeros(1, epochs);
mlp_tst_sag = zeros(1, epochs);
for i = 1:epochs
    [acc_tst_sag(i), ~, mlp_tst_sag(i), ~] = eval_pred_softmax(info_sag{1}.x(:,i), x_tst, y_tst);
end

% plot accuracy
figure;
plot(info_gd.epoch * N, acc_tst_gd, 'ks:', 'MarkerSize', markersize, 'Linewidth', linewidth); hold on;
plot(info_sgd{1}.grad_eval, acc_tst_sgd, 'r>:', 'MarkerSize', markersize, 'Linewidth', linewidth);
plot(info_s2gd{1}.grad_eval, acc_tst_s2gd, 'b^:', 'MarkerSize', markersize, 'Linewidth', linewidth);
plot(info_sag{1}.grad_eval, acc_tst_sag, 'go:', 'MarkerSize', markersize, 'Linewidth', linewidth);
xlabel('Number of gradient evaluations', 'FontSize', fontsize);
ylabel('Accuracy on test set', 'FontSize', fontsize);
legend('GD', 'SGD', 'S2GD', 'SAG', 'Location', 'SouthEast');
set(gca, 'FontSize', fontsize);
xlim([0 info_gd.epoch(end) * N / 4]);
ylim([0 1]);

% plot mean log probability
figure;
plot(info_gd.epoch * N, mlp_tst_gd, 'ks:', 'MarkerSize', markersize, 'Linewidth', linewidth); hold on;
plot(info_sgd{1}.grad_eval, mlp_tst_sgd, 'r>:', 'MarkerSize', markersize, 'Linewidth', linewidth);
plot(info_s2gd{1}.grad_eval, mlp_tst_s2gd, 'b^:', 'MarkerSize', markersize, 'Linewidth', linewidth);
plot(info_sag{1}.grad_eval, mlp_tst_sag, 'go:', 'MarkerSize', markersize, 'Linewidth', linewidth);
xlabel('Number of gradient evaluations', 'FontSize', fontsize);
ylabel('Mean log probability of test set', 'FontSize', fontsize);
legend('GD', 'SGD', 'S2GD', 'SAG', 'Location', 'SouthEast');
set(gca, 'FontSize', fontsize);
xlim([0 info_gd.epoch(end) * N / 4]);
