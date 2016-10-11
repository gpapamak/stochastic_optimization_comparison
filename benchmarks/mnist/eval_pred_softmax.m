function [acc_avg, acc_std, logprob_avg, logprob_std] = eval_pred_softmax(w, x, y_true)
% Evaluates the predictions of a softmax regression model. Returns accuracy
% and mean log probability, together with error bars.
% INPUTS
%       w            weights of softmax regression model
%       x            test input locations
%       y_true       true labels
% OUTPUTS
%       acc_avg      accuracy
%       acc_acc      accuracy error bar (+/- 1 stdev)
%       logprob_avg  mean log probability
%       logprob_std  mean log probability error bar (+/- 1 stdev)
%
% George Papamakarios, Nov 2014

% number of labels
L = size(y_true, 1);

% make predictions using softmax model
w = w(:);
y_prob = exp(logsoftmax(w, x, L));

% number of datapoints
N = size(y_prob, 2);

% predict with max probability
[~, y_pred] = max(y_prob);
[~, labels] = max(y_true);

% accuracy
acc = y_pred == labels;
acc_avg = mean(acc);
acc_std = std(acc) / sqrt(N);

% mean log probability
logprob = log(y_prob(logical(y_true)));
logprob_avg = mean(logprob);
logprob_std = std(logprob) / sqrt(N);
