% Prepares the MNIST dataset of handwritten digits for softmax regression.
%
% Preparation of the dataset is done as follows:
% (1) The MNIST dataset is loaded and the labels are transformed into a 
%     more convenient 1-out-of-K representation.
% (2) A regularized softmax regression model is fitted to the dataset using
%     bach gradient descent, which is run for a very long time in order to
%     obtain high accuracy.
% (3) Finally, both the dataset and the softmax regression weights are
%     saved to a mat file.
%
% NOTE!!
% To run this script you need to download the following 4 files:
%
% train-images.idx3-ubyte
% t10k-images.idx3-ubyte
% train-labels.idx1-ubyte
% t10k-labels.idx1-ubyte
%
% from the official MNIST website:
% http://yann.lecun.com/exdb/mnist/
%
% and place them in the same folder as this file. These 4 files contain the
% MNIST dataset in binary format.
%
% George Papamakarios, Nov 2014

clear;

% load data
x_trn = loadMNISTImages('train-images.idx3-ubyte');
x_tst = loadMNISTImages('t10k-images.idx3-ubyte');
y_trn_idx = loadMNISTLabels('train-labels.idx1-ubyte')';
y_tst_idx = loadMNISTLabels('t10k-labels.idx1-ubyte')';

% number of distinct labels
L = max(y_trn_idx) + 1;

% transform labels to 1-out-of-K representation
y_trn = zeros(L, size(x_trn, 2));
y_trn(sub2ind(size(y_trn), y_trn_idx + 1, 1:length(y_trn_idx))) = 1;
y_tst = zeros(L, size(x_tst, 2));
y_tst(sub2ind(size(y_tst), y_tst_idx + 1, 1:length(y_tst_idx))) = 1;

% size of train set
[D, N] = size(x_trn);

% negative log likelihood and its gradient of a regularized softmax 
% regression model
lambda = 0.001;
f = @(w) -softmax_regression_logprob(w, x_trn, y_trn, L) + lambda * w'*w / 2;
df = @(w) -softmax_regression_logprob_grad(w, x_trn, y_trn, L) + lambda * w;
assert(checkgrad(5 * randn(D*L, 5), f, df) < 1.0e-5);

% fit softmax model using batch gradient descent, run for a very long time
w_init = zeros(D*L, 1);
options.tol = -inf;
options.max_epoch = 1e+4;
options.step = 0.5;
options.verbose = true;
w_star = gd(w_init, f, df, -inf, options);

% save dataset and softmax weights
save('all_data_0.001.mat', 'x_trn', 'x_tst', 'y_trn', 'y_tst', 'N', 'D', 'L', 'w_star', 'lambda');
