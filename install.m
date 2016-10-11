% install.m
% Adds all necessary paths to the system's search path. Run this before
% running anything else in this folder.
%
% George Papamakarios, Nov 2014

addpath(fullfile(pwd, 'opt'));
addpath(fullfile(pwd, 'util'));

addpath(fullfile(pwd, 'benchmarks'));
addpath(fullfile(pwd, 'benchmarks/mnist'));
addpath(fullfile(pwd, 'benchmarks/synthetic'));

addpath(fullfile(pwd, 'data'));
addpath(fullfile(pwd, 'data/mnist'));
addpath(fullfile(pwd, 'data/synthetic'));
