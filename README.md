# Comparison of Modern Stochastic Optimization Algorithms

Code that reproduces the experiments in the technical report:
> G. Papamakarios. _Comparison of Modern Stochastic Optimization Algorithms_. Technical report. University of Edinburgh. 2014.
> [[pdf]](http://www.maths.ed.ac.uk/~richtarik/papers/Papamakarios.pdf) [[bibtex]](http://homepages.inf.ed.ac.uk/s1459647/bibtex/modern_stochastic_optimization.bib)

The experiments benchmark four optimization algorithms on two convex problems. The algorithms are:

* [Batch gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)
* [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
* [Semi-stochastic gradient descent](https://arxiv.org/pdf/1312.1666v2.pdf)
* [Stochastic average gradient](https://papers.nips.cc/paper/4633-a-stochastic-gradient-method-with-an-exponential-convergence-_rate-for-finite-training-sets.pdf)

And the tasks are:

* Logistic regression on synthetic data
* Softmax regression on the MNIST dataset of handwritten images


## How to run the experiments

First, run `install.m` to add all necessary paths to matlab's path. Then all scripts and functions in this folder will become executable.

### Logistic regression on synthetic data

1. Run `gen_synth_data.m` to generate a synthetic dataset for logistic regression. Modify parameters `N` and `D` to change number of datapoints and dimesions respectively.

2. Run `benchmark_logistic_synth.m` to benchmark all algorithms on the synthetic dataset. Results are written in the `results` folder.

### Softmax regression on MNIST

1. Download the following files from the MNIST [website](http://yann.lecun.com/exdb/mnist/):
 
 * train-images-idx3-ubyte.gz
 * train-labels-idx1-ubyte.gz
 * t10k-images-idx3-ubyte.gz
 * t10k-labels-idx1-ubyte.gz

2. Unzip them and place them in the folder `data/mnist` and run `prepare_mnist_data.m`.

3. Run `benchmark_softmax_mnist.m` to benchmark all algorithms on MNIST. Results are written in the `results` folder.


## This folder contains

- `install.m`: the script you need to run before you do anything else.
                      
- `opt`: contains implementations of the four optimization algorithms: GD, SGD, S2GD and SAG.
                      
- `data`: contains scripts for generating synthetic data and preparing the MNIST data. These datasets are needed for the benchmarks.
                  
- `benchmarks`: contains scripts that run experiments. Datasets must have been generated first. These scripts save and plot results.
                  
- `util`: some utility functions used throughout the project.
                              

