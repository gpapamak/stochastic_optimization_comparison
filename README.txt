+---------------------------------------------------------+
|                                                         |
| Comparison of Modern Stochastic Optimization Algorithms |
|                                                         |
|                   George Papamakarios                   |
|                      November 2014                      |
|                                                         |
+---------------------------------------------------------+

**************** Before running anything ******************

Run "install.m" to add all the necessary paths to matlab's
search path. Then all scripts and functions in this folder
will become executable.

Adding these paths to the matlab's search path is not 
permanent; it will only last for this matlab session. So
no need to worry about polluting the search path.


*****************  This folder contains  ******************

- install.m   :   the scipt you need to run before you do
                  anything else.
                      
+ opt         :   contains implementations of the four
                  optimization algorithms:
                  GD, SGD, S2GD and SAG.
                      
+ data        :   contains scripts for generating synthetic
                  data and preparing the MNIST data. these
                  datasets are needed for the benchmarks.
                  
+ benchmarks  :   contains scripts that run experiments.
                  datasets must have been generated first.
                  these scripts save and plot results.
                  
+ util        :   some utility functions used throughout
                  the project.
                              
***********************************************************
