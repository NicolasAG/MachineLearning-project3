###########################################
### COMP 598 - Applied Machine Learning ###
###				Project 3				###
###########################################

The	goal of this project is to devise a	machine learning algorithm to automatically classify images of hand written digits (from 0 to 9) represented in cropped image.

These are the algorithm we used:
 - (1) A baseline learner consisting of logistic regression.
 - (2) A linear SVM.
 - (3) A fully connected feedforward neural network trained by backpropagation, where the network architecture (number of nodes / layers), learning rate and termination are determined by cross‐validation.


Requirements:
=============

Python requirements:

	csv
	idx2numpy
	...


File structure requirements:
	|data_and_scripts/
	|	|test_inputs.csv
	|	|train_inputs.csv
	|	|train_outputs.csv
	|	|MNIST/
	|	|	|t10k-images.idx3-ubyte
	|	|	|t10k-labels.idx1-ubyte
	|	|	|train-images.idx3-ubyte
	|	|	|train-labels.idx3-ubyte

by:
 - Andres Felipe Rincón
 - Ryan Razani
 - Nicolas Angelard-Gontier

