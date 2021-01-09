Machine learning from data Homework 12 readme to acompany codes and solution

Author: Lucky Yerimah
Email: Lucky.yerimah@gmail.com
Date: Dec/10/2020

Note: All codes are written in MATLAB

Neural Network
	To run the neural network, have the digit data present in the same folder as the files, open and run "nn_main.m" in 
	the neural network folder. This will call on "getdata.m" to load the digitdata, compute features and normalize 
	the data. Next backprop is called where forward propagation and backpropagation takes place. Temporary weights 
	are then calculated and used to compute a new Ein. If New Ein is lower than previous Ein, temporary weights are 
	then used to replace model weights. 

	Note: Some of the function files are duplicated and modified to reduce running time but "main.m" should be able to run the most updated functions
	Note: my digit data is a .mat file. trainset, testset can be imported elsewhere and just used hence skipping "getdata.ma".

Support Vector Machines.
	To run the SVM algorithm, have digit data present and open "svm_main.mlx" Similar to the Neural Network, This will 
	call on "getdata.m" to load the digitdata, compute features and normalize the data. Next svmclassifier trains the 
	model to obtain svmm "SVM model in struct form"
	