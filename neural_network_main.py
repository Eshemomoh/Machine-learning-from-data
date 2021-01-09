# -*- coding: utf-8 -*-
"""
My neural network algorithm using numpy
Created on Mon Jan  4 11:46:31 2021

@author: Lucky Yerimah
"""
import numpy as np
import  matplotlib.pyplot as plt
from tqdm import tqdm



def signx(x):
    if x > 0:
        return 1
    else:
        return -1


act = lambda x: np.tanh(x)

dff_act = lambda x: 1 - (np.tanh(x))**2
    
sigmoid = lambda x: 1/(1+np.exp(np.negative(x)))


dff_sigmoid = lambda x: np.multiply(sigmoid(x),(1-sigmoid(x)))

sign = lambda x: signx(x)

    

identity = lambda x: x

def dff_identity1(x):
    nx = np.size(x)
    return np.ones((nx))

dff_identity = lambda x: dff_identity1(x)


class Neuralnet:
    def __init__(self,N,layers,activation,activation_final):
        # middle layer activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_dd = dff_sigmoid
        elif activation == 'tanh':
            self.activation = act
            self.activation_dd = dff_act
        elif activation == 'identity':
            self.activation = identity
            self.activation_dd = dff_identity
        
        # final layer
        if activation_final == 'sigmoid':
            self.activation_final = sigmoid
            self.activation_fin_dd = dff_sigmoid
        elif activation_final == 'tanh':
            self.activation_final == act
            self.activation_fin_dd == dff_act
        elif activation_final == 'identity':
             self.activation_final = identity
             self.activation_fin_dd = dff_identity
        elif activation_final == 'sign':
            self.activation_final = signx
            self.activation_fin_dd = 0
            
        self.weights = []
        #self.grad = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i]+1,layers[i+1])
            #grad = np.zeros(layers[i]+1,layers[i+1])
            self.weights.append(w)
            #self.grad.append(grad)
            
        self.sig = []
        self.sig2 = []
        for i in range(len(layers)):
            sig = np.zeros(layers[i])
            sig2 = np.zeros([layers[i],N])
            self.sig.append(sig)
            self.sig2.append(sig2)
            
        self.dff_sig = []
        for i in range(len(layers)):
            dff_sig = np.zeros(layers[i])
            self.dff_sig.append(dff_sig)
            
    
    def frontprop1(self,x,weights):
        sig1 = x.T
        ny,nx = np.shape(x)
        input_layer1 = np.concatenate((np.ones([1,ny]),sig1),axis=0)
        next_layer1 = np.matmul(weights[0].T,input_layer1)
        sig2 = self.activation(next_layer1)
        
        input_layer2 = np.concatenate((np.ones([1,ny]),sig2),axis=0)
        next_layer2 = np.matmul(weights[1].T,input_layer2)
        sig3 = self.activation_final(next_layer2)
        
        return sig3
    
    def frontprop(self,x):
        sig1 = x.T
        dff_sig = []
        dff_sig.append(self.activation_dd(sig1))
        
        input_layer1 = np.insert(sig1,0,1)
        next_layer1 = np.matmul(self.weights[0].T,input_layer1)
        dff_sig.append(self.activation_dd(next_layer1))
        sig2 = self.activation(next_layer1)
        
        input_layer2 = np.insert(sig2,0,1)
        next_layer2 = np.matmul(self.weights[1].T,input_layer2)
        dff_sig.append(self.activation_fin_dd(next_layer2))
        sig3 = self.activation_final(next_layer2)
        
        return sig1,sig2,sig3,dff_sig


    def errorf(self,hx,y):
        return 0.25*(hx-y)**2
    
    
    def backprop(self,x,y,grads,N):

        sig1,sig2,sig3,dff_sig = self.frontprop(x)
        
        error = 0.5*(sig3-y)*dff_sig[2]
        delta3 = error
        
        delta2 = np.multiply(dff_sig[1],(np.matmul(self.weights[1][1:],delta3)))
        delta2 = np.array([delta2])
        #delta1 = np.multiply(dff_sig[0],(np.matmul(self.weights[0][1:],delta2)))
        new_sig = np.array([np.insert(sig1,0,1)])
        
        grads[0] = grads[0]+ np.matmul(new_sig.T,delta2)/N #+ np.matmul((lambda1/(2*N)),self.weights[0])
        new_sig2 = np.array([np.insert(sig2,0,1)])
        delta3 = np.array([delta3])
        
        grad2new = np.multiply(new_sig2.T,delta3.T)
        grads[1] = grads[1]+ grad2new/N #+ np.matmul((lambda1/(2*N)),self.weights[1])
        
        return grads
        
        
    def weightupdate(self,grads,rate):
        self.weights[0] = self.weights[0] - rate*grads[0]
        self.weights[1] = self.weights[1] - rate*grads[1]
        return self.weights
    
    
    
    
        
if __name__ == "__main__":
    m = 10
    layers =  [2,m,1]
    activation = 'tanh'
    activation_final = 'identity'
    iteration = 2e3
    counter = 0
    rate = 1
    alpha = 1.1
    beta = 0.8
    
    
    #import data
    trainset = np.genfromtxt('trainset.csv', delimiter=',')
    testset = np.genfromtxt('testset.csv', delimiter=',')
    x = trainset[:,0:2]
    y = trainset[:,-1]
    N,M = np.shape(x)
    model = Neuralnet(10,layers,activation,activation_final)
    hx = model.frontprop1(x,model.weights)
    Error = model.errorf(hx,y)/N
    Ein = []
    nwy1,nwx1 = np.shape(model.weights[0])
    nwy2,nwx2 = np.shape(model.weights[1])
    
    Ein.append(np.sum(Error))
    
    for i in tqdm(range(2000000)):
    
        grads = []
        grads.append(np.zeros([nwy1,nwx1]))
        grads.append(np.zeros([nwy2,nwx2]))
        for j in range(N):
            grads = model.backprop(x[j,:],y[j],grads,N)
            
        temp_weights = model.weightupdate(grads,rate)
        hx2 = model.frontprop1(x,temp_weights)
        Error2 = model.errorf(hx2,y)/N
        New_error = np.sum(Error2)
        
        if New_error < Ein[-1]:
            model.weights = temp_weights
            rate = alpha*rate
        else:
            rate = beta*rate
            
        Ein.append(New_error)
        
    
    
    plt.plot(Ein)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
