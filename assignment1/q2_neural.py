#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    
    #print(params.shape)
    
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    
    h = X.dot(W1) + b1
    a = sigmoid(h)
    
    h2 = a.dot(W2) + b2
    
    # vector loss Nx1
    a2 = softmax(h2)
    loss_layer = a2[labels==1]
    
    #Note: Cost are computed but never used.
    cost = -np.sum(np.log(loss_layer))
    
    
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    
    # Compute loss gradient
    gradh2 = a2 - labels
    
    # loss signals on second layer
    gradb2 = np.sum(gradh2,axis=0, keepdims=True)
    #assert(gradb2.shape == (1,Dy))
    gradW2 = a.T.dot(gradh2) 
    #assert(gradW2.shape == (H,Dy))
    
    # crossing sigmoid layers
    grada = gradh2.dot(W2.T)
    #assert(a_grad.shape == a.shape)
    gradh = grada * sigmoid_grad(a)
    
    # signals on the first layer
    gradb1 = np.sum(gradh, axis=0, keepdims=True)
    #assert(gradb1.shape == (1,H))
    gradW1 = X.T.dot(gradh)
    #assert(gradW1.shape == (Dx,H))
   
    
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)
        

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    print("WELL DONE")
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
