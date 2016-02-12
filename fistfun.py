#!/bin/python

import numpy as np

def prox(z,alpha,dt):
    """
    Function implements the PROXIMAL solution 
    arguments:
       z:     array of dimension (p,)
       alpha: parameter scalar
       dt:    time step, a scalar
    returns:
       w:     vector with estimater parameters, dimension (p,)
    """
    gamma = 2 * dt * (1 - alpha) + 1
    w = ((z - alpha * dt) / gamma) * (z > alpha * dt) + ((z + alpha * dt) / gamma) *  (z < - alpha * dt)
    return w

def loss(X,y,w):
    """
    Logistic loss function:
    arguments:
        w: numpy array of feature coefficients with shape (p,)
        X: numpy array with shape (n,p)
        y: numpy array of responses with shape (n,)
    returns:
        l: loss function value, a scalar    
    """
    n = X.shape[0]
    e = 1 + np.exp(-np.dot(X,w))
    l = ( (1 - y) * np.dot(X,w) + np.log(e) ) / n
    return np.sum(l)

def linesearch(X,y,dt,p,w,alpha,lam):
    """
    """
    w_new = prox((w - dt*p),alpha,dt)
    
    while True:
        grad = log_gradient(w,X,y)
        q = loss(X,y,p,lam) + np.dot(w-p,grad) + np.dot(w-p,w-p)/(2*dt)
        if q >= loss(X,y,w_new,lam):
            dt = 1.1 *dt
            w = w_new
            if dt < 1e-12: raise ValueError('dt is too small')
            break

        else:
            dt = dt/2
            if dt < 1e-12: raise ValueError('dt is too small')

    return dt,w

def log_gradient(X,y,w):
    """
    Calculates gradient of the logistic loss;
    arguments:
        w: numpy array of feature coefficients with shape (p,)
        X: numpy array with shape (n,p)
        y: numpy array of responses with shape (n,)
    returns:
        g: numpy array containing gradient values with shape (p,)
    """
    n = X.shape[0]
    p = 1 / ( 1 + np.exp(-np.dot(X,w)) )
    g = np.dot((p - y),X) / n
    return g
