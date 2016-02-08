#!/bin/python

import numpy as np

def t_update(tp):
    """
    This function produces t-updates
    arguments:
       tp: the previous value of t
    returns:
       tn: updated value of t
    """
    tn = (1 + np.sqrt(1 + 4 * tp**2))/2
    return tn

def v_update():
    """
    """
    vn = wn + (wn - wp) * (tp - 1)/tn
    return vn



def logloss_gradient(w,X,y):
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
    z = np.dot(X,w)
    p = 1/(1 + np.exp(-z))
    g = np.dot((p - y),X)/n
    return g
