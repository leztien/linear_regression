"""
binned regression in 1D
"""

import numpy as np
import matplotlib.pyplot as plt


def preprocessing(x, bins=None, add_bias=False):
    try: X = x.reshape(-1,1)
    except AttributeError:
        from numpy import array
        X = array(x, dtype=float).reshape(-1,1)
    #binning
    from numbers import Integral
    from collections.abc import Sequence
    from numpy import linspace, digitize
    from sklearn.preprocessing import OneHotEncoder
    if bins:
        bins = bins if isinstance(bins, Sequence) and len(bins)==3 else (min(x), max(x), int(bins))
        breaks = linspace(*bins[:2], num=bins[-1]+1)
        nx = digitize(x, bins=breaks)
        nx[-1] = nx[-2]
        X = OneHotEncoder(sparse=False, categories='auto').fit_transform(nx.reshape(-1,1))
    preprocessing.bins = bins
    if add_bias:
        from numpy import c_, ones
        X = c_[ones(len(X)), X]
    return(X)


def normal_equasion(X,y):
    from scipy.linalg import inv
    weights = inv(X.T @ X) @ (X.T @ y)
    return weights


def predict(weights, x):
    from numpy import dot
    if len(weights) == len(x)+1: x = [1, *x]
    prediction = dot(weights, x)
    return prediction

#==============================================================

"""MAKE DATA"""  
x = np.linspace(0, 10, 50)
xtest = np.linspace(min(x), max(x), 500)
y = np.sin(x*2) + x/3 + np.random.randn(len(x))/3

"""PREPROCESSING, MODELING, PREDICTING"""
n_bins = 20
X = preprocessing(x, bins=n_bins)
weights = normal_equasion(X,y)

Xtest = preprocessing(xtest, bins=preprocessing.bins)
ypred = [predict(weights, x) for x in Xtest]

"""VISUALIZE"""
plt.plot(x,y, '.')
plt.gca().set_aspect("equal")
plt.plot(xtest, ypred)
