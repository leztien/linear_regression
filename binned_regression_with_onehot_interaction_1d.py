"""
binned regression with one-hot interaction (in 1D)
"""

import numpy as np
import matplotlib.pyplot as plt


def preprocessing(x, bins=None, add_original_feature=False, add_interaction=False, add_bias=False):
    from collections.abc import Sequence
    from numpy import linspace, digitize, c_, ones
    from sklearn.preprocessing import OneHotEncoder
    
    try: X = x.reshape(-1,1)
    except AttributeError:
        from numpy import array
        X = array(x, dtype=float).reshape(-1,1)
    #binning
    if bins:
        bins = bins if isinstance(bins, Sequence) and len(bins)==3 else (min(x), max(x), int(bins))
        breaks = linspace(*bins[:2], num=bins[-1]+1)
        nx = digitize(x, bins=breaks)
        nx[-1] = nx[-2]
        Xbinned = OneHotEncoder(sparse=False, categories='auto').fit_transform(nx.reshape(-1,1))
    preprocessing.bins = bins
    
    assert not(add_original_feature and add_interaction),"error: cant be both"
    if add_original_feature:
        Xbinned = c_[Xbinned, X]
    if add_interaction:
        Xbinned = c_[Xbinned, Xbinned*X]
    if add_bias:
        Xbinned = c_[ones(len(X)), Xbinned]
    return(Xbinned)


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
n_bins = 7
X = preprocessing(x, bins=n_bins, add_interaction=True, add_bias=False)
weights = normal_equasion(X,y)

Xtest = preprocessing(xtest, bins=preprocessing.bins, add_interaction=True, add_bias=False)
ypred = [predict(weights, x) for x in Xtest]

"""VISUALIZE"""
plt.plot(x,y, '.')
start, stop, num = preprocessing.bins
[plt.axvline(x, color='gray', linestyle=':') for x in np.linspace(start, stop, num+1)]
plt.gca().set_aspect("equal")
plt.plot(xtest, ypred)
plt.show()
