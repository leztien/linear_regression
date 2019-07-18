"""
binned regression in 2D
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import inv
from sklearn.preprocessing import OneHotEncoder


def preprocess(X, bins):
    l = []
    for x in X.T:
        breaks = np.linspace(min(x)-0.5, max(x)+0.5, bins+1)
        nx = np.digitize(x, bins=breaks)
        x = OneHotEncoder(sparse=False, categories='auto').fit_transform(nx.reshape(-1,1))
        l.append(x)
    X = np.hstack(l)
    return X


"""MAKE DATA"""
from itertools import product
X = np.array(tuple(product(range(10),range(10))))
y = np.sin(X[:,0]) + np.cos(X[:,1]) + X[:,0]*X[:,1]/100

r = np.linspace(X.min(), X.max(), 200)  # r =range
Xtest = np.array(tuple(product(r,r)))



"""VISUALIZE"""
n = int(np.sqrt(len(X)));  assert n%1==0
XX,YY,ZZ = (nd.reshape(n,n) for nd in (*X.T, y))

fig = plt.figure()
sp = fig.add_subplot(111, projection="3d")
#sp.plot_surface(XX,YY,ZZ, cmap="jet", alpha=0.3)
sp.plot(*X.T, y, '.', alpha=0.7)


"""MODEL, TEST"""
n_bins = 10

X = preprocess(X, bins=n_bins)
weights = inv(X.T @ X) @ (X.T@y)   # X one hot encoded

Xtest_oh = preprocess(Xtest, bins=n_bins)
ypred = np.array([weights.dot(x) for x in Xtest_oh])

n = int(np.sqrt(len(Xtest)));  assert n%1==0
XX,YY,ZZ = (nd.reshape(n,n) for nd in (*Xtest.T, ypred))
sp.plot_wireframe(XX,YY,ZZ, color="gray", alpha=0.7, rstride=2, cstride=2)

plt.show()
