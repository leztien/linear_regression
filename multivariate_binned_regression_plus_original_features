"""
multivariate binned regression with the original (continuous) features (visualization in 3D)
"""

import numpy as np
from itertools import product


"""MAKE DATA"""
def data():
    X = np.array(tuple(product(range(10),range(10))), dtype=np.float16)
    X += np.random.normal(loc=0, scale=0.1, size=X.shape)   # add some error/displacement to the x-points
    X += X.min(axis=0).__abs__()    # shift all xy-plane-points into the 1st quadrant (just for aesthetics)
    y = np.sin(X[:,0]) + np.cos(X[:,1]) + X[:,0]*X[:,1]/100
    y = y + np.random.normal(loc=0, scale=y.std()/6)  # add some error to the target
    return(X,y)
X,y = data()
n_bins = 5

"""MAKE TEST DATA"""
r = np.linspace(X.min(), X.max(), 200)  # r =range
Xtest = np.array(tuple(product(r,r)))


"""MAKE A PREPOCESSING AND MODELING PIPELINE"""
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import LinearRegression

class Subset:  
    def __init__(self, columns=None):  # columns = [indeces of columns to subset]
        self.columns = columns or Ellipsis
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        from numpy import ndarray, matrix
        assert isinstance(X, (ndarray, matrix)),"X must be ndarray or matrix"
        X = X[:, self.columns]
        return X

pl1 = make_pipeline(StandardScaler(), KBinsDiscretizer(n_bins=n_bins, encode='onehot', strategy='uniform'))
pl2 = make_pipeline(Subset(),)   # subsets all features (both)
fu = make_union(pl1, pl2)
md = LinearRegression(fit_intercept=False)    # md.coef_.size   will be 12  (5+5+2) 
pl = make_pipeline(fu, md)
pl.fit(X,y)      


"""PREDICT"""
ypred = pl.predict(Xtest)


"""VISUALIZE"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
sp = fig.add_subplot(111, projection="3d")

sp.plot(*X.T, y, '.', alpha=1, color='b')

n = int(np.sqrt(len(Xtest)));  assert n%1==0
XX,YY,ZZ = (nd.reshape(n,n) for nd in (*Xtest.T, ypred))
sp.plot_surface(XX,YY,ZZ, cmap=plt.cm.viridis_r)
plt.show()
