"""
multivariate binned regression (visualization in 3D)
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


"""MAKE TEST DATA"""
r = np.linspace(X.min(), X.max(), 200)  # r =range
Xtest = np.array(tuple(product(r,r)))


"""MAKE A PREPOCESSING AND MODELING PIPELINE"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import LinearRegression
n_bins = 10
pl = make_pipeline(StandardScaler(),   # the parameters will be saved in: pl.steps[0][1].mean_, pl.steps[0][1].scale_ 
                   KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='uniform'),  # pl.steps[1][1].bin_edges_
                   LinearRegression(fit_intercept=False))  # pl.steps[-1][-1].coef_.size == 20
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
sp.plot_surface(XX,YY,ZZ, cmap=plt.cm.RdBu_r, alpha=1)
plt.show()
