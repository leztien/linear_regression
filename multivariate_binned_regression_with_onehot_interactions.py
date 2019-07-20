"""
multivariate binned regression with one-hot interactions (visualization in 3D)
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
n_bins = 4


"""MAKE TEST DATA"""
r = np.linspace(X.min(), X.max(), 200)  # r =range
Xtest = np.array(tuple(product(r,r)))


"""MAKE A PREPOCESSING AND MODELING PIPELINE"""
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import LinearRegression


class OneHotInteraction:
    def __init__(self, n_bins):
        self.n_bins = n_bins
    def fit(self, X, y=None):return(self)
    def transform(self, X):
        from sklearn.preprocessing import KBinsDiscretizer
        from numpy import hstack
        Xonehots, Xinteractions = list(),list()
        for feature in X.T:
            Xoh = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense', strategy='uniform').fit_transform(feature.reshape(-1,1))
            Xonehots.append(Xoh)
            Xinteractions.append(Xoh * feature[:,None])
        X_onehots_interactions = hstack([*Xonehots, *Xinteractions])
        return X_onehots_interactions
            

tr = OneHotInteraction(n_bins=n_bins)
md = LinearRegression(fit_intercept=False)
pl = make_pipeline(tr, md)
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
