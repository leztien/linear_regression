"""
3-dimensional logistic regression (n=2), visualized in 2D (in the xy-plane)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def data(m=1000, random_state=None, verbose=False):
    """makes data (n=2) with two classes for logistic regression analysis"""
    n = 2
    np.random.seed(random_state if not(random_state is None) else np.random.randint(0,9999))
    X = np.random.normal(loc=0, scale=[3,2], size=(m,n))
    
    #rotate the data
    angle = 30
    θ = np.deg2rad(angle)
    R = [[np.cos(θ), -np.sin(θ)],
         [np.sin(θ), np.cos(θ)]]
    X = np.matmul(R, X.T).T
    X = X + X.min(0).__abs__() + 0.1  # move the data into the positive quadrant
    
    """the decision boundry line is at angle+90 degrees"""
    #get the slope of the decision boundry line
    angle = angle + 90
    θ = np.deg2rad(angle)
    rise = np.sin(θ)
    run = np.cos(θ)
    slope = rise/run

    #calculate the intercept
    xbar, ybar = X.mean(0)
    bias = ybar - xbar*slope
    f = lambda x : x*slope+bias
    
    #determine the true 0 and 1 cases
    y = np.array([f(x)-y <= 0 for x,y in X], dtype='uint8')
    
    #add noise
    factor = 0.4
    error = np.random.normal(loc=0, scale=X.std()*factor, size=X.shape)
    X += error
    
    #mround x1 feature
    X[:,0] = X[:,0]//0.5*0.5
    
    #info
    if verbose:
        print("\nINFO:")
        covariance_matrix = np.cov(X.T)
        covariance = np.cov(X.T)[0,1]
        r = np.corrcoef(X[:,0], X[:,1])[0,1]   # same as..
        r = covariance / np.sqrt(covariance_matrix[0,0] * covariance_matrix[1,1])
        print("cov(x1,x2) =", covariance.round(2), "\tcorrelation coeficient =", r.round(3))
        print("slope of the decsion boundry line", slope.round(1))
        ypred = np.array([f(x)-y <= 0 for x,y in X], dtype='uint8')
        print("hypothetically achievable accuracy =", (y==ypred).mean().round(3), end="\n"*2)
    return(X,y)

#==========================================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X,y = data(m=200, random_state=None, verbose=True)
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.25)

#build a model-pipeline
sc = StandardScaler()
λ = 1e-5
md = LogisticRegression(fit_intercept=True, solver='liblinear', C=1/λ, penalty='l2')
pl = make_pipeline(sc, md)

pl.fit(Xtrain, ytrain)
print("test accuracy =", pl.score(Xtest, ytest))

θ = np.array([*pl.steps[-1][-1].intercept_, *pl.steps[-1][-1].coef_[0]])
print("θ =", str(θ[:,None].round(2)).replace('\n', '\n    '))


#draw the decision boundry line
f = lambda x : -θ[0]/θ[2] + -θ[1]/θ[2]*x
Xsc = sc.transform(X)
x1,x2 = Xsc[:,0].min(), Xsc[:,0].max()
y1,y2 = f(x1), f(x2)

cmap = ListedColormap(['blue','green'])
mask = y==0
plt.scatter(*Xsc[mask].T, s=30, edgecolor="k", color=cmap.colors[0], label="false")
plt.scatter(*Xsc[~mask].T, s=30, edgecolor="k", color=cmap.colors[1], label="true")
plt.plot([x1,x2], [y1,y2])
#plt.axis("equal")
plt.ylim(Xsc[:,1].min()-0.5, Xsc[:,1].max()+0.5)
t = plt.axis()

#contourf
r1,r2 = t[:2], t[2:]
r1, r2 = (np.linspace(start,end,200) for start,end in (r1,r2))
XX,YY = np.meshgrid(r1,r2)
ZZ = md.predict(np.c_[XX.ravel(),YY.ravel()]).reshape(XX.shape)
plt.contourf(XX,YY,ZZ, cmap=cmap, alpha=0.4, zorder=-3)
plt.legend()
