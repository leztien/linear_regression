"""
demo of two algorithms for logistic classification:
one-vs-rest  vs  softmax
"""

def load_from_github(url):
    from urllib.request import urlopen
    from os import remove
    
    obj = urlopen(url)
    assert obj.getcode()==200,"unable to open"

    s = str(obj.read(), encoding="utf-8")
    NAME = "_temp.py"
    with open(NAME, mode='wt', encoding='utf-8') as fh: fh.write(s)
    module = __import__(NAME[:-3])
    remove(NAME)
    return module



def _predict(X, predict_probabilities=False, Weights=None, mu=None, sigma=None):   
    from numpy import c_
    X = (X-mu)/sigma
    x0 = [1]*len(X)
    X = c_[x0, X]
    Z = X @ Weights

    if not predict_probabilities: return Z.argmax(axis=1)
    
    from numpy import exp
    O = exp(Z)         # odds matrix
    P = O / O.sum(1, keepdims=True)   # probabilities matrix
    return(P)



def binary_logistic_classification(X,y):
    """returns the weights vector"""
    from numpy import array, ones, zeros, c_, exp, log as ln
    #hyperparameters:
    λ = 0.0001
    η = 0.05
    max_iter = 1000
    tol = 0.1
    #initialize:
    m,n = X.shape
    x0 = ones(m)
    X = c_[x0, X]
    y = y.reshape(-1,1)
    θ = zeros(n+1).reshape(-1,1)
    
    for epoch in range(max_iter):
        z = X @ θ
        p = 1 / (1 + exp(-z))
        ε = p - y
        
        θ_ = array([0, *θ[1:]]).reshape(-1,1)   # no regularization for the bias
        g = (X.T @ ε + λ*θ_) / m
        θ = θ - η*g
        
        #check convergence
        J = -(ln(p)*y + ln(1-p)*(1-y)).sum() / m
        if J < tol: break
    else: print("increase the number of max iterations")
    return(θ)



def one_vs_rest(X,y):  # this is a factory function in effect
    from numpy import concatenate, equal
    #scale X:
    μ,σ = X.mean(0), X.std(0, ddof=0)
    X = (X-μ)/σ
    
    classes = sorted(set(y))
    masks = [equal(y,k).astype('uint8') for k in classes]
    Θ = concatenate([binary_logistic_classification(X,y) for y in masks], axis=1)
    
    from functools import partial
    global _predict
    func = partial(_predict, Weights=Θ, mu=μ, sigma=σ)
    return(func)  # the factory returns a closure-function


def softmax(X,y):
    #hyperparameters:
    λ = 0.0001
    η = 0.1
    max_iter = 30000
    tol = 0.1
    #scale, add bias-feature
    from numpy import c_, zeros, matmul, exp, log as ln
    m,n = X.shape
    μ,σ = X.mean(0), X.std(0, ddof=0)
    X = (X-μ)/σ
    X = (X-μ)/σ
    x0 = [1]*len(X)
    X = c_[x0, X]
    #one-hot Y
    classes = sorted(set(y))
    k = len(classes)
    Y = zeros(shape=(m,k), dtype="uint8")
    Y[range(m),y] = 1
    
    W = zeros(shape=(n+1,k), dtype=float)
    
    for epoch in range(max_iter):
        Z = matmul(X,W)
        O = exp(Z)
        P = O / O.sum(axis=1, keepdims=True)
        E = P-Y
        G = (matmul(X.T,E) + λ*W) / m
        W = W - η*G
        J = -(ln(P)*Y).sum()/m
        if J < tol:break
    else:print("increase the number of iterations")

    from functools import partial
    global _predict
    func = partial(_predict, Weights=W, mu=μ, sigma=σ)
    return(func)  # the factory returns a closure-function


#######################################################################################

#data
path = r"https://raw.githubusercontent.com/leztien/synthetic_datasets/master/make_data_for_classification.py"
module = load_from_github(path)
X,y = module.make_data_for_classification(m=1000, n=5, k=3, blobs_density=0.1)

#split
from numpy.random import permutation
nx = permutation(len(X))
test = int(0.25 * len(X))
Xtrain,ytrain = (nd[:-test] for nd in (X,y))
Xtest,ytest = (nd[test:] for nd in (X,y))


#one-vs-rest
predict = one_vs_rest(Xtrain, ytrain)
ypred = predict(Xtest)
P = predict(Xtest, predict_probabilities=True)
ypred = P.argmax(1)
accuracy = (ytest==ypred).mean()
print("one-vs-rest test accuracy =", accuracy.round(3))

#softmax
predict = softmax(Xtrain, ytrain)
ypred = predict(Xtest)
P = predict(Xtest, predict_probabilities=True)
ypred = P.argmax(1)
accuracy = (ytest==ypred).mean()
print("softmax test accuracy =", accuracy.round(3))
