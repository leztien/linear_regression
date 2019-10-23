
"""
demo of binary logistic classification (batch gradient descent)
(no cross-validation, no stochastic data selection)
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


def binary_logistic_classification(X,y) -> "predicting function":
    from numpy import array, zeros, exp, log as ln
    #hyperparameters:
    λ = 0.001
    η = 0.01
    tol = 0.1
    max_iter = 10000
    
    m,n = X.shape
    μ,σ = X.mean(0), X.std(0, ddof=0)
    X = (X-μ)/σ  # standerdize
    y = y.reshape(-1,1)
    θ = zeros(shape=n).reshape(-1,1)
    
    #THE LOOP
    for epoch in range(max_iter):
        z = X @ θ 
        h = 1 / (1 + exp(-z))
        
        #cost
        J = -(ln(h)*y + ln(1-h)*(1-y)).sum() / m
        
        #check convergence
        if J < tol:
            print("breaking after loop", epoch)
            break
        
        #derivative
        θ_ = array([0, *θ[1:]]).reshape(-1,1)  # θ[0] is not regularized
        g = (1/m) * X.T @ (h-y) + λ/m*θ_
        
        #update
        θ = θ - η*g
    else: print("increase the number of max iterations")
    
    #accuracy
    ypred = (1 / (1 + exp(-X @ θ))) >= 0.5
    accuracy = (ypred==y).mean()
    print("cost =", J.round(3), "accuracy =", accuracy.round(3), end="\n\n")
    
    #return predicting function
    predict = lambda X : ((1 / (1 + exp(-(X-μ)/σ @ θ))) >= 0.5).astype("uint8").ravel()
    return(predict)

#######################################################################################

#data
path = r"https://raw.githubusercontent.com/leztien/synthetic_datasets/master/two_egg_carton_separable_blobs.py"
module = load_from_github(path)
X,y = module.two_egg_carton_separable_blobs(m=1000, n=30)

#split
from numpy.random import permutation
nx = permutation(len(X))
test = int(0.25 * len(X))
Xtrain,ytrain = (nd[:-test] for nd in (X,y))
Xtest,ytest = (nd[test:] for nd in (X,y))

#model and predict
func = binary_logistic_classification(Xtrain, ytrain)
ypred = func(Xtest)
accuracy = (ypred==ytest).mean()
print("test accuracy =", accuracy.round(3))
