

"""
batch gradient descent with early stopping / softmax classification 
"""

import numpy as np


def get_random_state(random_state):
    from random import randint
    from numpy.random import RandomState, mtrand
    rs = random_state if(random_state is not None) else randint(1,99999)
    rs = rs if isinstance(rs, mtrand.RandomState) else RandomState(int(rs))
    return(rs)


def make_blob(m, n, random_state=None, return_radius=False):
    rs = get_random_state(random_state)
    sigmas = rs.chisquare(df=3, size=n)**0.5 if bool(rs.randint(0,2)) else rs.uniform(low=0.1, high=3, size=n)
    blob = rs.normal(loc=0, scale=sigmas, size=(m,n))
    blob = blob - blob.mean(0)  # center the blob
    ix = (blob**2).sum(1).argmax()
    radius = np.sqrt((blob[ix]**2).sum())
    return (blob,radius) if return_radius else blob


def make_rotation_matrix(n, n_rotations=None, max_angle=None, random_state=None):
    """rotation matrix in n-dimensions"""
    from itertools import combinations
    rs = get_random_state(random_state)
    if n_rotations==0: return np.eye(n)
    plane_combinations = tuple(combinations(range(n), r=2))  # 2d-plane combinations
    if n_rotations==0: return np.eye(n)

    if n_rotations != "all":  # randomly select wich planes to rotate on
        n_rotations = n_rotations or rs.randint(1, len(plane_combinations))
        n_rotations = int(n_rotations)
        assert 0 < n_rotations <= n*(n-1)/2,"err"
        nx = np.sort(rs.permutation(len(plane_combinations))[:n_rotations])
        assert len(nx)>=1 and max(nx)<len(plane_combinations),"err"
        plane_combinations = np.array(plane_combinations)[nx]

    #create lists of id-matreces, angles
    max_angle = max_angle or 180
    rotations = [np.eye(n) for _ in plane_combinations]
    angles = np.deg2rad(rs.randint(-max_angle, max_angle, size=len(plane_combinations)))
    trigs = [np.array([np.cos(θ), -np.sin(θ), np.sin(θ), np.cos(θ)]).reshape(2,2)
                for θ in angles]    # the four trigonometric values

    for i,(ix,trig) in enumerate(zip(plane_combinations, trigs)):
        nx = np.ix_(ix,ix)
        rotations[i][nx] = trig

    from functools import reduce
    T = reduce(np.matmul, rotations[::-1])
    return T


def make_data_for_classification(m:'total number of data-points',
                                 n:'number of dimensions/features',
                                 k:'number of classes if int, proportions if float or sequence of floats',
                                 blobs_density:'ratio denoting the relative vicinity of the blobs to the central blob' = None,
                                 random_state=None):
    """make n-dimensional linearly seperable data with k classes.
    The structure of the data and how it fills the space can be visualized in 3d (if n==3)
    (the data is unnormalized/unstandardized and is located in the positive hyper-quadrant)"""

    if n < 3: raise ValueError("n must be 3 or higher")

    #calculate numbers of data-points in each blob/class
    from numbers import Integral, Real
    from collections.abc import Sequence
    n_datapoints = []
    if isinstance(k, Integral):
        if k<2: raise ValueError("k must be greater than 1")
        quotient,remainder = divmod(m,k)
        n_datapoints = [quotient,]*k
        n_datapoints[-1] += remainder
    elif isinstance(k, Real) and (0 < k < 1):
        i = int(round(float(k)*m))
        n_datapoints.append(max(i,1) if k<0.5 else min(i, m-1))
        n_datapoints.append(m - n_datapoints[0])
    elif isinstance(k, Sequence):
        from decimal import Decimal
        if sum(Decimal(str(p)) for p in k).__float__() != 1.0:
            k = [float(p)/sum(k) for p in k ]
        assert round(float(sum(k)),1) == 1.0,"all percentages must sum to 1"
        n_datapoints = [round(float(p)*m) for p in k]
    else: raise TypeError("bad argument k")

    # making sure there are no zeros in n_datapoints
    while min(n_datapoints)<1:
        ix_min = n_datapoints.index(min(n_datapoints))
        ix_max = n_datapoints.index(max(n_datapoints))
        n_datapoints[ix_min] += 1
        n_datapoints[ix_max] -= 1
    assert (sum(n_datapoints) == m) and (min(n_datapoints) >= 0), "bad sum or individual values"
    k = len(n_datapoints)

    #random state
    rs = get_random_state(random_state)

    # make k blobs with respective radii
    blobs_w_radii = [make_blob(m, n, random_state=rs, return_radius=True) for m in n_datapoints]
    blobs = [t[0] for t in blobs_w_radii]
    radii = [t[1] for t in blobs_w_radii]

    #make k rortation matreces and rotate each blob with the respective rotation matrix
    rotations = [make_rotation_matrix(n, max_angle=45, random_state=rs) for _ in range(k)]
    blobs = [(R@M.T).T for R,M in zip(rotations,blobs)]

    #make k-1 unit-vectors pointing in random directions in the n-dimensional space
    transformations_for_unit_vector = [make_rotation_matrix(n, n_rotations='all', random_state=rs) for _ in range(k)]
    v = np.array([1, *[0,]*(n-1)]).reshape(-1,1)  # i-hat basis-vector
    vectors = [T@v for T in transformations_for_unit_vector]  # vector[0] will be ignored

    #shift the k-1 blobs in the direction of the respective random unit vector
    blobs_density = blobs_density if not(blobs_density is None) else 1  # density of blobs (0, 1)
    for i in range(1,k):
        vector = vectors[i] * (radii[0] + radii[i]) * blobs_density
        blobs[i] = blobs[i] + vector.flatten()

    #make the target vector, concatinate the data
    y = sum(([label,]*len(blob) for blob,label in zip(blobs, range(k))), [])
    mx = np.concatenate(blobs, axis=0)
    mx = mx + mx.min(axis=0).__abs__()

    X,y = mx, np.array(y, dtype='uint8')
    assert len(X)==len(y),"err3"
    return(X,y)

  
#===========================================================================    


X,y = make_data_for_classification(m=100, n=50, k=4, 
                blobs_density=0.3, random_state=None)


#sklearn model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
md = LogisticRegression(fit_intercept=True, solver='sag', multi_class='multinomial', C=100, penalty='l2', max_iter=9999).fit(X,y)
accuracy = md.score(X,y)
print("accuracy on the entire set =", accuracy)
W = np.concatenate([md.intercept_[None,:], md.coef_.T], axis=0)
ypred = np.matmul(np.c_[np.ones(shape=(len(X),1)), X], W).argmax(axis=1)

accuracies = cross_val_score(md, X,y, cv=5, scoring="accuracy")
print("cross-validation =", accuracies.mean().round(3))
    






import numpy as np
class BatchGradientDescentEarlyStopping:
    def __init__(self, initial_learning_rate = 0.1,
                       l2_penalty = 0.1,
                       validation_set_size=0.25,
                       max_iter=200,
                       random_state=None,
                       verbose=False):
        self.η = initial_learning_rate
        self.λ = l2_penalty
        self.validation_set_size = validation_set_size
        self.max_iter = max_iter
        self.training_cost = []
        self.validation_cost = []
        self.model_parameters = []
        self.best_parameters = None
        self.verbose = verbose
        self.random_state = int(random_state) if(random_state is not None) else np.random.randint(0,99999) 
    
    def fit(self, X,y):
        #split into train and validation sets
        from sklearn.model_selection import train_test_split
        Xtrain,Xval,ytrain,yval = train_test_split(X,y, test_size=self.validation_set_size, random_state=self.random_state)
        (m,n),k = X.shape, len(frozenset(y))
        
        X_w_bias = np.c_[np.ones(len(Xtrain))[:,None], Xtrain] # X_train_with_bias_feature
        Ytrue = np.zeros(shape=(Xtrain.shape[0], k))
        Ytrue[range(len(Xtrain)), ytrain] = 1
        Θ = np.ones(shape=(n+1, k))  # current parameters
        
        #the loop
        for epoch in range(self.max_iter):
            Z = np.matmul(X_w_bias, Θ)
            E = np.exp(Z)
            Ypred = E / E.sum(axis=1)[:,None]
            
            #penalty term
            Θtemp = Θ.copy()
            Θtemp[0,:] = 0
            P = self.λ * Θtemp
            
            #gradient matrix
            G = (X_w_bias.T   @  (Ypred - Ytrue)) / m + P
            Θ = Θ - self.η * G
            
            #get cost
            self.model_parameters.append(Θ)
            
            cost = self.cross_entropy(Xtrain, ytrain, Θ)
            self.training_cost.append(cost)
            
            cost = self.cross_entropy(Xval, yval, Θ)
            self.validation_cost.append(cost)
        
            #adjust learning rate
            self.η = self._adjust_learning_rate()
            
            #check for early stopping
            ix = self._check_early_stopping()
            if ix:
                self.best_parameters = self.model_parameters[ix]
                if self.verbose:
                    print("\nSUMMARY:")
                    print("breaking out after epoch", epoch)
                    print("best model parameters are at index", ix)
                    print("current learning rate =", round(self.η, 3))
                    accuracy = self.accuracy(Xval, yval)
                    print("gradient descent accuracy on validation set =", accuracy)
                    import matplotlib.pyplot as plt
                    plt.plot(self.training_cost, color='red')
                    plt.plot(self.validation_cost, color='blue')
                    plt.ylim(0, 0.9)
                break
        else: 
            ix = self.validation_cost.index(min(self.validation_cost))
            self.best_parameters = self.model_parameters[ix]
            print("increase max_iter, current =", self.max_iter)
        return self
        

    def precict_probabilities(self, X):
        if self.best_parameters is None: raise TypeError("you must fit the model first")
        X_w_bias = np.c_[np.ones(len(X))[:,None], X]
        Z = np.matmul(X_w_bias, self.best_parameters)
        E = np.exp(Z)
        Ypred = E / E.sum(axis=1)[:,None]
        return Ypred
    
    def predict(self, X):
        if self.best_parameters is None: raise TypeError("you must fit the model first")
        X_w_bias = np.c_[np.ones(len(X))[:,None], X]
        Z = np.matmul(X_w_bias, self.best_parameters)
        ypred = Z.argmax(axis=1)
        return ypred
        
    
    def accuracy(self, X,y):
        if self.best_parameters is None: raise TypeError("you must fit the model first")
        ytrue,ypred = y, self.predict(X)
        return (ytrue==ypred).mean()
    
    
    @staticmethod
    def cross_entropy(X,y, model_parameters, add_bias_feature=True):
        X_w_bias = np.c_[np.ones(len(X))[:,None], X] if add_bias_feature else X
        Z = np.matmul(X_w_bias, model_parameters)
        E = np.exp(Z)
        Ypred = E / E.sum(axis=1)[:,None]
        Ytrue = np.zeros(shape=(X.shape[0], len(frozenset(y))))
        Ytrue[range(len(X)), y] = 1
        cross_entropy_cost = (np.log(Ypred) * Ytrue).sum() / -len(y)
        return(cross_entropy_cost)
    
    @staticmethod
    def root_mean_squared_error(X,y, model_parameters):
        return NotImplemented
    
 
    def _adjust_learning_rate(self):
        #settings
        learning_rate_decreasing_ratio = 0.9
        n = 5  # consider the last n values
        
        #from the self
        current_learning_rate = self.η
        cost_sequence = self.training_cost
        
        if len(cost_sequence) < n: return current_learning_rate
        
        #check for oscilation (jumping around of the gradient) >>> decrease learning rate
        a = np.array(cost_sequence)[-n:]
        signs = np.sign(a[1:] - a[:-1])
        counts = [tuple(signs).count(n) for n in (1,-1)]
        f = min(count/(sum(counts)+1e-9) for count in counts)
        b = f >= 0.2   # b == oscilating
        if b: current_learning_rate = current_learning_rate * learning_rate_decreasing_ratio
        
        #check for too low learning rate >>> increase learning rate
        a = np.array(self.validation_cost)[-n:]
        signs = np.sign(a[1:] - a[:-1])
        b = -sum(signs) == (n-1)   # b = True >>> going down continuously
        if b and len(self.validation_cost)>n:
            current_learning_rate = current_learning_rate * 1.05
        return current_learning_rate


    def _check_early_stopping(self):
        n = 5 # consider the last n values
        validation_cost_sequence = self.validation_cost
        if len(validation_cost_sequence) <= n: return None
        a = np.array(validation_cost_sequence)[-n:]
        signs = np.sign(a[1:] - a[:-1])
        b = sum(signs) == (n-1)   # b = True >>> going up
        if b:
            ix = validation_cost_sequence.index(min(validation_cost_sequence))  # index of the best model 
            return ix
        return b
#======= end of BatchGradientDescentEarlyStopping =============================================


md = BatchGradientDescentEarlyStopping(random_state=None, verbose=True)
md.fit(X,y)

probs = md.precict_probabilities(X)
y = md.predict(X)
acuracy = md.accuracy(X,y)
print("accuracy on the whole data set =", accuracy)

