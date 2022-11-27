import numpy as np
from numba import njit

# @njit
# def _bolsnumba(X:np.ndarray, y:np.ndarray, prior:float, sigma_delta:np.ndarray)




@njit
def bglsnp(X:np.ndarray,y:np.ndarray,prior:np.ndarray,sigma_delta:np.ndarray)->tuple:
    """
        Does a simple regression with a prior for beta allowing "wiggle" room through the variance in $\Sigma_{delta}$
        Does not verify any dimension. Do that on your own.
    """
    # get shapes
    N = y.shape[0]
    M = prior.shape[0]
    y = y.reshape((N,))
    prior = prior.reshape((M,))

    # stack matrices
    Xfull = np.vstack((X, np.eye(M)))
    yfull = np.hstack((y,prior))

    # regress and get ssr
    ssr = np.linalg.lstsq(X,y,rcond=-1)[1]
    ssr = ssr/(N-M)
    # estimate Sigma_epsilon
    sigma = ssr * np.ones((N+M,))

    # stack sigma
    sigma[-M:,] = sigma_delta#.reshape((M,))
    
    # calculate weights
    weights = 1/np.sqrt(sigma)

    # weighted regression
    for i in range(N+M):
        Xfull[i,:] = Xfull[i,:]*weights[i]
    # Xw = np.multiply(Xfull,weights[:,None])

    betahat = np.linalg.lstsq(Xfull, yfull * weights, rcond=-1.0)[0]

    # predict
    yhat = np.zeros((N,))

    for n in range(0, N):
        for i in range(0, M):
                yhat[n] += X[n, i] * betahat[i]
    # pause=1

    return betahat, yhat