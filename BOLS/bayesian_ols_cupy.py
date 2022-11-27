import cupy as cp
# from numba import njit

# @njit
# def _bolsnumba(X:cp.ndarray, y:cp.ndarray, prior:float, sigma_delta:cp.ndarray)




# @njit
def bglscp(X:cp.ndarray,y:cp.ndarray,prior:cp.ndarray,sigma_delta:cp.ndarray)->tuple:
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
    Xfull = cp.vstack((X, cp.eye(M)))
    yfull = cp.hstack((y,prior))

    # regress and get ssr
    ssr = cp.linalg.lstsq(X,y,rcond=-1)[1]
    ssr = ssr/(N-M)
    # estimate Sigma_epsilon
    sigma = ssr * cp.ones((N+M,))

    # stack sigma
    sigma[-M:,] = sigma_delta#.reshape((M,))
    
    # calculate weights
    weights = 1/cp.sqrt(sigma)

    # weighted regression
    # for i in range(N+M):
    #     Xfull[i,:] = Xfull[i,:]*weights[i]
    Xw = cp.multiply(Xfull,weights[:,None])

    betahat = cp.linalg.lstsq(Xw, yfull * weights, rcond=-1.0)[0]

    # predict
    # yhat = cp.zeros((N,))

    # for n in range(0, N):
    #     for i in range(0, M):
    #             yhat[n] += X[n, i] * betahat[i]
    # pause=1
    yhat = cp.matmul(X,betahat)

    return betahat, yhat