import BOLS
import numpy as np
import cupy as cp
import time
import gc
from cupyx.profiler import benchmark

def main():

    N=200
    M=4
    x = np.random.rand(N,M)
    prior = np.random.rand(M,)
    y = np.matmul(x,prior) + 0.02*np.random.rand(N,)
    # prior = np.random.rand(4,)
    sd = np.random.rand(M,)
    xcp = cp.array(x.copy())
    ycp = cp.array(y.copy())
    priorcp = cp.array(prior)
    sdcp = cp.array(sd)
    betahat, yhat = BOLS.bayesian_ols_numba.bglsnp(x,y, prior, sd)
    betaha2, yha2 = BOLS.bayesian_ols_cupy.bglscp(cp.array(xcp),cp.array(ycp), cp.array(prior), cp.array(sd))

    print(prior)
    print(betahat)
    print(betaha2)

    gc.disable()
    N=20
    # start = time.perf_counter_ns()
    # for _ in range(N):
    #     betahat, yhat = BOLS.bayesian_ols_numba.bglsnp(x,y, prior, sd)
    # print(f"numba took {(time.perf_counter_ns()-start)/1000} micro-seconds for {N} repeats")
    # start = time.perf_counter_ns()
    # for _ in range(N):
    #     betaha2, yha2 = BOLS.bayesian_ols_cupy.bglscp(xcp,ycp, priorcp, sdcp)
    # print(f"cupy took {time.perf_counter_ns()-start} ns for {N} repeats")

    print(benchmark(BOLS.bayesian_ols_numba.bglsnp, (x,y, prior, sd), n_repeat=N))
    print(benchmark(BOLS.bayesian_ols_cupy.bglscp, (xcp,ycp, priorcp, sdcp), n_repeat=N))

    gc.enable()

    pause=1


       

#### __name__ MAIN()
if __name__ == '__main__':
    # SET = int(sys.argv[1])
    main()


