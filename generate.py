import numpy as np
from scipy.stats import multivariate_normal

def GM_sample(N, K, d):
    
    """ 
    Generates a gaussian mixture sample given the lists of parameters: 
      N: number of samples 
      K: number of Gaussian vectors
      d: size of each observation
    """
    
    Pis0 = np.random.uniform(low=1/(2*K),high=2/K,size=K)
    Pis0 /= np.sum(Pis0)
    
    NPis0 = (N*Pis0).astype(int)
    Leftover = N - np.sum(NPis0)
    for k in range(Leftover):
        NPis0[k] += 1
    
    Pis0 = NPis0/N

    Mus0 = np.random.randint(-3*K,3*K,size=(K,d))
    Sigmas0 = np.random.rand(K,d,d)
    Sigmas0 = np.array([(s+s.T)/d + np.eye(d) for s in Sigmas0])
    
    X = []
    y = np.array([])
    
    for cluster, N_k, mu_k, sigma_k in zip(np.arange(K),NPis0, Mus0, Sigmas0):
        X.extend(multivariate_normal.rvs(mean=mu_k,cov=sigma_k,size=N_k))
        y = np.hstack([y,np.ones(N_k)*cluster])
        
    X = np.vstack(X)
    y = np.array(y).astype(int)
    params = [Pis0, Mus0, Sigmas0]
    
    return X, y, params