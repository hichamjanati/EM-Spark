import numpy as np
from pyspark import SparkContext
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
from matplotlib import pyplot as plt
from sklearn.exceptions import NotFittedError

class EM_noSpark(object):
    def __init__(self):
        self.fitted = False
        self.estimated_params = {}
        self.ll_all_ = []
        self.ll_final = None
    
    @staticmethod
    def log_pi_pdf(data, pis, mus, sigmas, n ,k):
        return np.log(pis[k]) + multivariate_normal.logpdf(data[n],mean=mus[k],cov=sigmas[k],allow_singular=True)

    def init_params(self,data):
        self.pis = np.ones(self.K)/self.K

        mini = np.min(data,axis=0)
        maxi = np.max(data,axis=0)

        self.mus = np.array([np.random.normal(loc=(maxi-mini)/2) for _ in range(self.K)])
        self.sigmas = np.vstack([np.identity(self.d) for _ in range(self.K)]).reshape(self.K,self.d,self.d)

    def all_gamma(self,data):
        L = np.array([[self.log_pi_pdf(data, self.pis, self.mus, self.sigmas, n ,k) for k in range(self.K)] for n in range(self.N)])
        log_sum = logsumexp(L,axis=1).reshape(self.N,1)
        return np.exp(L - log_sum)

    def log_likelihood(self, data):
        L = np.array([[self.log_pi_pdf(data, self.pis, self.mus, self.sigmas, n ,k) for k in range(self.K)] for n in range(self.N)])
        return np.sum(logsumexp(L,axis=1))

    def clustering(self, data):
        if not self.fitted:
            raise NotFittedError('EM must be fitted before performing clustering')
            
        L = np.array([[self.log_pi_pdf(data, self.pis, self.mus, self.sigmas, n ,k) for k in range(self.K)] for n in range(self.N)])
        return np.argmax(L, axis=1)

    def fit(self, data, n_clusters, max_iter=100, criterion = 'LL', tol=1e-3, verbose=True):

        # Global variables
        self.N,self.d = data.shape
        self.K = n_clusters

        # Initialization of parameters
        self.init_params(data)

        # Run E step and M step until criterion verified
        iteration = 0
        if criterion != "LL":
            tol = -1

        diff_lll = tol + 1
        list_lll = []
        while (iteration < max_iter and diff_lll > tol):

            # E step
            gammas = self.all_gamma(data)

            # M step
            NPis_t = np.sum(gammas,axis=0)
            NPis_t = np.clip(NPis_t,1e-10,np.max(NPis_t))
            self.pis = NPis_t / self.N

            NMus_t = np.sum(np.array([np.array([data[n]*gammas[n][k] for k in range(self.K)]) for n in range(self.N)]),axis=0)
            self.mus = NMus_t / NPis_t.reshape(self.K,1)

            NSigmas_t = np.sum(np.array([np.array([gammas[n][k]*np.outer(data[n]-self.mus[k],data[n]-self.mus[k]) 
                                                    for k in range(self.K)]) for n in range(self.N)]), axis=0)
            self.sigmas = NSigmas_t / NPis_t.reshape(self.K,1,1)

            # Log-likelihood
            if criterion == "LL":
                lll = self.log_likelihood(data)
                if len(list_lll) > 0:
                    diff_lll = abs(lll-list_lll[-1])
                list_lll.append(lll)

            # Update iteration
            iteration += 1

            # Verbose
            if verbose:
                print("Iteration", iteration)
                if criterion == 'LL':
                    print("| log-likelihood =",list_lll[-1])
                    if diff_lll <= tol:
                        print("Converged")
        self.fitted = True

        # Estimated parameters
        self.estimated_params = {'pis': self.pis, 'mus':self.mus, 'sigmas':self.sigmas}
        return self
    
    def fit_predict(self, data, n_clusters, max_iter=100, criterion = 'LL', tol=1e-3, verbose=True):
        self.fit(data, n_clusters, max_iter, criterion, tol, verbose)
        
        return self.clustering(data)

class EM_Spark(object):
    
    """ GMM Expectation-Maximization object written in a mapreduce form. 
    ---------------------------------------------------------------------
    
    Example: 
    >> EM = EM_Spark()
    >> #if X is the (N,d) shaped array containing N  observations in d-dimensions of a GM:
    >> data = sc.parallelize(X)
    >> EM.fit(data, n_clusters)
    >> print(EM.estimated_params)
    >> #Show estimated clusters:
    >> EM.clustering(data) 
    
    >> #Or combine both:
    >> EM.fit_predict(data, n_clusters)
    """
    
    def __init__(self):
        self.fitted = False
        self.estimated_params = {}
        self.ll_all_ = []
        self.ll_final = None
        
        
    def fit(self, data, n_clusters, max_iter=100, criterion = 'LL', tol=1e-3, verbose=True ):
        # Parameters N,d,k
        self.N = data.count()
        self.d = data.first().size
        self.K = n_clusters

        data_list = data.map(lambda x: [x])

        # Initialization of parameters
        self.init_params(data_list)

        # Run E step and M step until criterion satisfied
        iteration = 0
        
        # If loglikelihood is not used as a criterion: 
        if not criterion == 'LL':
            tol = -1
            
        diff_lll = tol + 1
        list_lll = []
        while (iteration < max_iter and diff_lll > tol):

            # E step
            rdd = self.E_step(data_list)

            # M step
            self.M_step(rdd)

            # Log-likelihood
            
            if criterion == 'LL':
                lll = self.log_likelihood(data_list)
                if len(list_lll) > 0:
                    diff_lll = abs(lll-list_lll[-1])
                list_lll.append(lll)
                self.ll_all_ = list_lll

            # Update iteration
            iteration += 1

            # Verbose
            if verbose:
                print("Iteration", iteration)
                if criterion == 'LL':
                    print("| log-likelihood =",list_lll[-1])
                    if diff_lll <= tol:
                        print("Converged")

        # Estimated parameters
        self.estimated_params = {'pis': self.pis, 'mus':self.mus, 'sigmas':self.sigmas}

        self.fitted = True
        
        return self
    
    def fit_predict(self, data, n_clusters, max_iter=100, criterion = 'LL', tol=1e-3, verbose=True):
        self.fit(data, n_clusters, max_iter, criterion, tol, verbose)
        data_list = data.map(lambda x: [x])
        return self.clustering(data_list)
    
    
    @staticmethod
    def log_pi_pdf(x, k, pis, mus, sigmas):
        return np.log(pis[k]) + multivariate_normal.logpdf(x[0],mean=mus[k],cov=sigmas[k],allow_singular=True)
    
    def map_Gamma(self,x):
        log_gammas = np.array([self.log_pi_pdf(x, k, self.pis, self.mus, self.sigmas) for k in range(self.K)])
        log_gammas = log_gammas - logsumexp(log_gammas)
        return [x[0], np.exp(log_gammas)]
    
    def reduce_NPis(self,x,y):
        return [0, [x[1][k]+y[1][k] for k in range(self.K)]]
    
    def map_Mus(self,x):
        return x + [[x[0]*x[1][k] for k in range(self.K)]]
    
    def reduce_Mus(self,x,y):
        return [0,0,[x[2][k] + y[2][k] for k in range(self.K)]]
    
    def map_Sigmas(self, x, mus):
        return x + [[x[1][k]*np.outer(x[0]-mus[k],x[0]-mus[k]) for k in range(self.K)]]
    
    def reduce_Sigmas(self, x,y):
        return [0,0,0,[x[3][k] + y[3][k] for k in range(self.K)]]
    
    def init_params(self, rdd):
        self.pis = np.ones(self.K)/self.K

        mini = np.array(rdd.reduce(lambda x,y: [np.array([min(x[0][i],y[0][i]) for i in range(self.d)])])[0])
        maxi = np.array(rdd.reduce(lambda x,y: [np.array([max(x[0][i],y[0][i]) for i in range(self.d)])])[0])

        self.mus = np.array([np.random.normal(loc=(mini+maxi)/2) for _ in range(self.K)])
        self.sigmas = np.vstack([np.identity(self.d) for i in range(self.K)]).reshape(self.K,self.d,self.d)

    def E_step(self, rdd):
        rdd_gammas = rdd.map(self.map_Gamma)
        return rdd_gammas

    def M_step(self,rdd):

        # NPis : reduce
        NPis_t = np.array(rdd.reduce(self.reduce_NPis)[1])
        NPis_t = np.clip(NPis_t,1e-10,np.max(NPis_t))
        self.pis = NPis_t / self.N

        # Mus : map 
        rdd_mus = rdd.map(self.map_Mus)

        # Mus : reduce
        Mus_t = rdd_mus.reduce(self.reduce_Mus)[2]
        self.mus = [Mus_t[k]/NPis_t[k] for k in range(self.K)]

        # Sigmas : map
        rdd_sigmas = rdd_mus.map(lambda x: self.map_Sigmas(x,mus=self.mus))

        # Sigmas : reduce
        Sigmas_t = rdd_sigmas.reduce(self.reduce_Sigmas)[3]
        self.sigmas = [Sigmas_t[k]/NPis_t[k] for k in range(self.K)]

    
    def log_likelihood(self,rdd):
        lll = rdd.map(lambda x: logsumexp(np.array([self.log_pi_pdf(x, k, self.pis, self.mus, self.sigmas) for k in range(self.K)])))\
                 .reduce(lambda x,y: x+y)
        return lll

    
    def clustering(self, rdd):
        
        """ Returns the predicted cluster for each observation of the rdd """
        if not self.fitted:
            raise NotFittedError('EM must be fitted before performing clustering')
            
        predictions = rdd.map(lambda x: np.array([self.log_pi_pdf(x, k, self.pis, self.mus, self.sigmas) for k in range(self.K)]))\
                     .map(lambda x: np.argmax(x))\
                     .take(self.N)
                
        self.predicted_clusters = np.array(predictions)
        
        return self.predicted_clusters