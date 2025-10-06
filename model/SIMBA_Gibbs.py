import torch
from tqdm import tqdm
import time
import numpy as np
import multiprocessing as mp
from model.helper import ker_approx, get_basis
from model.helper import get_basis

class SIMBA_Gibbs():
    def __init__(self, Y, grids, kernel, L, L_eta, 
                kernel_eta=None, X=None, include_intercept=False,
                burnin=100, thin=1, mcmc_sample=100,
                A2=100,dtype=torch.float32,
                ):
        """
        Initialization SIMBA model with Gibbs sampler

        Args:
            Y (torch.tensor): Observed image data (N, V)
            grids (torch.tensor): d-dimensional spatial coordinates (V, L)
            kernel (sklearn.gaussian_process.kernels): kernel objects
            L (int): number of basis used for kernel approximation (tuning parameter)
            kernel_eta (sklearn.gaussian_process.kernels): kernel objects for individual effect
            L_eta (int): number of basis used for individual effect kernel approximation 
            X (torch.tensor): Individual-level covariates (N, J)
            burnin (int): burn-in samples for MCMC sampling
            thin (int): number of thinning for MCMC sampling
            mcmc_sample (int): number of posterior samples for MCMC sampling
            A2 (float): scale paprameter for IG priors
            dtype (torch.dtype): output tensor dtype
        """
        self.y = Y.to(dtype)
        self.N, self.V = Y.shape
        self.dtype=dtype

        # include intercept
        if X is None:
            X = torch.ones(self.N,1)
        else:
            if include_intercept:
                X = torch.column_stack((torch.ones(self.N, 1), X))
        self.X = X.to(dtype)
        self.J = X.shape[1]

        # set same kernel for spatially varying covariates if eta kernel is not specifies 
        if kernel_eta is None:
            kernel_eta = kernel

        ### mcmc settings
        self.mcmc_burnin = burnin
        self.mcmc_thinning = thin
        self.mcmc_sample = mcmc_sample 
        self.total_iter = self.mcmc_burnin + self.mcmc_sample * self.mcmc_thinning

        assert L_eta < L
        
        # get basis functions for GP kernels
        self.L = L
        self.L_eta = L_eta
        self.B, self.eig_val_sqrt = get_basis(grids, L,  kernel, err = 1e-4, dtype=dtype)
        self.B_eta, self.eig_val_sqrt_eta = get_basis(grids, L_eta,  kernel_eta, err = 1e-4, dtype=dtype)
        self.basis =  self.B * (self.eig_val_sqrt.unsqueeze(0) )
        self.basis_eta =  self.B_eta * (self.eig_val_sqrt_eta.unsqueeze(0) )
        

        # pre-computing
        self.B_lamb = self.B * (self.eig_val_sqrt.unsqueeze(0) )
        self.B_lamb_inv = self.B * ( (1.0 / self.eig_val_sqrt).unsqueeze(0))  # V by L

        self.B_lamb_eta_t = (self.B_eta * (self.eig_val_sqrt_eta.unsqueeze(0) )).t()
 
        self.y_tilde = self.y @ self.B_lamb_inv
        self.B_lamb_inv_sumv = self.B_lamb_inv.sum(0)
        self.B_lamb_inv_sumv_ssq = (self.B_lamb_inv_sumv ** 2).sum()

        self.B_prime = self.B_lamb_eta_t @ self.B_lamb_inv # L' by L
        self.B_prime_sq = self.B_prime @ self.B_prime.t()
        
        self.A2 = A2
        
    def run_single_chain(self, chain_id):
        self.fit(chain_id, verbose=True)
        return self.get_samples()
    
    def run(self, n_chains=4, parallel = False, seed = 42):
        """
        run multiple MCMC chains
        Returns:
            samples: List of MCMC samples for all parameters across chains 
        """
        self.n_chains = n_chains
        torch.manual_seed(seed)
        if parallel:
            ctx = mp.get_context("spawn")  
            with ctx.Pool(processes=n_chains) as pool:
                results = pool.starmap(self.run_single_chain, [(i,) for i in range(n_chains)])
        else:
            results=[]
            for i in range(n_chains):
                results.append(self.run_single_chain(i))

        #stack mcmcm samples in each chain
        results_chain = {}
        keys = results[0].keys()
        for key in keys:
            temp = [r[key] for r in results]  
            results_chain[key] = torch.stack(temp, dim=0) 
        self.samples = results_chain
        return self.samples
    
    def init_paras(self):
        self.alpha = torch.randn(self.J, dtype=self.dtype) 
      
        self.theta_eta = torch.randn(self.N, self.L_eta, dtype=self.dtype) 
        self.eta = self.theta_eta @ self.B_lamb_eta_t
        self.eta -= self.eta.mean(1, keepdim=True)
        
        self.theta_beta = torch.randn(self.J, self.L, dtype=self.dtype) 

        self.sig2_eta = torch.rand(1, dtype=self.dtype) 
        self.sig2_alpha = torch.rand(1, dtype=self.dtype) 
        self.sig2_eps = torch.rand(1, dtype=self.dtype)
        self.sig2_beta = torch.rand(1, dtype=self.dtype) 

        self.a_eta, self.a_eps, self.a_beta, self.a_alpha = torch.ones(4)
        self.update_res()

    def fit(self, chain_id=0, verbose=False, mute=False):
        self.init_paras()
        self.make_mcmc_samples()
        start_time = time.time()
        for i in tqdm(range(self.total_iter), disable=mute): 
            self.update_alpha()
            self.update_theta_eta()
            self.update_theta_beta()

            self.update_sig2_eta()
            self.update_sig2_alpha()
            self.update_sig2_beta()
            self.update_sig2_eps()

            self.loglik_y[i] = self.update_loglik_y()
            if i >= self.mcmc_burnin:
                if (i - self.mcmc_burnin) % self.mcmc_thinning == 0:
                    mcmc_iter = int((i - self.mcmc_burnin) / self.mcmc_thinning)
                    self.save_mcmc_samples(mcmc_iter)
        self.runtime = time.time() - start_time
        if verbose:
            print(f"Chain {chain_id + 1} finished in {self.runtime:.2f} seconds")
        
   
    def update_res(self):
        self.res = self.y_tilde - (self.X @ self.alpha)[:,None] * self.B_lamb_inv_sumv - self.theta_eta @ self.B_prime - self.X @ self.theta_beta

    def update_alpha(self):
        for j in range(self.J):
            self.res +=  (self.X[:,j] * self.alpha[j])[:,None] * self.B_lamb_inv_sumv
            sig2 = 1 / ( (self.X[:,j] ** 2 ).sum() * self.B_lamb_inv_sumv_ssq / self.sig2_eps + 1 / self.sig2_alpha)
            mu = sig2 * (self.X[:,j:j+1] * self.B_lamb_inv_sumv / self.sig2_eps  * self.res).sum() 
            self.alpha[j] = torch.randn(1) * sig2.sqrt() + mu
            self.res -= (self.X[:,j] * self.alpha[j])[:,None] * self.B_lamb_inv_sumv

    def update_theta_eta(self):
        self.res += self.theta_eta @ self.B_prime
        precision = self.B_prime_sq / self.sig2_eps + torch.eye(self.L_eta) / self.sig2_eta
       
        R = torch.linalg.cholesky(precision, upper=True)  + torch.eye(precision.shape[0]) * 1e-4
        temp = (self.res @ self.B_prime.t()) / self.sig2_eps
        Z = torch.randn(self.N, self.L_eta)

        for i in range(self.N):
            b = torch.linalg.solve(R.t(), temp[i])
            self.theta_eta[i] = torch.linalg.solve(R, Z[i]+b)

        self.theta_eta -= self.theta_eta.mean(1,keepdim=True) # center for identifiability
        self.eta = self.theta_eta @ self.B_lamb_eta_t
        self.eta -= self.eta.mean(1, keepdim=True)
        self.res -= self.theta_eta @ self.B_prime
    
    def update_theta_beta(self):

        for j in range(self.J):
            self.res += self.X[:,j:j+1] @ self.theta_beta[j:j+1,:]
            sig2 = 1 / ( (self.X[:,j] ** 2 ).sum()  / self.sig2_eps + 1 / self.sig2_beta )
            mu = sig2 * ( self.res * self.X[:,j:j+1] / self.sig2_eps ).sum(0) 
            self.theta_beta[j,:] = torch.randn(self.L) * sig2.sqrt() + mu
            self.theta_beta[j,:] -= self.theta_beta[j,:].mean()
            self.res -= self.X[:,j:j+1] @ self.theta_beta[j:j+1,:]

    def update_sig2_eps(self):
        a_new = (1 + self.L * self.N )/ 2 
        b_new = (self.res ** 2).sum() / 2  + 1 / self.a_eps
        m = torch.distributions.Gamma(a_new, b_new)
        self.sig2_eps = 1 / m.sample()

        m = torch.distributions.Gamma(1, 1/self.A2 + 1 / self.sig2_eps)
        self.a_eps = 1 / m.sample()

    def update_sig2_eta(self):
        a_new = (1 + self.L_eta * self.N )/ 2 
        b_new = ( self.theta_eta ** 2).sum() / 2 + 1 / self.a_eta
        m = torch.distributions.Gamma(a_new, b_new)
        self.sig2_eta = 1 / m.sample()

        m = torch.distributions.Gamma(1, 1/self.A2 + 1 / self.sig2_eta)
        self.a_eta = 1 / m.sample()

    def update_sig2_beta(self):
        a_new = (1 + self.J * self.L )/ 2 
        b_new = ( self.theta_beta ** 2).sum() / 2 + 1 / self.a_beta
        m = torch.distributions.Gamma(a_new, b_new)
        self.sig2_beta = 1 / m.sample()

        m = torch.distributions.Gamma(1, 1/self.A2 + 1 / self.sig2_beta)
        self.a_beta = 1 / m.sample()

    def update_sig2_alpha(self):
        a_new = (1 + self.J )/ 2 
        b_new = ( self.alpha ** 2).sum() / 2 + 1 / self.a_alpha
        m = torch.distributions.Gamma(a_new, b_new)
        self.sig2_alpha = 1 / m.sample()

        m = torch.distributions.Gamma(1, 1/self.A2 + 1 / self.sig2_alpha)
        self.a_alpha = 1 / m.sample()

    def update_loglik_y(self):
        logll = (- self.N / 2 * torch.log(2. * torch.pi * self.sig2_eps)).sum() - 0.5 * (self.res ** 2 / self.sig2_eps ).sum() 
        return logll

    def make_mcmc_samples(self):
        self.loglik_y = torch.zeros(self.total_iter)
        self.mcmc_alpha = torch.zeros(self.mcmc_sample, self.J, dtype=self.dtype)
        self.mcmc_theta_beta = torch.zeros(self.mcmc_sample, self.J, self.L, dtype=self.dtype)
        self.mcmc_theta_eta = torch.zeros(self.mcmc_sample,  self.N, self.L_eta, dtype=self.dtype)

        self.mcmc_sig2_beta = torch.zeros(self.mcmc_sample, dtype=self.dtype)
        self.mcmc_sig2_alpha = torch.zeros(self.mcmc_sample, dtype=self.dtype)
        self.mcmc_sig2_eta = torch.zeros(self.mcmc_sample, dtype=self.dtype)
        self.mcmc_sig2_eps = torch.zeros(self.mcmc_sample, dtype=self.dtype)


    def save_mcmc_samples(self, mcmc_iter):
        self.mcmc_alpha[mcmc_iter,:] = self.alpha
        self.mcmc_theta_eta[mcmc_iter, :,:] = self.theta_eta
        self.mcmc_theta_beta[mcmc_iter,:, :] = self.theta_beta

        self.mcmc_sig2_alpha[mcmc_iter] = self.sig2_alpha
        self.mcmc_sig2_beta[mcmc_iter] = self.sig2_beta
        self.mcmc_sig2_eta[mcmc_iter] = self.sig2_eta
        self.mcmc_sig2_eps[mcmc_iter] = self.sig2_eps

    def get_samples(self):
        return {
            "alpha": self.mcmc_alpha,
            "theta_eta": self.mcmc_theta_eta,
            "theta_beta": self.mcmc_theta_beta,
            "sig2_beta": self.mcmc_sig2_beta,
            "sig2_eta": self.mcmc_sig2_eta,
            "sig2_eps": self.mcmc_sig2_eps,
            "loglik": self.loglik_y,
            'runtime':torch.tensor(self.runtime),
        }
    
    def PPC(self, n_samples=100, dtype=torch.float32):
        """
        Posterior Predictive Check (PPC)

        Args:
            n_sample (int): number of posterior samples
            dtype (torch.dtype): output tensor dtype

        Returns:
            pred_y: tensor of shape (n_sample, N, V)
        """

        if n_samples > self.mcmc_sample:
            print("number of draws needs to be smaller than available mcmc samples")
        idx = torch.linspace(0, self.mcmc_sample - 1, n_samples, dtype=torch.int32)
        pred_y = torch.zeros(self.n_chains, n_samples, self.N, self.V, dtype=dtype)
        

        basis_eta_t = self.basis_eta.t()

        for s in range(self.n_chains):
            for i, ind in enumerate(idx):
                alpha = self.samples['alpha'][s, ind]
                theta_eta = self.samples['theta_eta'][s, ind]
                theta_beta = self.samples['theta_beta'][s, ind]
                sig2_eps = self.samples['sig2_eps'][s, ind]
                #sig2_eps = samples['sig2_eps'][s, ind]

                mean = (self.X @ alpha)[:,None] + theta_eta @ basis_eta_t + self.X @ (theta_beta @ self.basis.t())
                noise = self.basis @ (torch.randn(self.L, self.N) * sig2_eps.sqrt())
                pred_y[s, i] = mean + noise.t()
        return pred_y

