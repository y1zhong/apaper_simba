import torch
from tqdm import tqdm
import time
import numpy as np
import multiprocessing as mp
from model.helper import ker_approx, get_basis


class SIMBA_VI():
    def __init__(self, Y, grids, kernel, L=None, L_eta=None, 
                kernel_eta=None, X=None, include_intercept = True,
                max_iter = 1000, ELBO_diff_tol=1e-4, para_diff_tol=1e-4, elbo_stop=True,
                dtype=torch.float32,verbose=200,A2=100,seed=42
                ):
        torch.manual_seed(seed)
        self.y = Y.to(dtype) # V by n
        self.N, self.V = Y.shape # num of subVect
        self.dtype=dtype

        if X is None:
            X = torch.ones(self.N,1)
        else:
            if include_intercept:
                X = torch.column_stack((torch.ones(self.N,1), X))
        self.X = X.to(dtype)
        self.J = X.shape[1]

        if kernel_eta is None:
            kernel_eta = kernel

        ### VI settings
        self.max_iter = max_iter
        self.ELBO_diff_tol = ELBO_diff_tol
        self.para_diff_tol = para_diff_tol

        assert L_eta < L

        self.L = L
        self.L_eta = L_eta
        self.B, self.eig_val_sqrt = get_basis(grids, L,  kernel, dtype=dtype)
        self.B_eta, self.eig_val_sqrt_eta = get_basis(grids, L_eta,  kernel_eta, dtype=dtype)
        self.basis =  self.B * (self.eig_val_sqrt.unsqueeze(0) )
        self.basis_eta =  self.B_eta * (self.eig_val_sqrt_eta.unsqueeze(0) )
        
        self.B_lamb = self.B * (self.eig_val_sqrt.unsqueeze(0) )
        self.B_lamb_inv = self.B * ( (1.0 / self.eig_val_sqrt).unsqueeze(0))  # V by L

        self.B_lamb_eta_t = (self.B_eta * (self.eig_val_sqrt_eta.unsqueeze(0) )).t()
 
        self.y_tilde = self.y @ self.B_lamb_inv
        self.B_lamb_inv_sumv = self.B_lamb_inv.sum(0)
        self.B_lamb_inv_sumv_ssq = (self.B_lamb_inv_sumv ** 2).sum()

        self.B_prime = self.B_lamb_eta_t @ self.B_lamb_inv # L' by L
        self.B_prime_sq = self.B_prime @ self.B_prime.t()


        self.A2 = A2
        self.digamma_one = torch.digamma(torch.ones(1))[0]

        
        self.loglik_y = []
        self.ELBO = []
        self.elbo = 0
        self.elbo_stop = elbo_stop
        self.verbose = verbose


    def run(self, verbose=True, mute=True):
        self.fit(verbose=verbose, mute=mute)
        self.paras = self.get_samples()
        self.profile = self.get_profile()
        return self.paras, self.profile

    def init_paras(self):
        self.E_alpha = torch.randn(self.J, dtype=self.dtype) 
        self.Var_alpha = torch.ones(self.J)
        self.E_SS_alpha = torch.ones(self.J)

        self.E_theta_eta = torch.randn(self.N, self.L_eta, dtype=self.dtype)
        self.Var_theta_eta = torch.ones(self.L_eta, self.L_eta)
        self.E_SS_theta_eta = torch.ones(self.N, self.L_eta)
        
        self.E_theta_beta = torch.randn(self.J, self.L, dtype=self.dtype)
        self.Var_theta_beta = torch.ones(self.J)
        self.E_SS_theta_beta = torch.ones(self.J, self.L)

        self.E_inv_sig2_eps = torch.rand(1, dtype=self.dtype) 
        self.E_inv_sig2_eta = torch.rand(1, dtype=self.dtype)
        self.E_inv_sig2_beta = torch.rand(1, dtype=self.dtype) 
        self.E_inv_sig2_alpha = torch.rand(1, dtype=self.dtype) 
        self.E_inv_a_beta, self.E_inv_a_eta, self.E_inv_a_eps,  self.E_inv_a_alpha = torch.ones(4)
        self.update_res()

    def fit(self, verbose=False, mute=False):
        self.init_paras()
        start_time = time.time()
        for i in tqdm(range(self.max_iter), disable=mute): 
            self.update_E_alpha()
            self.update_E_theta_eta()
            self.update_E_theta_beta()
            self.update_E_inv_sig2_alpha()
            self.update_E_inv_sig2_beta()
            self.update_E_inv_sig2_eta()
            self.update_E_inv_sig2_eps()

            #self.loglik_y[i] = self.update_loglik_y()

            elbo_prev = self.elbo
            self.monitor_vb()
            if self.elbo_stop:
                if((self.elbo - elbo_prev).abs() < self.ELBO_diff_tol):
                    break
            else:
                para_diff = self.E_alpha_diff + self.E_theta_beta_diff + self.E_theta_eta_diff
                para_diff += self.E_inv_sig2_beta_diff + self.E_inv_sig2_eps_diff + self.E_inv_sig2_eta_diff
                if(para_diff < self.ELBO_diff_tol):
                    break
            
            if self.verbose > 0:
                if(i % self.verbose == 0):
                    print(f"iter: {i}, ELBO: {self.elbo}")
               
        self.runtime = time.time() - start_time
        if verbose:
            print(f"Used iter {i} finished in {self.runtime:.2f} seconds")

    def update_res(self):
        self.res = self.y_tilde -  (self.X @ self.E_alpha)[:,None]  * self.B_lamb_inv_sumv  - self.E_theta_eta @ self.B_prime - self.X @ self.E_theta_beta

    def update_E_alpha(self):
        E_alpha_old = self.E_alpha
        self.norm_alpha = 0 
        for j in range(self.J):
            self.res += (self.X[:,j] * self.E_alpha[j])[:,None] * self.B_lamb_inv_sumv
            self.Var_alpha[j] = 1 / ( (self.X[:,j] ** 2 ).sum() * self.B_lamb_inv_sumv_ssq * self.E_inv_sig2_eps + self.E_inv_sig2_alpha)
            self.E_alpha[j] = self.Var_alpha[j] * (self.B_lamb_inv_sumv * self.E_inv_sig2_eps * self.res * self.X[:,j:j+1]).sum() 
            self.res -= (self.X[:,j] * self.E_alpha[j])[:,None] * self.B_lamb_inv_sumv
            
        self.norm_alpha += ((self.X ** 2).sum(0) * self.Var_alpha).sum() * self.B_lamb_inv_sumv_ssq 
        self.E_SS_alpha = self.E_alpha ** 2 + self.Var_alpha
        self.E_alpha_diff = ((E_alpha_old - self.E_alpha).abs()).mean()
 
    
    def update_E_theta_beta(self):
        #print(self.eta)
        E_theta_beta_old = self.E_theta_beta
        for j in range(self.J):
            self.res += self.X[:,j:j+1] @ self.E_theta_beta[j:j+1,:]
            
            self.Var_theta_beta[j] = 1 / ((self.X[:,j] ** 2 ).sum() * self.E_inv_sig2_eps +  self.E_inv_sig2_beta)
            self.E_theta_beta[j,:] = self.Var_theta_beta[j] * ( self.res * self.X[:,j:j+1] * self.E_inv_sig2_eps).sum(0) 
            #self.E_theta_beta -= self.E_theta_beta.mean() 

            self.res -= self.X[:,j:j+1] @ self.E_theta_beta[j:j+1,:]
        self.E_SS_theta_beta = self.E_theta_beta ** 2 + self.Var_theta_beta.unsqueeze(-1)
        #self.norm_theta_beta = self.N * self.L * self.Var_theta_beta
        self.norm_theta_beta = self.L * torch.trace(self.X @ torch.diag(self.Var_theta_beta) @ self.X.t())
        self.E_theta_beta_diff = ((E_theta_beta_old - self.E_theta_beta).abs()).mean()

        

    def update_E_theta_eta(self):
        #print(self.eta)
        E_theta_eta_old = self.E_theta_eta
        self.res += self.E_theta_eta @ self.B_prime
        precision = self.B_prime_sq * self.E_inv_sig2_eps + torch.eye(self.L_eta) * self.E_inv_sig2_eta
        self.Cov_theta_eta = precision.inverse()
        self.Var_theta_eta = self.Cov_theta_eta.diag()
        self.norm_theta_eta = torch.trace(self.B_prime.t() @ self.Cov_theta_eta @ self.B_prime)

        temp = (self.res @ self.B_prime.t()) * self.E_inv_sig2_eps

        for i in range(self.N):
            self.E_theta_eta[i] = self.Cov_theta_eta @ temp[i]
            self.E_theta_eta[i] -= self.E_theta_eta[i].mean()
            self.E_SS_theta_eta[i] = self.E_theta_eta[i] ** 2 + self.Var_theta_eta
            
        self.res -= self.E_theta_eta @ self.B_prime
        self.E_theta_eta_diff = ((E_theta_eta_old - self.E_theta_eta).abs()).mean()

    def update_E_inv_sig2_eps(self):
        E_inv_sig2_eps_old = self.E_inv_sig2_eps
        self.a_sig2_eps = torch.tensor(( 1 + self.L * self.N )/ 2 )
        self.b_sig2_eps = self.E_inv_a_eps + ((self.res ** 2).sum() + self.norm_alpha + self.norm_theta_beta + self.norm_theta_eta) / 2

        self.E_inv_sig2_eps = self.a_sig2_eps / self.b_sig2_eps
        self.E_inv_a_eps = 1 / (1/self.A2 + self.E_inv_sig2_eps)
        
        self.Var_inv_sig2_eps = self.a_sig2_eps / self.b_sig2_eps ** 2
        self.Var_inv_a_eps = 1 / (1/self.A2 + self.E_inv_sig2_eps) ** 2

        self.E_log_sig2_eps = torch.log(self.b_sig2_eps) - torch.digamma(self.a_sig2_eps)
        self.E_log_a_eps = torch.log((1/self.A2 + self.E_inv_sig2_eps)) - self.digamma_one

        self.E_inv_sig2_eps_diff = ((E_inv_sig2_eps_old - self.E_inv_sig2_eps).abs()).mean()

    def update_E_inv_sig2_alpha(self):
        E_inv_sig2_alpha_old = self.E_inv_sig2_alpha
        self.a_sig2_alpha = torch.tensor(( 1 + self.J )/ 2 )
        self.b_sig2_alpha = self.E_SS_alpha.sum() / 2 + self.E_inv_a_alpha

        self.E_inv_sig2_alpha = self.a_sig2_alpha / self.b_sig2_alpha
        self.E_inv_a_alpha = 1 / (1/self.A2 + self.E_inv_sig2_alpha)

        self.Var_inv_sig2_alpha = self.a_sig2_alpha / self.b_sig2_alpha ** 2
        self.Var_inv_a_alpha = 1 / (1/self.A2 + self.E_inv_sig2_alpha) ** 2

        self.E_log_sig2_alpha = torch.log(self.b_sig2_alpha) - torch.digamma(self.a_sig2_alpha)
        self.E_log_a_alpha = torch.log((1/self.A2 + self.E_inv_sig2_alpha)) - self.digamma_one
        
        self.E_inv_sig2_alpha_diff = ((E_inv_sig2_alpha_old - self.E_inv_sig2_alpha)**2).mean()



    def update_E_inv_sig2_eta(self):
        E_inv_sig2_eta_old = self.E_inv_sig2_eta
        self.a_sig2_eta = torch.tensor(( 1 + self.L_eta * self.N )/ 2 )
        self.b_sig2_eta = self.E_SS_theta_eta.sum() / 2 + self.E_inv_a_eta

        self.E_inv_sig2_eta = self.a_sig2_eta / self.b_sig2_eta
        self.E_inv_a_eta = 1 / (1/self.A2 + self.E_inv_sig2_eta)

        self.Var_inv_sig2_eta = self.a_sig2_eta / self.b_sig2_eta ** 2
        self.Var_inv_a_eta = 1 / (1/self.A2 + self.E_inv_sig2_eta) ** 2

        self.E_log_sig2_eta = torch.log(self.b_sig2_eta) - torch.digamma(self.a_sig2_eta)
        self.E_log_a_eta = torch.log((1/self.A2 + self.E_inv_sig2_eta)) - self.digamma_one
        
        self.E_inv_sig2_eta_diff = ((E_inv_sig2_eta_old - self.E_inv_sig2_eta)**2).mean()

    def update_E_inv_sig2_beta(self):
        E_inv_sig2_beta_old = self.E_inv_sig2_beta
        self.a_sig2_beta = torch.tensor(( 1 + self.L )/ 2 )
        self.b_sig2_beta = self.E_SS_theta_beta.sum() / 2 + self.E_inv_a_beta

        self.E_inv_sig2_beta = self.a_sig2_beta / self.b_sig2_beta
        self.E_inv_a_beta = 1 / (1/self.A2 + self.E_inv_sig2_beta)

        self.Var_inv_sig2_beta = self.a_sig2_beta / self.b_sig2_beta ** 2
        self.Var_inv_a_beta = 1 / (1/self.A2 + self.E_inv_sig2_beta) ** 2

        self.E_log_sig2_beta = torch.log(self.b_sig2_beta) - torch.digamma(self.a_sig2_beta)
        self.E_log_a_beta = torch.log((1/self.A2 + self.E_inv_sig2_beta)) - self.digamma_one

        self.E_inv_sig2_beta_diff = ((E_inv_sig2_beta_old - self.E_inv_sig2_beta)**2).mean()

    def update_loglik_y(self):
        self.logll = (- self.N / 2 * torch.log(2. * torch.pi * self.E_inv_sig2_eps)).sum() - 0.5 * (self.res ** 2 * self.E_inv_sig2_eps).sum() 

    def update_ELBO(self):
        self.update_E_log_post()
        self.update_entropy()
        self.elbo = self.E_log_post - self.entropy

    def update_E_log_post(self):
        self.E_log_post = -0.5 * self.N * self.L * self.E_log_sig2_eps - 0.5 * self.E_inv_sig2_eps * ((self.res ** 2).sum() + self.norm_alpha + self.norm_theta_beta + self.norm_theta_eta)
        self.E_log_post += -0.5 * self.J * self.E_log_sig2_alpha - 0.5 * self.E_inv_sig2_alpha * self.E_SS_alpha.sum()
        self.E_log_post += -0.5 * self.L * self.E_log_sig2_beta - 0.5 * self.E_inv_sig2_beta * self.E_SS_theta_beta.sum()
        self.E_log_post += -0.5 * self.L_eta * self.N * self.E_log_sig2_eta - 0.5 * self.E_inv_sig2_eta * self.E_SS_theta_eta.sum()
        self.E_log_post += -1.5 * self.E_log_sig2_eps - 0.5 * self.E_inv_sig2_eps * self.E_inv_a_eps
        self.E_log_post += -1.5 * self.E_log_sig2_alpha - 0.5 * self.E_inv_sig2_alpha * self.E_inv_a_alpha
        self.E_log_post += -1.5 * self.E_log_sig2_beta - 0.5 * self.E_inv_sig2_beta * self.E_inv_a_beta
        self.E_log_post += -1.5 * self.E_log_sig2_eta - 0.5 * self.E_inv_sig2_eta *  self.E_inv_a_eta
        #print(self.E_log_a_eps, self.E_inv_a_eps,self.A2,  self.E_log_post)
        self.E_log_post += -1.5 * self.E_log_a_eps - 0.5 *  self.E_inv_a_eps / self.A2
        self.E_log_post += -1.5 * self.E_log_a_alpha - 0.5 *  self.E_inv_a_alpha / self.A2
        self.E_log_post += -1.5 * self.E_log_a_beta - 0.5 * self.E_inv_a_beta / self.A2
        self.E_log_post += -1.5 * self.E_log_a_eta - 0.5 * self.E_inv_a_eta / self.A2

    def enable_grad(self):
        self.E_alpha = self.E_alpha.requires_grad_()
        self.E_theta_beta = self.E_theta_beta.requires_grad_()
        self.E_theta_eta = self.E_theta_eta.requires_grad_()
        self.E_inv_sig2_eps= self.E_inv_sig2_eps.requires_grad_()
        self.E_inv_sig2_beta= self.E_inv_sig2_beta.requires_grad_()
        self.E_inv_sig2_eta= self.E_inv_sig2_eta.requires_grad_()

        self.E_inv_a_eps= self.E_inv_a_eps.requires_grad_()
        self.E_inv_beta= self.E_inv_a_beta.requires_grad_()
        self.E_inv_eta= self.E_inv_a_eta.requires_grad_()

        
        
    def update_entropy(self):
        self.entropy = -0.5 * torch.log(self.Var_alpha).sum()
        self.entropy += -0.5 * torch.log(self.L * self.Var_theta_beta).sum()
        self.entropy += -0.5 * self.N * torch.logdet(self.Cov_theta_eta)
        self.entropy += - self.a_sig2_eps - self.b_sig2_eps.log() - self.a_sig2_eps.lgamma() + (self.a_sig2_eps+1) * self.a_sig2_eps.digamma()
        self.entropy += - self.a_sig2_alpha - self.b_sig2_alpha.log() - self.a_sig2_alpha.lgamma() + (self.a_sig2_alpha+1) * self.a_sig2_alpha.digamma()
        self.entropy += - self.a_sig2_beta - self.b_sig2_beta.log() - self.a_sig2_beta.lgamma() + (self.a_sig2_beta+1) * self.a_sig2_beta.digamma()
        self.entropy += - self.a_sig2_eta - self.b_sig2_eta.log() - self.a_sig2_eta.lgamma() + (self.a_sig2_eta+1) * self.a_sig2_eta.digamma()
        self.entropy += - torch.log((1/self.A2 + self.E_inv_sig2_beta))
        self.entropy += - torch.log((1/self.A2 + self.E_inv_sig2_alpha))
        self.entropy += - torch.log((1/self.A2 + self.E_inv_sig2_eta))
        self.entropy += - torch.log((1/self.A2 + self.E_inv_sig2_eps))
    
    def monitor_vb(self):
        self.update_ELBO()
        self.update_loglik_y()
        self.ELBO.append(self.elbo)
        self.loglik_y.append(self.logll)
        
        #vb_profile.E_beta_mat.col(profile_iter) = vb_paras.E_beta


    def get_samples(self):
        return {
            "E_alpha": self.E_alpha,
            "Var_alpha":self.Var_alpha,

            "E_theta_eta": self.E_theta_eta,
            "Var_theta_eta":self.Var_theta_eta,

            "E_theta_beta": self.E_theta_beta,
            "Var_theta_beta":self.Var_theta_beta,

            "E_inv_sig2_eps": self.E_inv_sig2_eps,
            "E_inv_sig2_alpha": self.E_inv_sig2_alpha,
            "E_inv_sig2_beta": self.E_inv_sig2_beta,
            "E_inv_sig2_eta": self.E_inv_sig2_eta,

            "Var_inv_sig2_eps": self.E_inv_sig2_eps,
            "Var_inv_sig2_alpha": self.E_inv_sig2_alpha,
            "Var_inv_sig2_beta": self.E_inv_sig2_beta,
            "Var_inv_sig2_eta": self.E_inv_sig2_eta,
            

            "E_inv_a_eps": self.E_inv_a_eps,
            "E_inv_a_alpha": self.E_inv_a_alpha,
            "E_inv_a_beta": self.E_inv_a_beta,
            "E_inv_a_eta": self.E_inv_a_eta,
            
            "Var_inv_a_eps": self.E_inv_a_eps,
            "Var_inv_a_alpha": self.E_inv_a_alpha,
            "Var_inv_a_beta": self.E_inv_a_beta,
            "Var_inv_a_eta": self.E_inv_a_eta,

            'a_sig2_eps':self.a_sig2_eps,
            'b_sig2_eps':self.b_sig2_eps,
        }
    
    def get_profile(self):
        return {
            "ELBO": self.ELBO,
            "ll": self.loglik_y,
        }
    
    def get_eff_mcmc(self, n_mcmc=1000):
        E_alpha = self.paras['E_alpha']
        E_theta_beta = self.paras['E_theta_beta']

        Var_alpha = self.paras['Var_alpha']
        Var_theta_beta = self.paras['Var_theta_beta']

        mcmc_alpha = torch.randn(n_mcmc, self.J) * Var_alpha[None,:].sqrt() + E_alpha[None,:]
        mcmc_theta_beta = torch.randn(n_mcmc,self.J, self.L) * Var_theta_beta.sqrt()[None,:,None] + E_theta_beta[None,:,:]
        voxel_mcmc = mcmc_alpha[:,None] + mcmc_theta_beta @ self.basis.t()
        return voxel_mcmc

    def post_samples(self, n_mcmc=500):
        E_alpha = self.paras['E_alpha']
        E_theta_beta = self.paras['E_theta_beta']

        Var_alpha = self.paras['Var_alpha']
        Var_theta_beta = self.paras['Var_theta_beta']

        mcmc_alpha = torch.randn(n_mcmc, self.J) * Var_alpha.sqrt() + E_alpha
        mcmc_theta_beta= torch.randn(n_mcmc, self.J, self.L) * Var_theta_beta.sqrt()[:,None] + E_theta_beta[None,:,:]

        voxel_mcmc = mcmc_alpha[:,:,None] + mcmc_theta_beta @ self.basis.t()

        return voxel_mcmc
    
    def PPC(self, n_mcmc=100, dtype=torch.float32):
        """
        Posterior Predictive Check (PPC)

        Args:
            n_sample (int): number of posterior samples
            dtype (torch.dtype): output tensor dtype

        Returns:
            pred_y: tensor of shape (n_sample, N, V)
        """

        E_alpha = self.paras['E_alpha']
        E_theta_beta = self.paras['E_theta_beta']
        E_theta_eta = self.paras['E_theta_eta']

        Var_alpha = self.paras['Var_alpha']
        Var_theta_beta = self.paras['Var_theta_beta']
        Var_theta_eta = self.paras['Var_theta_eta']

        a_sig2_eps = self.paras['a_sig2_eps']
        b_sig2_eps = self.paras['b_sig2_eps']


        mcmc_alpha = torch.randn(n_mcmc, self.J) * Var_alpha[None,:].sqrt() + E_alpha[None,:]
        mcmc_theta_eta = torch.randn(n_mcmc, self.N, self.L_eta) * Var_theta_eta[None, None,:].sqrt() + E_theta_eta[None,:,:]
        mcmc_theta_beta = torch.randn(n_mcmc,self.J, self.L) * Var_theta_beta.sqrt()[None,:,None] + E_theta_beta[None,:,:]
        mcmc_inv_sig2_e = torch.distributions.Gamma(a_sig2_eps, b_sig2_eps).sample((n_mcmc,))
        
        pred_y = torch.zeros(n_mcmc, self.N, self.V, dtype=dtype)
        basis_eta_t = self.basis_eta.t()
        
        for s in range(n_mcmc):
            alpha = mcmc_alpha[s]
            theta_eta = mcmc_theta_eta[s]
            theta_beta = mcmc_theta_beta[s]
            inv_sig2_e = mcmc_inv_sig2_e[s]

            mean = (self.X @ alpha)[:,None] + theta_eta @ basis_eta_t +  self.X @ (theta_beta @ self.basis.t())
            noise = self.basis @ (torch.randn(self.L, self.N) / inv_sig2_e.sqrt())
            pred_y[s] = mean + noise.t()
        return pred_y

    
    # def get_basis(self):
    #     return self.B_lamb



def PPC(data, X, paras, basis, basis_eta, dtype=torch.float32, n_mcmc=500):
    """
    Posterior Predictive Check (PPC)

    Args:
        n_sample (int): number of posterior samples
        dtype (torch.dtype): output tensor dtype

    Returns:
        pred_y: tensor of shape (n_sample, N, V)
    """

    E_alpha = paras['E_alpha']
    E_theta_beta = paras['E_theta_beta']
    E_theta_eta = paras['E_theta_eta']

    Var_alpha = paras['Var_alpha']
    Var_theta_beta = paras['Var_theta_beta']
    Var_theta_eta = paras['Var_theta_eta']

    a_sig2_eps = paras['a_sig2_eps']
    b_sig2_eps = paras['b_sig2_eps']

    N, V = data.shape
    L = basis.shape[1]
    L_eta = basis_eta.shape[1]
    J = E_alpha.shape[0]


    mcmc_alpha = torch.randn(n_mcmc, J) * Var_alpha[None,:].sqrt() + E_alpha[:,None]
    mcmc_theta_eta = torch.randn(n_mcmc, N, L_eta) * Var_theta_eta[None, None,:].sqrt() + E_theta_eta[None,:,:]
    mcmc_theta_beta = torch.randn(n_mcmc,J, L) * Var_theta_beta.sqrt() + E_theta_beta[None,:,:]
    mcmc_alpha = torch.randn(n_mcmc, J) * Var_alpha[None,:].sqrt() + E_alpha[:,None]
    mcmc_inv_sig2_e = torch.distributions.Gamma(a_sig2_eps, b_sig2_eps).sample((n_mcmc,))
    
    pred_y = torch.zeros(n_mcmc, N, V, dtype=dtype)
    basis_eta_t = basis_eta.t()
    

    for s in range(n_mcmc):
        
        alpha = mcmc_alpha[s]
        theta_eta = mcmc_theta_eta[s]
        theta_beta = mcmc_theta_beta[s]
        inv_sig2_e = mcmc_inv_sig2_e[s]

        mean = (X @ alpha)[:,None] + theta_eta @ basis_eta_t +  X @ (theta_beta @ basis.t())
        noise = basis @ (torch.randn(L, N) / inv_sig2_e.sqrt())
        pred_y[s] = mean + noise.t()
    return pred_y


def get_ll(data, samples, dtype=torch.float32):
    """
    get log-likelihood for each observation (voxel-wise)

    Args:
        data (tensor): input data
        samples (dic): dictionary contains mcmc samples for model 1

    Returns:
        pred_y: tensor of shape (n_sample, N, V)
    """
    n_chains, n_mcmc = samples['alpha'].shape
    N, V = data.shape
    basis = samples['basis'][0]

    ll_mat = torch.zeros(n_chains, n_mcmc, V,  dtype=dtype)
    for s in range(n_chains):
        for ind in range(n_mcmc):
            alpha = samples['alpha'][s, ind].to(dtype=dtype)
            eta = samples['eta'][s, ind].to(dtype=dtype)
            theta_beta = samples['theta_beta'][s, ind].to(dtype=dtype)
            sig2_eps = samples['sig2_eps'][s, ind].to(dtype=dtype)

            mu = alpha + eta.unsqueeze(-1) + (basis @ theta_beta).unsqueeze(0)
            res = data - mu
            ll = -0.5 * torch.log(2 * torch.pi * sig2_eps) - 0.5 * (res**2) / sig2_eps
            #ll_mat[s,ind] = ll.sum(dim=1) # individual level
            ll_mat[s,ind] = ll.sum(dim=0)
    return ll_mat
    
