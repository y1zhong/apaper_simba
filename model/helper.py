import numpy as np
import torch
from model.utils import save_pickle 

def scale(vec, tmin=-1, tmax=1):
    ''' scale function tp [-1, 1]'''

    vmin = np.min(vec, axis=0)
    vmax = np.max(vec, axis=0)
    scaled = (vec - vmin) / (vmax - vmin) * (tmax - tmin) + tmin
    return scaled

def generate_grids(img_shape, V_range=[-1,1]):
    V = np.prod(img_shape)
    d = len(img_shape)
    assert((d >= 1) & (d <=3))
    if (d == 1) :
          x_ = np.linspace(V_range[0], V_range[1], img_shape[0])
          S = torch.from_numpy(x_).reshape(V,1)
    elif (d == 2) :
            x_ = np.linspace(V_range[0], V_range[1], img_shape[0])
            y_ = np.linspace(V_range[0],  V_range[1], img_shape[1])
            x, y = np.meshgrid(x_, y_,  indexing='ij')
            S = torch.from_numpy(np.column_stack((x.reshape(-1), y.reshape(-1))))
    else:
            x_ = np.linspace(V_range[0],  V_range[1], img_shape[0])
            y_ = np.linspace(V_range[0],  V_range[1], img_shape[1])
            z_ = np.linspace(V_range[0],  V_range[1], img_shape[2])

            x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
            S = torch.from_numpy(np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1))))
    return S.float()



def matern_kernel(X, Y, lengthscale=1.0, variance=1.0, nu=1.5):
    pairwise_dists = torch.cdist(X, Y, p=2)  # Euclidean distance
    sqrt3 = torch.sqrt(torch.tensor(3.0))
    sqrt5 = torch.sqrt(torch.tensor(5.0))
    
    if nu == 0.5:
        K = torch.exp(-pairwise_dists / lengthscale)  # Exponential kernel
    elif nu == 1.5:
        K = (1.0 + sqrt3 * pairwise_dists / lengthscale) * torch.exp(-sqrt3 * pairwise_dists / lengthscale)
    elif nu == 2.5:
        K = (1.0 + sqrt5 * pairwise_dists / lengthscale + (5.0/3.0) * (pairwise_dists / lengthscale)**2) * torch.exp(-sqrt5 * pairwise_dists / lengthscale)
    else:
        raise ValueError("Currently supports only nu = 0.5, 1.5, 2.5")
    
    return variance * K

def ker_approx(kernel, grids, grids_c, err = 1e-4, dtype=torch.float32):
    '''
    Get low-rank approxiamted kernel using inducing points

    Args:
        kernel (sklearn.gaussian_process.kernels): kernel objects
        grids (torch.tensor): spatial locations
        grids_c: inducing coordiantes
    Returns:
        U: tensor of basis functions (V, L)
        D: tensor of square root of eigenvalues associated to the basis (L)
    '''
    dtype = dtype
    Kc = torch.from_numpy(kernel(grids_c, grids_c)).to(dtype=dtype)
    Kc  += torch.eye(Kc.shape[0]) * err
    Ksc = torch.from_numpy(kernel(grids, grids_c)).to(dtype=dtype)

    R = torch.linalg.cholesky(Kc)
    R_inv = torch.linalg.inv(R.T)
    KR = (Ksc @ R_inv)
    U, D, _ = torch.linalg.svd(KR, full_matrices=False)
    return U, D

def get_basis(grids, L, kernel, err = 1e-4, dtype=torch.float32):
    """
    Get basis functions from low-rank approximated kernel

    Args:
        grids (torch.tensor): d-dimensional spatial coordinates (V, L)
        L (int): number of basis used for kernel approximation (tuning parameter)
        kernel (sklearn.gaussian_process.kernels): kernel objects
        err (float): a small jitter to ensure valid cholesky decomposition
        dtype (torch.dtype): output tensor dtype

    Returns:
        B: tensor of basis functions (V, L)
        eig_val_sqrt: tensor of square root of eigenvalues associated to the basis (L)
    """
    V = grids.shape[0]
    # select equally spaced inducing points
    induce = torch.linspace(0, V, steps=L+2)[1:-1].long()
    grids_c = grids[induce,:]
    B, eig_val_sqrt = ker_approx(kernel, grids, grids_c, err = err, dtype = dtype)
    return B, eig_val_sqrt

def scale(vec, tmin=-1, tmax=1):
    ''' scale function tp [-1, 1]'''

    vmin = np.min(vec, axis=0)
    vmax = np.max(vec, axis=0)
    scaled = (vec - vmin) / (vmax - vmin) * (tmax - tmin) + tmin
    return scaled
       
def est_ls(y, S, k = 'RBF', init_ls = 0.5, init_var = 1.0, lr = 0.01, prop=0.2, iters=400):
    #y = (y - y.mean(1)[:,None]) / y.std(1)[:,None]
    #cov_mat = torch.cov(y)
    if k == 'RBF':
        model = RBFKernel(ls=init_ls, var=init_var)
    elif k == 'Matern32':
        model = Matern32Kernel(ls=init_ls, var=init_var)
    elif k == 'Matern52':
        model = Matern52Kernel(ls=init_ls, var=init_var)
    
    V_est = int(S.shape[0] * prop)
    y = y[:V_est]
    S = S[:V_est]
    cov_mat = torch.cov(y)
    #obs = torch.triu(cov_mat[:V_est, :V_est], diagonal=1)
    #obs = obs[obs != 0]
    triu_indices = torch.triu_indices(V_est, V_est, offset=1)
    obs = cov_mat[triu_indices[0], triu_indices[1]]
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for iteration in range(iters):
        optimizer.zero_grad()  
        est = model(S)
        loss = torch.mean((obs - est) ** 2)
        loss.backward()
        optimizer.step()
        if iteration % 1000 == 0:
            print(f"Iteration {iteration + 1}/{iters}, "
                f"ls: {model.rho.item():.4f}, var: {model.var.item():.4f}, ")
    print(f"Iteration {iteration + 1}/{iters}, "
                f"ls: {model.rho.item():.4f}, var: {model.var.item():.4f},")
    
    return model.raw_rho.detach(), model.raw_rho.detach()