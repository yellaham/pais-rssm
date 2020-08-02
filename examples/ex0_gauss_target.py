import numpy as np
import scipy as sp
from monte_carlo_tools import ais

# Define a multivariate Gaussian target distribution
mu_pi = np.array([-1.50, 5.25, 0.45, -0.52, -2.6])
sig_pi = np.array([[1, -0.9, 0, 0, 0], [-0.9, 1, -0.25, 0, 0], [0, -0.25, 1, 0, 0],
                     [0, 0, 0, 1, -0.75], [0, 0, 0, -0.75, 1]])
dim = np.shape(mu_pi)[0]
log_pi = lambda x: np.log(100) + sp.stats.multivariate_normal.logpdf(x, mean=mu_pi, cov=0.1*sig_pi)

# Define the sampler parameters
N = 25       # number of samples per proposal
I = 150      # number of iterations

# Generate the initial parameters
D = 10      # number of proposals
# Select initial proposal parameters
mu_init = np.random.uniform(low=-5, high=5, size=(D, dim))        # Initialize means on [-10,10]^dim hypercube
sig_init = np.tile(np.eye(dim), (D, 1, 1))                          # Use identity covariance for all proposals

# Run it for a small number of iterations to initialize the sampler
output = ais.ais(log_target=log_pi, d=dim, mu=mu_init, sig=sig_init, samp_per_prop=N, iter_num=I,
                 eta_mu0=1e-1, eta_sig0=1e-2)
