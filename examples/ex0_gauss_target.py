import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from monte_carlo_tools import ais

# Define a multivariate Gaussian target distribution
mu_pi = np.array([-1.50, 5.25, 0.45, -0.52, -2.6])
sig_pi = np.array([[1, -0.9, 0.2, 0, 0], [-0.9, 1, -0.25, 0, 0], [0.2, -0.25, 1, -0.5, 0],
                     [0, 0, -0.5, 1, 0.75], [0, 0, 0, 0.75, 1]])
dim = np.shape(mu_pi)[0]
log_pi = lambda x: np.log(0.01) + sp.stats.multivariate_normal.logpdf(x, mean=mu_pi, cov=0.05*sig_pi)

# Define the sampler parameters
N = 50          # number of samples per proposal
I = 400         # number of iterations
N_w = 25        # number of samples per proposal (warm-up period)
I_w = 500       # number of iterations (warm-up period)

# Generate the initial parameters
D = 10    # number of proposals
# Select initial proposal parameters
mu_init = np.random.uniform(low=-10, high=10, size=(D, dim))        # Initialize means on [-10,10]^dim hypercube
sig_init = np.tile(np.eye(dim), (D, 1, 1))                          # Use identity covariance for all proposals

# Warm up the sampler by running it
init_sampler = ais.ais(log_target=log_pi, d=dim, mu=mu_init, sig=sig_init, samp_per_prop=N_w, iter_num=I_w,
                       temporal_weights=False, weight_smoothing=False, eta_mu0=1, eta_sig0=0.01)

# Run sampler with initialized parameters
output = ais.ais(log_target=log_pi, d=dim, mu=init_sampler.means[-D:], sig=init_sampler.covariances[-D:],
                 samp_per_prop=N, iter_num=I, weight_smoothing=False, temporal_weights=False, eta_mu0=0.1,
                 eta_sig0=0.001)

# Use sampling importance resampling to extract posterior samples
theta = ais.importance_resampling(output.particles, output.log_weights, num_samp=1000)

# Matrix plot of the approximated target distribution
plt.figure()
count = 1
for i in range(dim):
    for j in range(dim):
        plt.subplot(dim, dim, count)
        if i != j:
            sns.kdeplot(theta[:, i], theta[:, j], cmap="Blues", shade=True, shade_lowest=False)
        else:
            plt.hist(theta[:, i])
        count += 1
plt.show()
