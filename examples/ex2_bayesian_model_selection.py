import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from monte_carlo_tools import ais

# Geerate synthetic data from a student-t with nu=3 degrees of freedom
n = 1000                                                # Number of observations to generate
y = np.random.standard_t(df=3, size=(n, 1))+4           # Generate observations


# Write down the target distribution for unknown constant (as a function of mean and degrees of freedom in likelihood)
# - Likelihood function is non-central student's t with centrality parameter = 0
# - Vague prior distribution is assumed x~N(mean=0, var=10000)
def log_target(x, nu): return np.sum(sp.nct.logpdf(y, df=nu, nc=0, loc=x), axis=0)+sp.norm.logpdf(x, loc=0, scale=100)


# Specify candidate models by the different degrees of freedom each of them can have
df_candidates = np.arange(1, 7)

# Allocate array to store evidence of each model
log_z_store = np.zeros(np.shape(df_candidates))

# Define the sampler parameters
dim = 1         # dimension of the unknown parameter
N = 25          # number of samples per proposal
I = 200         # number of iterations
N_w = 25        # number of samples per proposal (warm-up period)
I_w = 200       # number of iterations (warm-up period)
D = 5           # number of proposals

# Loop over the candidate models and approximate the model evidence for each
K = np.shape(df_candidates)[0]
for k in range(K):
    # Create a lambda for the log_target
    log_pi = lambda x: log_target(x.T, nu=df_candidates[k])
    # Generate the initial parameters
    mu_init = np.random.uniform(low=-10, high=10, size=(D, dim))        # Initialize means on [-10,10]^dim hypercube
    sig_init = np.tile(np.eye(dim), (D, 1, 1))                          # Use identity covariance for all proposals

    # Warm up the sampler by running it for some number of iterations
    init_sampler = ais.ais(log_target=log_pi, d=dim, mu=mu_init, sig=sig_init, samp_per_prop=N_w, iter_num=I_w,
                           temporal_weights=False, weight_smoothing=False, eta_mu0=1, eta_sig0=0.01)
    # Run sampler with initialized parameters
    output = ais.ais(log_target=log_pi, d=dim, mu=init_sampler.means[-D:], sig=init_sampler.covariances[-D:],
                     samp_per_prop=N, iter_num=I, weight_smoothing=False, temporal_weights=False, eta_mu0=0.1,
                     eta_sig0=0.001)
    # Obtain an approximation of the evidence
    log_z_store[k] = np.mean(np.exp(output.log_weights-np.max(output.log_weights)))+np.max(output.log_weights)
# Compute the posterior probability of each model
model_prob = np.exp(log_z_store-np.max(log_z_store))
model_prob = model_prob/np.sum(model_prob)
# Create a bar plot to show the distribution over models.
plt.figure()
plt.bar(df_candidates, model_prob, zorder=1)
line = plt.plot(3, 0, 'o', c='r', markersize=12, zorder=2)[0]
line.set_clip_on(False)
plt.title('Posterior Distribution over Models', fontsize=20)
plt.xlabel('Degrees of Freedom', fontsize=16)
plt.ylabel('Posterior Probability', fontsize=16)
plt.ylim([0, 1])
plt.legend(['Ground Truth', 'Posterior'])
plt.show()