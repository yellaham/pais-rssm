import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import multiprocessing
import seaborn as sns
from applications import penguins
from monte_carlo_tools import pf
from monte_carlo_tools import ais
from functools import partial

## DEFINITIONS
def log_jacobian_sigmoid(x): return -x-2*np.log(1+np.exp(-x))


# PART 1: DATA SIMULATION

# Parameters of the true model
#   -param[0] - juvenile survival
#   -param[1] - adult survival
#   -param[2] - reproductive success bias (bad year)
#   -param[3] - reproductive success bias (good year)
#   -param[4] - reproductive success slope
#   -param[5] - probability of bad year
#   -param[6] - variance of observations (breeders)
#   -param[7] - variance of observations (chicks)
param = np.array([0.35, 0.875, -1, -0.1, 0.1, 0.15, 0.01, 0.01])
# Number of stages to use for model
num_stages = 5

# Define the list of candidate models
candidate_models = [penguins.AgeStructuredModel(psi_juv=param[0], psi_adu=param[1], alpha_r=param[2], beta_r=param[4],
                                                var_s=param[6], var_c=param[7], nstage=num_stages),
                    penguins.AgeStructuredModel(psi_juv=param[0], psi_adu=param[1], alpha_r=param[3], beta_r=param[4],
                                                var_s=param[6], var_c=param[7], nstage=num_stages)]

# Define the regime dynamics
regime_dynamics_rand = lambda model_idx, num_samp: np.random.choice(np.arange(start=0, stop=2), num_samp,
                                                                    replace=True, p=np.array([param[5], 1-param[5]]))
regime_dynamics_log_pdf = lambda model_idx: model_idx*np.log(1-param[5])+(1-model_idx)*np.log(param[5])


# Create a multiple regime SSM and generate synthetic data
model = pf.MultiRegimeSSM(candidate_models, regime_dynamics_rand, regime_dynamics_log_pdf)

# Determine initial state
x_init = np.random.randint(low=500, high=2000, size=2*num_stages-2)

# Determine length of time to generate data for
time_generate = 100

# Generate ground truth for the regime switching system
y, x, m_idx = model.generate_data(init_state=x_init, T=time_generate)

# Cutoff the first 30 time points
cut_off = 60
time_length = time_generate-cut_off
y = y[-time_length:]
x = x[-(time_length+1):]
m_idx = m_idx[-time_length:]

# Plot the generated observations
plt.figure()
plt.plot(y[:, 0])
plt.plot(y[:, 1])
plt.legend(['Observed sum of adults', 'Observed sum of chicks'])
plt.show()


# PART 2: ASSUMED MODEL
def log_likelihood_per_sample(input_parameters):
    # First check to see that the monotonicity constraints are satisfied
    if not (input_parameters[2] < input_parameters[3]):
        return -np.inf
    # Set the random seed
    np.random.seed()
    # Define the number of particles
    num_particles = 500
    # Apply relevant transformations to the sample (sigmoid transformation to probability parameters)
    z = np.copy(input_parameters)
    z[0] = 1/(1+np.exp(-z[0]))
    z[1] = 1/(1+np.exp(-z[1]))
    z[3] = np.exp(z[3])
    z[4] = np.exp(z[4])
    z[5] = 1/(1+np.exp(-z[5]))
    # Evaluate prior distribution at transformed samples (don't forget to factor in Jacobian from transformation)
    log_prior = sp.beta.logpdf(z[0], 3, 3)+log_jacobian_sigmoid(input_parameters[0])
    log_prior += sp.beta.logpdf(z[1], 3, 3)+log_jacobian_sigmoid(input_parameters[1])
    log_prior += sp.norm.logpdf(z[2], 0, 1e6)
    log_prior += sp.gamma.logpdf(z[3], 0.001, 0.001)+input_parameters[3]
    log_prior += sp.gamma.logpdf(z[4], 0.001, 0.001)+input_parameters[4]
    log_prior += sp.beta.logpdf(z[5], 1, 4)+log_jacobian_sigmoid(input_parameters[5])
    # Create the model (assuming the noise variances are known)
    regimes = [penguins.AgeStructuredModel(psi_juv=z[0], psi_adu=z[1], alpha_r=z[2], beta_r=z[4], var_s=param[6],
                                          var_c=param[7], nstage=num_stages),
               penguins.AgeStructuredModel(psi_juv=z[0], psi_adu=z[1], alpha_r=z[2]+z[3], beta_r=z[4], var_s=param[6],
                                    var_c=param[7], nstage=num_stages)]
    draw_regimes = lambda model_idx, num_samp: np.random.choice(np.arange(start=0, stop=2), num_samp, replace=True,
                                                                        p=np.array([z[5], 1 - z[5]]))
    regimes_log_pdf = lambda model_idx: model_idx*np.log(1-z[5])+(1-model_idx)*np.log(z[5])
    # Create regime switching system
    model = pf.MultiRegimeSSM(regimes, draw_regimes, regimes_log_pdf)
    # Draw the initial particles
    x_init = np.array([x[0]]).T+np.random.randint(low=-100, high=100, size=(2*num_stages-2, num_particles))
    # Run the particle filter and return the log-likelihood
    output = pf.brspf(y, model, x_init)
    return output.log_evidence+log_prior


# PART 3: PARAMETER INFERENCE
if __name__ == '__main__':
    # Set a random seed
    np.random.seed()
    # Setup the multiprocessing bit
    multiprocessing.Process()
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)
    # Define the target distribution
    log_pi = lambda x: pool.map(log_likelihood_per_sample, x)
    # Define the sampler parameters
    dim = 6  # dimension of the unknown parameter
    N = 100  # number of samples per proposal
    I = 100  # number of iterations
    N_w = 60  # number of samples per proposal (warm-up period)
    I_w = 100  # number of iterations (warm-up period)
    D = 5   # number of proposals
    var_0 = 1e-1   # initial variance
    eta_loc = 5e-2  # learning rate for the mean
    eta_scale = 5e-2    # learning rate for the covariance matrix
    # Select initial proposal parameters
    mu_init = np.zeros((D, dim))
    sig_init = np.zeros((D, dim, dim))
    for j in range(D):
        mu_init[j, 0] = np.random.uniform(-1, 0)
        sig_init[j, 0, 0] = var_0
        mu_init[j, 1] = np.random.uniform(0.75, 1.5)
        sig_init[j, 1, 1] = var_0
        mu_init[j, 2] = np.random.uniform(-1, -0.35)
        sig_init[j, 2, 2] = var_0
        mu_init[j, 3] = np.random.uniform(-0.25, 0.25)
        sig_init[j, 3, 3] = var_0
        mu_init[j, 4] = np.random.uniform(0, 1)
        sig_init[j, 4, 4] = var_0
        mu_init[j, 5] = np.random.uniform(-1, -0.25)
        sig_init[j, 5, 5] = var_0
    # Warm up the sampler by running it for some number of iterations
    init_sampler = ais.ais(log_target=log_pi, d=dim, mu=mu_init, sig=sig_init, samp_per_prop=N_w, iter_num=I_w,
                           temporal_weights=False, weight_smoothing=True, eta_mu0=eta_loc, eta_sig0=eta_scale,
                           optimizer='Constant')
    # Run sampler with initialized parameters
    output = ais.ais(log_target=log_pi, d=dim, mu=init_sampler.means[-D:], sig=init_sampler.covariances[-D:],
                     samp_per_prop=N, iter_num=I, weight_smoothing=True, temporal_weights=True, eta_mu0=eta_loc,
                     eta_sig0=eta_scale, optimizer='RMSprop')
    # Use sampling importance resampling to extract posterior samples
    theta = ais.importance_resampling(output.particles, output.log_weights, num_samp=1000)
    # Apply transformations to the samples
    theta[:, 0] = 1/(1+np.exp(-theta[:, 0]))
    theta[:, 1] = 1/(1+np.exp(-theta[:, 1]))
    theta[:, 3] = np.exp(theta[:, 3])
    theta[:, 4] = np.exp(theta[:, 4])
    theta[:, 5] = np.exp(theta[:, 5])
    # Matrix plot of the approximated target distribution
    plt.figure()
    count = 1
    for i in range(dim):
        for j in range(dim):
            plt.subplot(dim, dim, count)
            if i != j:
                sns.kdeplot(theta[:, j], theta[:, i], cmap="Blues", shade=True, shade_lowest=False)
            else:
                plt.hist(theta[:, i])
            count += 1
    plt.show()



