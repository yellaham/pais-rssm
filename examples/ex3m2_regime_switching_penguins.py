import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import multiprocessing
import seaborn as sns
from applications import penguins
from monte_carlo_tools import pf
from monte_carlo_tools import ais


## DEFINITIONS
def log_jacobian_sigmoid(x): return -x-2*np.log(1+np.exp(-x))


# PART 1: LOAD DATA
with np.load('simulated_data.npz') as data:
    param = data['param']
    num_stages = data['num_stages']
    x_init = data['x_init']
    x = data['x']
    y = data['y']
    m_idx = data['m_idx']

# PART 2: ASSUMED MODEL
def log_likelihood_per_sample(input_parameters):
    # Set the random seed
    np.random.seed()
    # Define the number of particles
    num_particles = 2000
    # Apply relevant transformations to the sample (sigmoid transformation to probability parameters)
    z = np.copy(input_parameters)
    z[0] = 1/(1+np.exp(-z[0]))
    z[1] = np.exp(z[1])
    psi_juv_big = 1/(1+np.exp(-(input_parameters[0]+z[1])))
    z[2] = 1/(1+np.exp(-z[2]))
    z[4] = np.exp(z[4])
    z[5] = 1/(1+np.exp(-z[5]))
    # Evaluate prior distribution at transformed samples (don't forget to factor in Jacobian from transformation)
    log_prior = sp.beta.logpdf(z[0], 3, 3)+log_jacobian_sigmoid(input_parameters[0])
    log_prior += sp.gamma.logpdf(z[1], 0.001, 0.001)+input_parameters[1]
    log_prior += sp.beta.logpdf(z[2], 3, 3)+log_jacobian_sigmoid(input_parameters[2])
    log_prior += sp.norm.logpdf(z[3], 0, 1)
    log_prior += sp.norm.logpdf(z[4], 0, 0.1)+input_parameters[4]
    log_prior += sp.beta.logpdf(z[5], 1, 9)+log_jacobian_sigmoid(input_parameters[5])
    # Create the model (assuming the noise variances are known)
    regimes = [penguins.AgeStructuredModel(psi_juv=z[0], psi_adu=z[2], alpha_r=z[3], beta_r=z[4], var_s=param[6],
                                          var_c=param[7], nstage=num_stages),
               penguins.AgeStructuredModel(psi_juv=psi_juv_big, psi_adu=z[2], alpha_r=z[3], beta_r=z[4], var_s=param[6],
                                    var_c=param[7], nstage=num_stages)]
    draw_regimes = lambda model_idx, num_samp: np.random.choice(np.arange(start=0, stop=2), num_samp, replace=True,
                                                                        p=np.array([z[5], 1 - z[5]]))
    regimes_log_pdf = lambda model_idx: model_idx*np.log(1-z[5])+(1-model_idx)*np.log(z[5])
    # Create regime switching system
    model = pf.MultiRegimeSSM(regimes, draw_regimes, regimes_log_pdf)
    # Draw the initial particles
    x_init = np.array([x[0]]).T+np.random.randint(low=-10, high=10, size=(2*num_stages-2, num_particles))
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
    N = 200  # number of samples per proposal
    I = 50  # number of iterations
    N_w = 50  # number of samples per proposal (warm-up period)
    I_w = 200   # number of iterations (warm-up period)
    D = 7      # number of proposals
    var_0 = 1e-1   # initial variance
    eta_loc = 5e-2  # learning rate for the mean
    eta_scale = 5e-2    # learning rate for the covariance matrix
    # Select initial proposal parameters
    mu_init = np.zeros((D, dim))
    sig_init = np.zeros((D, dim, dim))
    for j in range(D):
        mu_init[j, 0] = np.random.uniform(-2, -0.5)
        sig_init[j, 0, 0] = var_0
        mu_init[j, 1] = np.random.uniform(-0.5, 0)
        sig_init[j, 1, 1] = var_0
        mu_init[j, 2] = np.random.uniform(1, 2)
        sig_init[j, 2, 2] = var_0
        mu_init[j, 3] = np.random.uniform(-0.5, 0)
        sig_init[j, 3, 3] = var_0
        mu_init[j, 4] = np.random.uniform(-2, 0)
        sig_init[j, 4, 4] = var_0
        mu_init[j, 5] = np.random.uniform(-2, -0.5)
        sig_init[j, 5, 5] = var_0
    # Warm up the sampler by running it for some number of iterations
    init_sampler = ais.ais(log_target=log_pi, d=dim, mu=mu_init, sig=sig_init, samp_per_prop=N_w, iter_num=I_w,
                           temporal_weights=False, weight_smoothing=True, eta_mu0=eta_loc, eta_sig0=eta_scale,
                           criterion='Moment Matching', optimizer='Constant')
    # Run sampler with initialized parameters
    output = ais.ais(log_target=log_pi, d=dim, mu=init_sampler.means[-D:], sig=init_sampler.covariances[-D:],
                     samp_per_prop=N, iter_num=I, weight_smoothing=True, temporal_weights=True, eta_mu0=0.05*eta_loc,
                     eta_sig0=0.1*eta_scale, criterion='Minimum Variance', optimizer='RMSprop')
    # Use sampling importance resampling to extract posterior samples
    theta = ais.importance_resampling(output.particles, output.log_weights, num_samp=1000)
    # Apply transformations to the samples
    theta[:, 0] = 1/(1+np.exp(-theta[:, 0]))
    theta[:, 1] = 1/(1+np.exp(-theta[:, 1]))
    theta[:, 3] = theta[:, 2] + np.exp(theta[:, 3])
    theta[:, 4] = np.exp(theta[:, 4])
    theta[:, 5] = 1/(1+np.exp(-theta[:, 5]))
    # Create labels for each parameter
    labels = ['Juvenile Survival', 'Adult Survival', 'Logit Intercept (Bad)', 'Logit Intercept (Good)', 'Logit Slope',
              'Probability of Bad Year']
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
    # Plot all of the one-dimensional histograms
    for i in range(dim):
        plt.figure()
        plt.hist(theta[:, i])
        plt.axvline(param[i])
        plt.xlabel(labels[i])
        plt.show()
    # # Plot all of the two-dimensional histograms
    # for i in range(dim):
    #     for j in range(dim):
    #         if i != j:
    #             plt.figure()
    #             sns.kdeplot(theta[:, i], theta[:, j], cmap="Blues", shade=True, shade_lowest=False)
    #             plt.xlabel(labels[i])
    #             plt.ylabel(labels[j])
    #             plt.show()


