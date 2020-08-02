import numpy as np
from scipy.stats import multivariate_normal as mvn
from monte_carlo_tools import psis


class Sampler:
    """
    Class for adaptive importance sampling methods. The main attributes of the class are the particles (or samples) and
    their corresponding importance weights (transformed to the log domain).
    """
    def __init__(self, x, log_w, mu, sig, z_est, mu_est):
        self.particles = x
        self.log_weights = log_w
        self.evidence = z_est
        self.target_mean = mu_est
        self.means = mu
        self.covariances = sig


def ais(log_target, d, mu, sig, samp_per_prop=100, iter_num=100, eta_mu0=1e-1, eta_sig0=1e-4, g_mu_max=1, g_sig_max=1):
    """
    Runs the adaptive population importance sampling algorithm
    :param log_target: Logarithm of the target distribution
    :param d: Dimension of the sampling space
    :param mu: [num_prop][d] array which stores the initial means of the proposal distributions
    :param sig: [num_prop][d][d] array which stores the initial covariances of the proposal distributions
    :param samp_per_prop: Number of samples to draw per proposal distribution (DxN total samples drawn per iteration)
    :param iter_num: Number of iterations
    :param eta_mu0: learning rate for the means in the stochastic gradient updates
    :param eta_sig0: learning rate for the covariances in the stochastic gradient updates
    :param g_mu_max: maximum value of the norm (L2) for the gradient w.r.t. the mean of each proposal
    :param g_sig_max: maximum value of the norm (Frobenius) for the gradient w.r.t. the covariance of each proposal
    :return Sampler object 
    """
    # Determine number of proposals
    num_prop = np.shape(mu)[0]

    # Determine the total number of particles
    samp_num = num_prop*samp_per_prop

    # Initialize storage of particles and log weights
    particles = np.zeros((samp_num * iter_num, d))
    log_target_eval = np.ones(samp_num * iter_num) * (-np.inf)
    log_weights = np.ones(samp_num * iter_num) * (-np.inf)
    means = np.zeros((num_prop * (iter_num + 1), d))
    covariances = np.tile(np.zeros((d, d)), (num_prop * (iter_num + 1), 1, 1))

    # Set initial locations to be the parents
    means[0:num_prop] = mu
    covariances[0:num_prop] = sig

    # Initialize storage of evidence and target mean estimates
    evidence = np.zeros(iter_num)
    target_mean = np.zeros((iter_num, d))

    # For the optimizer
    vmu = np.zeros((num_prop, d))
    vsig = np.zeros((num_prop, d, d))

    # Initialize start counter
    start = 0
    startd = 0

    # Loop for the algorithm
    for i in range(iter_num):
        # Update start counter
        stop = start + samp_num
        stopd = startd + num_prop

        # Generate particles
        idx = np.repeat(np.arange(0, num_prop), samp_per_prop)
        children = np.random.multivariate_normal(np.zeros(d), np.eye(d), samp_num)
        for j in range(num_prop):
            children[idx == j] = mu[j] + np.matmul(children[idx == j], np.linalg.cholesky(sig[j]).T)
        particles[start:stop] = children

        # Compute log proposal
        prop_j = np.zeros((samp_num * (i + 1), num_prop * (i + 1)))
        log_prop_j = np.copy(prop_j)
        prop = np.zeros(samp_num * (i + 1))
        for j in range(num_prop * (i + 1)):
            prop_j[:, j] = mvn.pdf(particles[0:stop], mean=means[j], cov=covariances[j], allow_singular=True)
            log_prop_j[:, j] = mvn.logpdf(particles[0:stop], mean=means[j], cov=covariances[j], allow_singular=True)
            prop += prop_j[:, j] / (num_prop * (i + 1))
        log_prop = np.log(prop)

        # Compute log weights and store
        log_target_eval[start:stop] = log_target(children)
        log_weights[0:stop] = log_target_eval[0:stop] - log_prop

        # Smoothing of the importance weights (in log domain)
        lws, kss = psis.psislw(log_weights[0:stop])
        elbo = np.mean(log_weights[0:stop])

        # Obtain the unnormalized importance weights
        max_log_weight = np.max(lws)
        weights = np.exp(lws - max_log_weight)

        # Compute ESS
        weights_norm = weights / np.sum(weights)
        ess = np.sum(weights_norm ** 2) ** (-1)

        # Estimate the evidence
        log_z = np.log(1 / (samp_num * (i + 1))) + max_log_weight + np.log(np.sum(weights))
        evidence[i] = np.exp(log_z)

        # Print out stuff
        print("ITER = %d, ESS = %.3f, K_HAT = %.3f, ELBO = %.3f, log_Z = %.5f" % (i+1, ess, kss, elbo, log_z))

        # Compute estimate of the target mean
        target_mean[i] = np.average(particles[0:stop, :], axis=0, weights=weights)

        # Adapt the parameters of the proposal distribution
        start_j = np.copy(start)
        for j in range(num_prop):
            # Update stop parameter
            stop_j = start_j + samp_per_prop
            # Get local children
            x_j = particles[start_j:stop_j]
            # Obtain the proposal ratio using the evaluate log weights
            prop_dm = np.zeros(samp_per_prop)
            for jj in range(num_prop):
                prop_dm += (1/num_prop)*prop_j[start_j:stop_j, jj + num_prop*i]
            q_ratio = prop_j[start_j:stop_j, j + num_prop*i]/prop_dm
            # Compute DM weights
            log_w = log_target_eval[start_j:stop_j]-np.log(prop_dm)
            w = np.exp(log_w - np.max(log_w))
            # Obtain local log weights
            log_wj = log_target_eval[start_j:stop_j] - log_prop_j[start_j:stop_j, j + num_prop*i]
            # log_wj, temp = psis.psislw(log_wj)
            # Convert to weights using LSE
            wj = np.exp(log_wj - np.max(log_wj))
            # Normalize the weights
            wjn = wj / np.sum(wj)
            "COMPUTATION OF GRADIENTS"
            # g_mu = (mu[j] - np.average(x_j, axis=0, weights=wjn))
            # g_sig = (sig[j] - np.cov(x_j, rowvar=False, bias=True, aweights=wjn))
            # Compute the local ESS
            wjtn = wjn
            ESS = np.sum(wjtn**2)**(-1)
            T_s = 1
            while ESS < np.round(0.1 * samp_per_prop):
                T_s += 0.5
                log_wjt = (1/T_s)*log_wj
                wjt = np.exp(log_wjt -np.max(log_wjt))
                wjtn = wjt/np.sum(wjt)
                ESS = np.sum(wjtn**2)**(-1)
            # g_sig = (sig[j] - np.cov(x_j, rowvar=False, bias=True, aweights=wjtn))
            g_mu = np.zeros(d)
            g_sig = np.zeros((d, d))
            # Compute inverse of the covariance matrix
            prec_j = np.linalg.inv(sig[j])
            for n in range(np.shape(x_j)[0]):
                # Compute ds_dmu and ds_dsig
                ds_dmu = -(1/num_prop)*(w[n]**1)*q_ratio[n]*np.matmul((x_j[n] - mu[j]), prec_j)
                ds_dsig = (0.5/num_prop)*(w[n]**1)*q_ratio[n]*(prec_j-np.dot(np.dot(prec_j,
                                                                        np.outer(x_j[n]-mu[j], x_j[n]-mu[j])), prec_j))
                # Compute gradients based on ds
                g_mu = g_mu + wjn[n] * ds_dmu
                g_sig = g_sig + wjtn[n] * ds_dsig
            "CLIP THE GRADIENTS"
            if np.linalg.norm(g_mu) > g_mu_max:
                g_mu = g_mu * (g_mu_max / np.linalg.norm(g_mu))
            if np.linalg.norm(g_sig) > g_sig_max:
                g_sig = g_sig * (g_sig_max / np.linalg.norm(g_sig))
            " ADAPTATION OF MEAN AND PRECISION "
            # Compute the square of the gradient
            g_mu_sq = g_mu ** 2
            g_sig_sq = g_sig ** 2
            # Update the learning rate parameters
            vmu[j] = 0.9 * vmu[j] + 0.1 * g_mu_sq
            vsig[j] = 0.9 * vsig[j] + 0.1 * g_sig_sq
            # Compute the learning rates
            eta_mu = eta_mu0 * ((np.sqrt(vmu[j]) + 1e-1) ** (-1))
            eta_sig = eta_sig0 * ((np.sqrt(vsig[j]) + 1e-1) ** (-1))
            # Update the proposal parameters
            mu[j] = mu[j] - eta_mu*g_mu
            sig[j] = sig[j] - eta_sig*g_sig
            # Force the covariance to be PD
            eig_val = np.linalg.eigvals(sig[j])
            if ~np.all(eig_val > 0):
                # Force all eigenvalues to be positive
                eig_val = np.maximum(eig_val, 1e-6)
                # Obtain eigenvectors
                eig_vec = np.linalg.eig(sig[j])[1]
                # Project onto the PSD cone
                sig[j] = np.matmul(np.matmul(eig_vec, np.diag(eig_val)), eig_vec.T)
            # Update start parameter
            start_j = stop_j

        # Store parameters
        means[startd + num_prop: stopd + num_prop] = mu
        covariances[startd + num_prop: stopd + num_prop] = sig

        # Update start counters
        start = stop
        startd = stopd

    # Generate output
    return Sampler(particles, lws, means, covariances, evidence, target_mean)