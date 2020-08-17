import numpy as np
import scipy.stats as sp

class SSM:
    """
    A class designed for state-space models.
    * Attributes
        - transition distribution (random number generation)
        - transition distribution (logarithm of the pdf)
        - observation distribution (random number generation)
        - observation distribution (logarithm of the pdf)
    """
    def __init__(self, trn_rand, trn_pdf, obs_rand, obs_pdf):
        self.transition_rand = trn_rand
        self.transition_log_pdf = trn_pdf
        self.observation_rand = obs_rand
        self.observation_log_pdf = obs_pdf

    def generate_data(self, init_state, T):
        """
        Generate synthetic data using the specified state-space model"
        :param T: Length of the generated time-series
        :return: observations [T][dim_y], latent_states [T+1][dim_x]
        """
        # Determine the dimension of the states
        dim_x = np.shape(init_state)[0]
        # Determine the dimension of the observations
        dim_y = np.shape(self.observation_rand(init_state))[0]
        # Allocate arrays for the outputs
        observations = np.zeros((T, dim_y))
        latent_states = np.zeros((T+1, dim_x))
        # Initialize the latent states
        latent_states[0] = init_state
        # Generate the data in a loop
        for t in range(T):
            # Draw a sample from the transition distribution (conditioned on the drawn model index)
            latent_states[t+1] = self.transition_rand(latent_states[t])
            # Draw a sample from the observation distribution (conditioned on the drawn model index)
            observations[t] = self.observation_rand(latent_states[t+1])
        return observations, latent_states


class MultiRegimeSSM:
    """
    A class for regime switching state-space models. The attributes of the class include a list of different candidate
    state-space models, as well as the distribution defining the dynamics of that regime-switching state-space model.
    The pdf of the distribution defining the dynamics is also an attribute of the class object.
    """
    def __init__(self, regime_list, switch_rand, switch_lpdf):
        self.regimes = regime_list
        self.switching_dynamics_rand = switch_rand
        self.switching_dynamics_lpdf = switch_lpdf

    def generate_data(self, init_state, T):
        """
        Generate synthetic data using the specified state-space model"
        :param T: Length of the generated time-series
        :return: observations [T][dim_y], latent_states [T+1][dim_x], model_indexes = [T]
        """
        # Determine the dimension of the states
        dim_x = np.shape(init_state)[0]
        # Determine the dimension of the observations
        dim_y = np.shape(self.regimes[0].observation_rand(init_state))[0]
        # Allocate arrays for the outputs
        observations = np.zeros((T, dim_y))
        latent_states = np.zeros((T+1, dim_x))
        model_indexes = np.zeros(T, dtype='int')
        # Initialize the latent states
        latent_states[0] = init_state
        # Initialize empty list for model history
        model_hist = []
        # Generate the data in a loop
        for t in range(T):
            # Draw a model index
            model_indexes[t] = self.switching_dynamics_rand(model_hist, 1)
            model_hist.append(model_indexes[t])
            # Draw a sample from the transition distribution (conditioned on the drawn model index)
            latent_states[t+1] = self.regimes[model_indexes[t]].transition_rand(latent_states[t])
            # Draw a sample from the observation distribution (conditioned on the drawn model index)
            observations[t] = self.regimes[model_indexes[t]].observation_rand(latent_states[t+1])

        return observations, latent_states, model_indexes


class RegimeSwitchingParticleFilter:
    """
    A class for outputs to regime-switching particle filters.
    """
    def __init__(self, x, m_idx, log_w, log_Z):
        self.particles = x
        self.model_indexes = m_idx
        self.log_weights = log_w
        self.log_evidence = log_Z
        # Compute the normalized weights
        weights = np.exp(self.log_weights[-1]-np.max(self.log_weights[-1]))
        self.normalized_weights = weights/np.sum(weights)
        # Estimate the states
        self.state_estimates = np.average(x, axis=2)
        # Estimate the trajectory of models
        self.model_estimates = sp.mode(m_idx.T)[0].squeeze()

    def generate_state_trajectory(self, num_samples=1):
        # Multinomial resampling according to normalized weights of particle streams
        idx = np.random.choice(np.shape(self.particles)[2], num_samples, replace=True, p=self.normalized_weights)
        # Obtained unweighted samples from referenced target distribution
        x_sample = self.particles[:, :, idx]
        m_sample = self.model_indexes[:, idx]
        return x_sample, m_sample


def brspf(data, model, x_init):
    """
    Implementation of the bootstrap regime switching particle filter
    :param data: [T][dim_y] numpy array containing observations to be processed. Missing data is denoted by 'nan'.
    :param model: object containing the candidate models for the regime switching
    :param x_init: [dim_x][N] Initial particles used for propagation, where N is the number of particles
    :return ...
    """
    # Determine relevant dimensions
    dim_y = np.shape(data)[1]   # Dimension of each observation vector
    dim_x = np.shape(x_init)[0]  # Dimension of each state vector

    # Determine the time horizon
    T = np.shape(data)[0]

    # Determine the number of particles in the algorithm
    N = np.shape(x_init)[1]

    # Determine the number of models
    K = len(model.regimes)

    # Memory allocation for numpy arrays
    x = np.zeros((T, dim_x, N))
    m_idx = np.zeros((T, N), dtype='int')
    log_w = np.zeros((T, N))

    for t in range(T):
        # Step 1: Determine previous state and generate model indexes
        if t != 0:
            # Determine previous state
            x_old = x[t-1]
            # Generate model indexes
            m_idx[t] = model.switching_dynamics_rand(m_idx[:t-1], N)
        else:
            # Determine previous state
            x_old = x_init
            # Generate model indexes
            m_idx[t] = model.switching_dynamics_rand(np.array([]), N)

        # Step 2: Generate particles based on generated model indexes and compute log_likelihood
        log_likelihood = np.zeros(N)
        for k in range(K):
            # Slicing indexes to determine which particles are aligned with model k
            idx = (m_idx[t] == k)
            # Draw particles from the kth transition distribution
            children = model.regimes[k].transition_rand(x_old[:, idx])
            # Store particles accordingly
            x[t, :, idx] = children.T
            # TODO: Need to account for the fact that there can be missing data
            # Compute log-likelihood and store accordingly
            log_likelihood[idx] = model.regimes[k].observation_log_pdf(data[t], children).flatten()

        # Step 3: Compute the importance weights (in log domain)
        if t == 0:
            log_w[t] = log_likelihood
        else:
            log_w[t] = log_likelihood

        if np.isnan(np.max(log_w[t])):
            return RegimeSwitchingParticleFilter(x, m_idx, log_w, -np.inf)

        # Step 4: Normalize the weights
        w_t = np.exp(log_w[t]-np.max(log_w[t]))
        w_t_n = w_t/np.sum(w_t)

        # Step 5: Multinomial resampling to avoid particle degeneracy
        idx_rs = np.random.choice(np.arange(0, N), N, replace=True, p=w_t_n)
        x[t] = x[t, :, idx_rs].T
        m_idx[t] = m_idx[t, idx_rs]

        # Step 6: Correct the logarithm of the importance weights since full resampling is done
        #log_z_est = np.max(log_w[t]) + np.log(np.mean(w_t))
        #log_w[t] = np.ones(N)*log_z_est

    # Compute estimate of log evidence
    max_log_w = np.array([np.max(log_w, axis=1)]).T
    log_Z = np.sum(np.log(np.mean(np.exp(log_w - max_log_w), axis=1))+max_log_w.squeeze())

    return RegimeSwitchingParticleFilter(x, m_idx, log_w, log_Z)


