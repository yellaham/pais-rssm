import numpy as np


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
    # TODO: Create an attribute for synthetic data generation
    #


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
    # TODO: Create an attribute for synthetic data generation
    #


def brspf(data, regimes, x_init):
    """
    Implementation of the bootstrap regime switching particle filter
    :param data: [dim_y][T] numpy array containing observations to be processed. Missing data is denoted by 'nan'.
    :param regimes: object containing the candidate models for the regime switching
    :param x_init: [dim_x][N] Initial particles used for propagation, where N is the number of particles
    :return ...
    """
    # Determine relevant dimensions
    dim_y = np.shape(data)[0]   # Dimension of each observation vector
    dim_x = np.shape(x_init)[0]  # Dimension of each state vector

    # Determine the time horizon
    T = np.shape(data)[1]

    # Determine the number of particles in the algorithm
    N = np.shape(x_init)[1]

    # Determine the number of models
    K = len(regimes)

    # Memory allocation for numpy arrays
    x = np.zeros((T, dim_x, N))
    m_idx = np.zeros((T, N))

    for t in range(T):
        # Step 1: Determine previous state
        if t == 0:
            x_old = x[t-1]
        else:
            x_old = x_init

        # Step 2: Generate model indexes
        m_idx[t] = regimes.switching_dynamics_rand(N)

        # Step 3: Generate particles based on generated model indexes
        for k in range(K):
            print('help')

    return 0


