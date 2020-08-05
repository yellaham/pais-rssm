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


def brspf(data, regimes, x_init, N=100):
    """
    Implementation of the bootstrap regime switching particle filter
    :param data: [][] numpy array containing observations to be processed. Missing data is denoted by 'nan'.
    :param regimes: object containing the candidate models for the regime switching
    :param x_init: Initial particles used for propagation
    :return ...
    """
    return 0


