class SSM:
    """
    A class designed specifically for state-space models.
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