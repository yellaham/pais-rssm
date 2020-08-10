class AgeStructuredModel:
    """
    A class which contains all necessary methods for analyzing an age-structured model for penguin colonies. Objects are
    initialized by the number of assumed adult stages and the demographic parameters.
    """
    def __init__(self, psi_juv, psi_adu, alpha_r, beta_r, var_s, var_c, J=5):
        self.juvenile_survivorship = psi_juv
        self.adult_survivorship = psi_adu
        self.reproductive_rate_bias = alpha_r
        self.reproductive_rate_slope = beta_r
        self.variance_adults = var_s
        self.variance_chicks = var_c
        self.num_stages = J
    ## TODO: We need to write the relevant methods of this class
    #   - Transition distribution (random number generator)
    #   - Transition distribution (log pdf)
    #   - Observation distribution (random number generator)
    #   - Observation distribution (log pdf)