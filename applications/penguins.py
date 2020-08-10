import numpy as np

class AgeStructuredModel:
    """
    A class which contains all necessary methods for analyzing an age-structured model for penguin colonies. Objects are
    initialized by the number of assumed adult stages and the demographic parameters.
    """
    def __init__(self, psi_juv, psi_adu, alpha_r, beta_r, var_s, var_c, J=5):
        self.juvenile_survival = psi_juv
        self.adult_survival = psi_adu
        self.reproductive_success_bias = alpha_r
        self.reproductive_success_slope = beta_r
        self.variance_adults = var_s
        self.variance_chicks = var_c
        self.num_stages = J

    def transition_rand(self, x_old):
        """
        Propagates penguin populations from previous year using stage-structured dynamics.
        :param x_old:
            - x[:, :J] indicate the stage 1 to stage J adult penguins
            - x[:, -J-2:] indicate the stage 1 to stage J-2 chicks
        :return an object with attributes S, Sb, C
        """
        # Figure out how many things we are propagating
        N = np.shape(x_old)[0]
        # Set up matrix to output everything
        x = np.zeros((N, 2*self.num_stages-2))
        # Compute the reproductive rate for each stage in logit space
        logit_mu = self.reproductive_success_bias+self.reproductive_success_slope*np.linspace(0, self.num_stages-1,
                                                                                              self.num_stages)
        # Obtain reproductive rate in real space by applying sigmoid transformation
        pr = 1./(1.+np.exp(logit_mu))
        # Compute the total number of chicks
        Ct_old = np.sum(x_old[:, -self.num_stages:2], axis=1)
        # From total number of chicks to state 1 adults
        x[:, 0] = np.array(np.random.binomial((Ct_old/2).astype(int), self.juvenile_survival)).flatten()
        # Remainder of cycle
        for j in range(self.num_stages-1):
            # Propagate adults first
            if j < self.num_stages-2:
                x[:, j+1] = np.random.binomial(x_old[:, j].astype(int), self.adult_survival).flatten()
            else:
                x[:, j+1] = np.random.binomial((x_old[:, j] + x_old[:, j+1]).astype(int), self.adult_survival).flatten()
            # Obtain the chicks for the penguins that can breed
            if j >= 1:
                # Chicks obtained = binomial draw
                x[:, -(self.num_stages+j-1)] = np.random.binomial(x[:, j+1].astype(int), pr[j-1]).flatten()
        return x
        ## TODO: We need to write the relevant methods of this class
        #   - Transition distribution (log pdf)
        #   - Observation distribution (random number generator)
        #   - Observation distribution (log pdf)
