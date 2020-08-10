import numpy as np
import scipy.stats as sp

class AgeStructuredModel:
    """
    A class which contains all necessary methods for analyzing an age-structured model for penguin colonies. Objects are
    initialized by the number of assumed adult stages and the demographic parameters.
    """
    def __init__(self, psi_juv, psi_adu, alpha_r, beta_r, var_s, var_c, nstage=5):
        self.juvenile_survival = psi_juv
        self.adult_survival = psi_adu
        self.reproductive_success_bias = alpha_r
        self.reproductive_success_slope = beta_r
        self.variance_adults = var_s
        self.variance_chicks = var_c
        self.num_stages = nstage
        logit_rates = self.reproductive_success_bias+self.reproductive_success_slope*np.linspace(0, self.num_stages-1,
                                                                                              self.num_stages)
        self.reproductive_rates = 1./(1.+np.exp(-logit_rates))

    def transition_rand(self, x_old):
        """
        Propagates penguin populations from previous year using stage-structured dynamics.
        :param x_old: Latent penguin populations from the previous year
            - x_old[:, :num_stages] references the stage 1 to stage J adult penguins
            - x_old[:, -num_stages-2:] references the stage 1 to stage J-2 chicks
        :return x
        """
        if len(np.shape(x_old)) == 1:
            x = np.zeros(2*self.num_stages-2)
        else:
            # Determine the number of samples
            num_samples = np.shape(x_old)[1]
            # Set up matrix to output everything
            x = np.zeros((2*self.num_stages-2, num_samples))
        # Obtain reproductive rate in real space by applying sigmoid transformation
        pr = self.reproductive_rates
        # Compute the total number of chicks
        ct_old = np.sum(x_old[-(self.num_stages-2):], axis=0)
        # From total number of chicks to state 1 adults
        x[0] = np.array(np.random.binomial((ct_old/2).astype(int), self.juvenile_survival)).flatten()
        # Remainder of cycle
        for j in range(self.num_stages-1):
            # Propagate adults first
            if j < self.num_stages-2:
                x[j+1] = np.random.binomial(x_old[j].astype(int), self.adult_survival)
            else:
                x[j+1] = np.random.binomial((x_old[j] + x_old[j+1]).astype(int), self.adult_survival)
            # Obtain the chicks for the penguins that can breed
            if j >= 1:
                # Chicks obtained = binomial draw
                x[self.num_stages+j-1] = np.random.binomial(2*x[j+1].astype(int), pr[j-1])
        return x

    def transition_log_pdf(self, x, x_old):
        """
        Evaluate the logarithm of the transition distribution for the age-structured penguin model.
        :param x, x_old: Latent penguin populations (current year and previous year)
            - x[:, :num_stages] references the stage 1 to stage J adult penguins
            - x[:, -num_stages-2:] references the stage 1 to stage J-2 chicks
        :return logarithm of the transition distribution
        """
        if len(np.shape(x_old)) == 1:
            x = np.zeros(2*self.num_stages-2)
            log_transition = np.zeros(1)
        else:
            # Determine the number of samples
            num_samples = np.shape(x_old)[1]
            log_transition = np.zeros(num_samples)
        # Compute the total number of chicks
        ct_old = np.sum(x_old[-(self.num_stages-2):], axis=0)
        # From total number of chicks to state 1 adults
        log_transition += sp.binom.logpmf(x[0], ct_old, p=self.juvenile_survival)
        # Remainder of cycle
        for j in range(self.num_stages-1):
            # Propagate adults first
            if j < self.num_stages-2:
                log_transition += sp.binom.logpmf(x[j+1], x_old[j].astype(int), p=self.adult_survival)
            else:
                log_transition += sp.binom.logpmf(x[j+1], (x_old[j] + x_old[j+1]).astype(int), p=self.adult_survival)
            # Obtain the chicks for the penguins that can breed
            if j >= 1:
                log_transition += sp.binom.logpmf(x[self.num_stages+j-1], 2*x[j+1].astype(int),
                                                  p=self.reproductive_rates[j-1])
        return log_transition

    def observation_rand(self, x):
        """
        Generates noisy observations for latent penguin populations
        :param x: Latent penguin populations of the current year
            - x[:, :num_stages] references the stage 1 to stage J adult penguins
            - x[:, -num_stages-2:] references the stage 1 to stage J-2 chicks
        :return observed number of total breeders and observed number of total chicks
        NOTE: Need to keep in mind that I can have numerical errors due to the dimension of the array being used
        """
        # Determine the number of samples and allocate array for sample generation
        if len(np.shape(x)) == 1:
            y = np.zeros(2)
        else:
            num_samples = np.shape(x)[1]
            y = np.zeros((2, num_samples))
        # Extract the total number of breeders and chicks
        st = np.sum(x[self.num_stages:], axis=0)         # total number of breeders
        ct = np.sum(x[-(self.num_stages-2):], axis=0)    # total number of chicks
        # Generate observations
        y[0] = np.random.normal(loc=st, scale=np.sqrt(self.variance_adults)*st)
        y[1] = np.random.normal(loc=ct, scale=np.sqrt(self.variance_chicks)*ct)
        return y.astype(int)

    def observation_log_pdf(self, y, x):
        """
        Evaluate the logarithm of the observation distribution
        :param y Observed total breeders and total chciks
        :param x: Latent penguin populations of the current year
            - x[:, :num_stages] references the stage 1 to stage J adult penguins
            - x[:, -num_stages-2:] references the stage 1 to stage J-2 chicks
        :return logarithm of the observation distribution
        """
        # Determine the number of samples and allocate array for sample generation
        if len(np.shape(x)) == 1:
            log_observation = np.zeros(1)
        else:
            num_samples = np.shape(x)[1]
            log_observation = np.zeros(num_samples)
        # Extract the total number of breeders and chicks
        st = np.sum(x[self.num_stages:], axis=0)            # total number of breeders
        ct = np.sum(x[-(self.num_stages - 2):], axis=0)     # total number of chicks
        # Generate observations
        log_observation += sp.norm.logpdf(y[0], loc=st, scale=np.sqrt(self.variance_adults)*st)
        log_observation += sp.norm.logpdf(y[1], loc=ct, scale=np.sqrt(self.variance_chicks)*ct)
        return log_observation
