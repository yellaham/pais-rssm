import numpy as np
from applications import penguins
from monte_carlo_tools import pf

# Create an object for a new penguin model
penguin_model = penguins.AgeStructuredModel(psi_juv=0.4, psi_adu=0.9, alpha_r=0, beta_r=0.25, var_s=0.01, var_c=0.01,
                                            nstage=5)
# Set a state value for testing
xtest = 1000*np.ones((1, 8))
print(penguin_model.transition_rand(xtest))