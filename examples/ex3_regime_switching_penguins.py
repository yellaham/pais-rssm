import numpy as np
from applications import penguins
from monte_carlo_tools import pf

# Create an object for a new penguin model
penguin_model = penguins.AgeStructuredModel(psi_juv=0.4, psi_adu=0.9, alpha_r=0, beta_r=0.25, var_s=0.01, var_c=0.01,
                                            nstage=5)
# Set a state value for testing
xold = np.random.random_integers(low=500, high=2000, size=(50, 8))
xnew = penguin_model.transition_rand(xold)

# Print generated state
print(xnew)

# Evaluate log pdf
print(penguin_model.transition_log_pdf(xnew, xold))