import numpy as np
from applications import penguins
from monte_carlo_tools import pf

# Create an object for a new penguin model
penguin_model = penguins.AgeStructuredModel(psi_juv=0.4, psi_adu=0.9, alpha_r=0, beta_r=0.25, var_s=0.01, var_c=0.01,
                                            nstage=5)
# Set a state value for testing
xold = np.random.randint(low=500, high=2000, size=(1, 8))
xnew = penguin_model.transition_rand(xold)

# Print generated state
print(xnew)

# Evaluate log pdf
print(penguin_model.transition_log_pdf(xnew, xold))

# Generate observations
y = penguin_model.observation_rand(np.array([xnew[0]]))

# Print the generated observations
print(y)

# Evaluate the logarithm of the observation distribution
print(penguin_model.obsevation_log_pdf(y, xnew))