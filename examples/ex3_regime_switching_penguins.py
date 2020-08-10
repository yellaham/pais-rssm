import numpy as np
import matplotlib.pyplot as plt
from applications import penguins
from monte_carlo_tools import pf

# Parameters of the true model
#   -param[0] - juvenile survival
#   -param[1] - adult survival
#   -param[2] - reproductive success bias (bad year)
#   -param[3] - reproductive success bias (good year)
#   -param[4] - reproductive success slope
#   -param[5] - probability of bad year
#   -param[6] - variance of observations (breeders)
#   -param[7] - variance of observations (chicks)
param = np.array([0.45, 0.85, -0.5, 0.5, 0.1, 0.1, 0.01, 0.01])
# Number of stages to use for model
num_stages = 5

# Define the list of candidate models
candidate_models = [penguins.AgeStructuredModel(psi_juv=param[0], psi_adu=param[1], alpha_r=param[2], beta_r=param[4],
                                                var_s=param[6], var_c=param[7], nstage=num_stages),
                    penguins.AgeStructuredModel(psi_juv=param[0], psi_adu=param[1], alpha_r=param[3], beta_r=param[4],
                                                var_s=param[6], var_c=param[7], nstage=num_stages)]

# Define the regime dynamics
regime_dynamics_rand = lambda model_idx, num_samp: np.random.choice(np.arange(start=0, stop=2), num_samp,
                                                                    replace=True, p=np.array([param[5], 1-param[5]]))
regime_dynamics_log_pdf = lambda model_idx: model_idx*np.log(1-param[5])+(1-model_idx)*np.log(1-param[5])


# Create a multiple regime SSM and generate synthetic data
model = pf.MultiRegimeSSM(candidate_models, regime_dynamics_rand, regime_dynamics_log_pdf)

# Determine initial state
x_init = np.random.randint(low=500, high=2000, size=2*num_stages-2)

# Determine length of time to generate data for
time_length = 30

# Generate ground truth for the regime switching system
y, x, m_idx = model.generate_data(init_state=x_init, T=time_length)

# Plot the generated observations
plt.figure()
plt.plot(y)
plt.show()


# Set up a penguin model to test other stuff
penguin_model = penguins.AgeStructuredModel(psi_juv=param[0], psi_adu=param[1], alpha_r=param[2], beta_r=param[4],
                                                var_s=param[6], var_c=param[7], nstage=num_stages)

# Set a state value for testing
xold = np.random.randint(low=500, high=2000, size=(8, 50))
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