import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
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
param = np.array([0.35, 0.875, -1, -0.1, 0.1, 0.15, 0.01, 0.01])
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
time_generate = 100

# Generate ground truth for the regime switching system
y, x, m_idx = model.generate_data(init_state=x_init, T=time_generate)

# Cutoff the first 30 time points
cut_off = 70
time_length = time_generate-cut_off
y = y[-time_length:]
x = x[-(time_length+1):]
m_idx = m_idx[-time_length:]

# Plot the generated observations
plt.figure()
plt.plot(y[:, 0])
plt.plot(y[:, 1])
plt.legend(['Observed sum of adults', 'Observed sum of chicks'])
plt.show()

# See if you can successfully track the latent  states
# Run the Bootstrap regime switching filter to estimate the latent state trajectory
num_particles = 250
initial_particles = np.random.randint(low=500, high=2000, size=(2*num_stages-2, num_particles))
output = pf.brspf(y, model, initial_particles)

# Print log evidence
print('Log evidence is %.4f' %output.log_evidence)

# Draw a sample from posterior
x_gen, m_gen = output.generate_state_trajectory()


# Plot the model selection accuracy
plt.figure()
plt.scatter(np.arange(1, time_length+1), m_idx, s=100)
plt.scatter(np.arange(1, time_length+1), output.model_estimates, s=40)
plt.ylim([-0.5, 3])
plt.title('Model Selection Performance', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('Model Index', fontsize=16)
plt.legend(['Ground Truth', 'Detected Model'])
plt.show()



# # Set up a penguin model to test other stuff
# penguin_model = penguins.AgeStructuredModel(psi_juv=param[0], psi_adu=param[1], alpha_r=param[2], beta_r=param[4],
#                                                 var_s=param[6], var_c=param[7], nstage=num_stages)
#
# # Set a state value for testing
# xold = np.random.randint(low=500, high=2000, size=(8, 50))
# xnew = penguin_model.transition_rand(xold)
#
# # Print generated state
# print(xnew)
#
# # Evaluate log pdf
# print(penguin_model.transition_log_pdf(xnew, xold))
#
# # Generate observations
# y = penguin_model.observation_rand(xnew[:, 0])
#
# # Print the generated observations
# print(y)
#
# # Evaluate the logarithm of the observation distribution
# print(penguin_model.obsevation_log_pdf(y, xnew))