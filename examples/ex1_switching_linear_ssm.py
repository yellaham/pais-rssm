import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from monte_carlo_tools import pf


# Generate linear Gaussian random number
def linear_gaussian_rand(x, slope, bias, err): return np.random.normal(slope*x+bias, err)


# Compute linear Gaussian log pdf
def linear_gaussian_log_pdf(x1, x2, slope, bias, err): return sp.norm.logpdf(x1, x2*slope+bias, err)


# Time length for the generated time-series
time_length = 40

# Different parameters for the candidate models
a = [0.9, 0.9, 0.9]
b = [-10,  0.,  10]

# Define the number of models
K = len(a)

# Initialize the candidate state space models and store into a list
candidate_models = []
for i in range(K):
    # Create a linear state-space model
    candidate_models.append(pf.SSM(lambda x, i=i: linear_gaussian_rand(x, a[i], 0, 1),
                                   lambda x1, x2, i=i: linear_gaussian_log_pdf(x1, x2, a[i], 0, 1),
                                   lambda x, i=i: linear_gaussian_rand(x, 1, b[i], 1),
                                   lambda y, x, i=i: linear_gaussian_log_pdf(y, x, 1, b[i], 1)))

# Define the regime dynamics
regime_dynamics_rand = lambda model_idx, num_samp: np.random.choice(np.arange(start=0, stop=K), num_samp,
                                                                    replace=True, p=np.ones(K)/K)
regime_dynamics_log_pdf = lambda model_idx: np.log(1/K)

# Create a multiple regime SSM and generate synthetic data
model = pf.MultiRegimeSSM(candidate_models, regime_dynamics_rand, regime_dynamics_log_pdf)
y, x, m_idx = model.generate_data(init_state=np.array([0.25]), T=time_length)

# Plot the generated observations
plt.figure()
plt.plot(np.arange(1, time_length+1), y)
plt.show()

# Run the Bootstrap regime switching filter to estimate the latent state trajectory
num_particles = 2000
initial_particles = np.random.uniform(-0.5, 0.5, size=(1, num_particles))
output = pf.brspf(y, model, initial_particles)

# Plot the tracking of the latent states
plt.figure()
plt.plot(np.arange(0, time_length+1), x)
plt.plot(np.arange(1, time_length+1), np.mean(output.particles, axis=2))
plt.title('Tracking Results', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('x(t)', fontsize=16)
plt.legend(['Ground Truth', 'Estimate'])
plt.show()


# Plot the model selection accuracy
plt.figure()
plt.scatter(np.arange(1, time_length+1), m_idx, s=100)
plt.scatter(np.arange(1, time_length+1), sp.mode(output.model_indexes.T)[0].squeeze(), s=40)
plt.ylim([-0.5, 3])
plt.title('Model Selection Performance', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('Model Index', fontsize=16)
plt.legend(['Ground Truth', 'Detected Model'])
plt.show()
